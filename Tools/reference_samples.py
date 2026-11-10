"""Print reference results for the existing WAVs using the original checkpoints."""
import json
import os
from pathlib import Path
import re

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import onnxruntime as ort
import sherpa_onnx
import soundfile
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer, M2M100Tokenizer
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "onevoice_pipeline"
MODELS = PIPE / "models"


@torch.inference_mode()
def vietnamese(audio):
    folder = MODELS / "whisper-vi-allin"
    processor = WhisperFeatureExtractor.from_pretrained(folder)
    tokenizer = WhisperTokenizerFast.from_pretrained(folder, extra_special_tokens={})
    model = WhisperForConditionalGeneration.from_pretrained(folder, dtype=torch.float32).eval()
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    generated = model.generate(features, language="vi", task="transcribe", num_beams=1, max_new_tokens=220)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def korean(audio):
    folder = MODELS / "ko-zipformer-mt/onnx"
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        **{key: str(folder / f"{key}.onnx") for key in ("encoder", "decoder", "joiner")},
        tokens=str(folder / "tokens.txt"), num_threads=4,
        decoding_method="modified_beam_search", max_active_paths=8,
    )
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, np.pad(audio, (0, 20480)))
    recognizer.decode_stream(stream)
    return "".join(stream.result.tokens).replace("▁", " ").strip()


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    samples = {}
    for lang, transcribe in (("vi", vietnamese), ("ko", korean)):
        audio, rate = soundfile.read(PIPE / f"samples/{lang}_sample.wav", dtype="float32")
        assert rate == 16000
        samples[lang] = {"transcript": transcribe(audio)}

    folder = MODELS / "viko_tagger_minilm"
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).eval()
    for sample in samples.values():
        normalized = re.sub(r"[!?.,;:~…\"'()\[\]]+", " ", sample["transcript"]).lower().strip()
        tokens = tokenizer(normalized, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        sample["tag"] = ("SOS", "BUSY", "ERROR", "ROUTINE")[model(**tokens).logits.argmax().item()]
    del model

    translate_samples(samples)
    print(json.dumps(samples, ensure_ascii=False, indent=2))


def translate_samples(samples):
    folder = ROOT / "Models/M2M_v8_Trimmed_ONNX_INT8/model"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    translator = ORTModelForSeq2SeqLM.from_pretrained(
        folder, provider="CPUExecutionProvider", session_options=options,
        encoder_file_name="encoder_model_quantized.onnx",
        decoder_file_name="decoder_model_quantized.onnx",
        decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
    )
    tokenizer = M2M100Tokenizer.from_pretrained(folder)
    tags = json.loads((folder / "tags.json").read_text())
    cjk = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
    suppressed = [i for piece, i in tokenizer.get_vocab().items() if cjk.search(piece)]
    suppressed.extend(range(min(tags.values()), translator.config.vocab_size))
    languages = set(tokenizer.lang_token_to_id.values())
    for source, sample in samples.items():
        tokenizer.src_lang = source
        target = "vi" if source == "ko" else "ko"
        ids = tokenizer(sample["transcript"])["input_ids"]
        ids.insert(1, tags[f"[{sample['tag']}]"])
        input_ids = torch.tensor([ids])
        generated = translator.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            num_beams=1, do_sample=False, no_repeat_ngram_size=3,
            max_new_tokens=min(256, 2 * len(ids) + 10),
            forced_bos_token_id=tokenizer.get_lang_id(target), suppress_tokens=suppressed,
        )
        sample["translation"] = tokenizer.decode(
            [i for i in generated[0].tolist() if i not in languages], skip_special_tokens=True,
        ).strip()


if __name__ == "__main__":
    main()
