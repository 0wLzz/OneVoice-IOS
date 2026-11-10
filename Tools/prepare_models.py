"""Export and stage the local checkpoints. No Hugging Face downloads are used."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import onnx
import onnxruntime as ort
import sentencepiece as spm
import torch
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer, M2M100Tokenizer, WhisperFeatureExtractor

IOS = Path(__file__).resolve().parents[1]
REPO = IOS.parent
PIPE = REPO / "onevoice_pipeline"
OUTPUT = IOS / "ModelAssets"
BUILD = IOS / ".prepare"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def stage(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha(source) != sha(destination):
        # APFS can clone files cheaply. Fall back to an ordinary copy.
        subprocess.run(["/bin/cp", "-c", str(source), str(destination)], check=True)
    if sha(source) != sha(destination):
        raise ValueError(f"Staging changed {destination}")


class LogMel(torch.nn.Module):
    """Whisper's centered, periodic-Hann STFT expressed with ONNX Conv."""
    def __init__(self, mel_filters):
        super().__init__()
        freq = 2 * torch.pi * torch.arange(201, dtype=torch.float64)[:, None] / 400
        angle = freq * torch.arange(400, dtype=torch.float64)
        window = torch.hann_window(400, dtype=torch.float64)
        self.register_buffer("kernel", torch.cat((angle.cos(), angle.sin())).mul(window).float()[:, None])
        self.register_buffer("mel", torch.from_numpy(mel_filters.T.copy()).float())

    def forward(self, audio):
        padded = torch.nn.functional.pad(audio[:, None, :], (200, 200), mode="reflect")
        bins = torch.nn.functional.conv1d(padded, self.kernel, stride=160)[..., :-1]
        power = bins[:, :201].square() + bins[:, 201:].square()
        log_mel = torch.matmul(self.mel, power).clamp(min=1e-10).log10()
        return (torch.maximum(log_mel, log_mel.amax() - 8) + 4) / 4


def export(source, name, task, extra_files=()):
    work = BUILD / name
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=True)
    signature = {path.name: sha(path) for path in sorted(source.iterdir()) if path.is_file()}
    signature["export_script"] = sha(Path(__file__))
    signature["requirements"] = sha(IOS / "Tools/requirements.txt")
    stamp = work / "source.json"
    if not stamp.exists() or json.loads(stamp.read_text()) != signature:
        if work.exists():
            shutil.rmtree(work)
        main_export(str(source), output=work, task=task, opset=17,
                    device="cpu", local_files_only=True, no_post_process=True, do_validation=False)
        write_json(stamp, signature)
    graphs = sorted(work.glob("*.onnx"))
    if not graphs:
        raise ValueError(f"No ONNX export for {name}")
    for graph in graphs:
        onnx.checker.check_model(str(graph))
        stage(graph, destination / graph.name)
        model = onnx.load(str(graph), load_external_data=False)
        for tensor in model.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    stage(work / entry.value, destination / entry.value)
    for filename in extra_files:
        stage(source / filename, destination / filename)
    return destination


def validate_pipeline_sources():
    manifest = json.loads((PIPE / "MANIFEST.json").read_text())
    prefixes = ("models/whisper-vi-allin/", "models/viko_tagger_minilm/", "models/ko-zipformer-mt/onnx/")
    selected = {"ASR_SELECTION.json", "samples/vi_sample.wav", "samples/ko_sample.wav"}
    for name, checksum in manifest.items():
        if (name in selected or name.startswith(prefixes)) and sha(PIPE / name) != checksum:
            raise ValueError(f"Source does not match onevoice_pipeline/MANIFEST.json: {name}")


def main():
    validate_pipeline_sources()
    torch.set_num_threads(2)
    models = PIPE / "models"
    whisper_src = models / "whisper-vi-allin"
    whisper = export(whisper_src, "Whisper", "automatic-speech-recognition-with-past",
                     ("tokenizer.json", "generation_config.json"))
    processor = WhisperFeatureExtractor.from_pretrained(whisper_src)
    audio = torch.zeros(1, 480000)
    frontend = whisper / "log_mel.onnx"
    torch.onnx.export(LogMel(processor.mel_filters), (audio,), frontend, opset_version=17,
                      input_names=["audio"], output_names=["input_features"], dynamo=False)
    import soundfile as sf
    sample, rate = sf.read(PIPE / "samples/vi_sample.wav", dtype="float32")
    expected = processor(sample, sampling_rate=rate, return_tensors="np").input_features
    padded = np.pad(sample, (0, 480000 - len(sample)))[None, :]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    actual = ort.InferenceSession(frontend, options, providers=["CPUExecutionProvider"]).run(None, {"audio": padded})[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.001)

    tag_src = models / "viko_tagger_minilm"
    tagger = export(tag_src, "Tagger", "text-classification", ("sentencepiece.bpe.model",))
    tokenizer = AutoTokenizer.from_pretrained(tag_src)
    sp = spm.SentencePieceProcessor(model_file=str(tag_src / "sentencepiece.bpe.model"))
    lookup = tokenizer.get_vocab()
    ids = [lookup.get(sp.id_to_piece(i), 3) for i in range(sp.get_piece_size())]
    ids[sp.unk_id()] = tokenizer.unk_token_id
    write_json(tagger / "piece_ids.json", ids)

    ko = models / "ko-zipformer-mt/onnx"
    report = json.loads((ko / "export_report.json").read_text())
    selection = json.loads((PIPE / "ASR_SELECTION.json").read_text())
    if not report["exact_match"] or report["source_weights_sha256"] != selection["selected"]["ko"]["weights_sha256"]:
        raise ValueError("Korean export does not match the selected checkpoint")
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        stage(ko / name, OUTPUT / "Zipformer" / name)

    mt_src = REPO / "Models/M2M_v8_Trimmed_ONNX_INT8/model"
    mt = OUTPUT / "Translation"
    for name in ("encoder_model_quantized.onnx", "decoder_model_quantized.onnx",
                 "decoder_with_past_model_quantized.onnx", "sentencepiece.bpe.model",
                 "vocab.json", "tags.json", "trained_tags.json", "config.json",
                 "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        stage(mt_src / name, mt / name)
    tokenizer = M2M100Tokenizer.from_pretrained(mt_src)
    cjk = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
    tags = json.loads((mt / "tags.json").read_text())
    config = json.loads((mt / "config.json").read_text())
    vocab = tokenizer.get_vocab()
    suppressed = {i for p, i in vocab.items() if cjk.search(p)}
    suppressed.update(range(min(tags.values()), config["vocab_size"]))
    write_json(mt / "mobile_config.json", {
        "languages": {lang: tokenizer.get_lang_id(lang) for lang in ("vi", "ko")},
        "language_ids": sorted(tokenizer.lang_token_to_id.values()),
        "suppressed": sorted(suppressed),
    })
    for name in ("vi_sample.wav", "ko_sample.wav"):
        stage(PIPE / "samples" / name, OUTPUT / "Samples" / name)
    write_json(OUTPUT / "checksums.json", {
        str(path.relative_to(OUTPUT)): sha(path)
        for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != "checksums.json"
    })
    print("Prepared", OUTPUT)


if __name__ == "__main__":
    main()
