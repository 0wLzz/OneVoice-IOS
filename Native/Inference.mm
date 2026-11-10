#import "Inference.h"

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <SherpaOnnxC/sherpa-onnx/c-api/c-api.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {
using IDs = std::vector<int64_t>;
using Tensors = std::map<std::string, Ort::Value>;
using Inputs = std::map<std::string, const OrtValue *>;

std::string file(NSURL *directory, NSString *name) {
    NSURL *url = [directory URLByAppendingPathComponent:name];
    if (![[NSFileManager defaultManager] fileExistsAtPath:url.path]) {
        throw std::runtime_error("Missing bundled resource: " + std::string(name.UTF8String));
    }
    return url.fileSystemRepresentation;
}

id json(NSURL *directory, NSString *name) {
    NSURL *url = [directory URLByAppendingPathComponent:name];
    NSData *data = [NSData dataWithContentsOfURL:url];
    id result = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (!result) {
        throw std::runtime_error("Cannot read " + std::string(name.UTF8String));
    }
    return result;
}

NSString *trim(NSString *text) {
    return [text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
}

NSString *unicode(const std::string &value) {
    return [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding];
}

class Graph {
    Ort::Session session{nullptr};
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

public:
    Graph(NSURL *directory, NSString *name, int threads = 2,
          GraphOptimizationLevel optimization = ORT_ENABLE_EXTENDED) {
        static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "OneVoice");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.DisableCpuMemArena();
        options.DisableMemPattern();
        options.SetGraphOptimizationLevel(optimization);
        session = Ort::Session(environment, file(directory, name).c_str(), options);
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t index = 0; index < session.GetInputCount(); ++index) {
            inputNames.emplace_back(session.GetInputNameAllocated(index, allocator).get());
        }
        for (size_t index = 0; index < session.GetOutputCount(); ++index) {
            outputNames.emplace_back(session.GetOutputNameAllocated(index, allocator).get());
        }
    }

    Tensors run(const Inputs &inputs) {
        std::vector<const char *> names;
        std::vector<const OrtValue *> values;
        for (const auto &name : inputNames) {
            names.push_back(name.c_str());
            values.push_back(inputs.at(name));
        }
        std::vector<const char *> requested;
        for (const auto &name : outputNames) requested.push_back(name.c_str());
        std::vector<OrtValue *> outputs(requested.size(), nullptr);
        Ort::Status status(Ort::GetApi().Run(session, nullptr, names.data(), values.data(), names.size(),
                                            requested.data(), requested.size(), outputs.data()));
        Tensors result;
        for (size_t i = 0; i < outputs.size(); ++i) {
            result.emplace(outputNames[i], Ort::Value(outputs[i]));
        }
        Ort::ThrowOnError(status.release());
        return result;
    }
};

template <typename T>
Ort::Value tensor(std::vector<T> &data) {
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape[] = {1, static_cast<int64_t>(data.size())};
    return Ort::Value::CreateTensor<T>(memory, data.data(), data.size(), shape, 2);
}

using Blocked = std::unordered_set<int64_t>;

Blocked tokenSet(NSArray<NSNumber *> *numbers) {
    Blocked result;
    for (NSNumber *number in numbers) result.insert(number.longLongValue);
    return result;
}

int64_t bestToken(const Ort::Value &logits, const Blocked &suppressed, const Blocked &extra) {
    const auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[1] < 1 || shape[2] < 1) {
        throw std::runtime_error("Invalid decoder logits");
    }
    const auto width = shape[2];
    const float *scores = logits.GetTensorData<float>() + (shape[1] - 1) * width;
    int64_t best = -1;
    for (int64_t token = 0; token < width; ++token) {
        if (suppressed.count(token) || extra.count(token) || !std::isfinite(scores[token])) continue;
        if (best < 0 || scores[token] > scores[best]) best = token;
    }
    if (best < 0) throw std::runtime_error("Decoder returned no usable token");
    return best;
}

Blocked repeatedTrigrams(const IDs &history) {
    Blocked result;
    if (history.size() < 2) return result;
    const size_t last = history.size() - 2;
    for (size_t i = 0; i + 2 < history.size(); ++i) {
        if (history[i] == history[last] && history[i + 1] == history[last + 1]) {
            result.insert(history[i + 2]);
        }
    }
    return result;
}

class DecoderCache {
    Tensors past;

public:
    Tensors step(Graph &graph, IDs &ids, Inputs inputs) {
        auto sequence = tensor(ids);
        inputs.emplace("input_ids", sequence);
        for (const auto &[name, value] : past) inputs.emplace(name, value);
        auto outputs = graph.run(inputs);
        for (auto &[name, value] : outputs) {
            if (name.rfind("present.", 0) == 0) {
                past.insert_or_assign("past_key_values." + name.substr(8), std::move(value));
            }
        }
        return outputs;
    }
};

NSString *zipformer(std::vector<float> pcm, NSURL *directory) {
    const auto encoder = file(directory, @"encoder.onnx");
    const auto decoder = file(directory, @"decoder.onnx");
    const auto joiner = file(directory, @"joiner.onnx");
    const auto tokens = file(directory, @"tokens.txt");
    SherpaOnnxOfflineRecognizerConfig config{};
    config.feat_config.sample_rate = 16000;
    config.feat_config.feature_dim = 80;
    config.model_config.transducer.encoder = encoder.c_str();
    config.model_config.transducer.decoder = decoder.c_str();
    config.model_config.transducer.joiner = joiner.c_str();
    config.model_config.tokens = tokens.c_str();
    config.model_config.num_threads = 4;
    config.model_config.provider = "cpu";
    config.decoding_method = "modified_beam_search";
    config.max_active_paths = 8;

    std::unique_ptr<const SherpaOnnxOfflineRecognizer, decltype(&SherpaOnnxDestroyOfflineRecognizer)>
        recognizer(SherpaOnnxCreateOfflineRecognizer(&config), SherpaOnnxDestroyOfflineRecognizer);
    if (!recognizer) throw std::runtime_error("Could not load Korean Zipformer");
    std::unique_ptr<const SherpaOnnxOfflineStream, decltype(&SherpaOnnxDestroyOfflineStream)>
        stream(SherpaOnnxCreateOfflineStream(recognizer.get()), SherpaOnnxDestroyOfflineStream);
    if (!stream) throw std::runtime_error("Could not open Korean audio stream");
    pcm.resize(pcm.size() + 20480, 0);
    SherpaOnnxAcceptWaveformOffline(stream.get(), 16000, pcm.data(), static_cast<int32_t>(pcm.size()));
    SherpaOnnxDecodeOfflineStream(recognizer.get(), stream.get());
    std::unique_ptr<const SherpaOnnxOfflineRecognizerResult, decltype(&SherpaOnnxDestroyOfflineRecognizerResult)>
        result(SherpaOnnxGetOfflineStreamResult(stream.get()), SherpaOnnxDestroyOfflineRecognizerResult);
    if (!result) throw std::runtime_error("Korean recognizer returned no result");
    NSMutableString *text = [NSMutableString string];
    for (int32_t i = 0; i < result->count; ++i) [text appendString:@(result->tokens_arr[i])];
    return trim([text stringByReplacingOccurrencesOfString:@"▁" withString:@" "]);
}

NSString *whisperText(const IDs &ids, NSURL *directory) {
    NSDictionary *tokenizer = json(directory, @"tokenizer.json");
    NSDictionary<NSString *, NSNumber *> *vocab = tokenizer[@"model"][@"vocab"];
    NSMutableDictionary<NSNumber *, NSString *> *pieces = [NSMutableDictionary dictionaryWithCapacity:vocab.count];
    [vocab enumerateKeysAndObjectsUsingBlock:^(NSString *piece, NSNumber *index, BOOL *stop) {
        pieces[index] = piece;
    }];
    // Whisper's byte-level BPE stores non-printable bytes as U+0100 and above.
    std::map<unichar, uint8_t> bytes;
    unichar next = 256;
    for (int value = 0; value < 256; ++value) {
        bool printable = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || value >= 174;
        bytes[printable ? value : next++] = value;
    }
    NSMutableData *decoded = [NSMutableData data];
    for (int64_t id : ids) {
        if (id >= 50257) continue;
        NSString *piece = pieces[@(id)];
        for (NSUInteger index = 0; index < piece.length; ++index) {
            uint8_t byte = bytes.at([piece characterAtIndex:index]);
            [decoded appendBytes:&byte length:1];
        }
    }
    return trim([[NSString alloc] initWithData:decoded encoding:NSUTF8StringEncoding] ?: @"");
}

NSString *whisper(std::vector<float> pcm, NSURL *directory) {
    pcm.resize(480000, 0);
    Ort::Value features{nullptr};
    {
        Graph frontend(directory, @"log_mel.onnx");
        auto audio = tensor(pcm);
        auto outputs = frontend.run({{"audio", audio}});
        features = std::move(outputs.at("input_features"));
    }
    Ort::Value hidden{nullptr};
    {
        Graph encoder(directory, @"encoder_model.onnx");
        auto outputs = encoder.run({{"input_features", features}});
        hidden = std::move(outputs.at("last_hidden_state"));
    }
    features = Ort::Value(nullptr);
    NSDictionary *config = json(directory, @"generation_config.json");
    IDs prompt = {
        [config[@"decoder_start_token_id"] longLongValue],
        [config[@"lang_to_id"][@"<|vi|>"] longLongValue],
        [config[@"task_to_id"][@"transcribe"] longLongValue],
        [config[@"no_timestamps_token_id"] longLongValue],
    };
    const int64_t eos = [config[@"eos_token_id"] longLongValue];
    const auto suppress = tokenSet(config[@"suppress_tokens"]);
    DecoderCache cache;
    IDs output;
    {
        Graph decoder(directory, @"decoder_model.onnx");
        auto first = cache.step(decoder, prompt, {{"encoder_hidden_states", hidden}});
        output.push_back(bestToken(first.at("logits"), suppress, tokenSet(config[@"begin_suppress_tokens"])));
    }
    hidden = Ort::Value(nullptr);
    if (output.back() != eos) {
        Graph decoder(directory, @"decoder_with_past_model.onnx");
        while (output.size() < 220 && output.back() != eos) {
            IDs current{output.back()};
            auto step = cache.step(decoder, current, {});
            output.push_back(bestToken(step.at("logits"), suppress, {}));
        }
    }
    return whisperText(output, directory);
}

void loadPieces(sentencepiece::SentencePieceProcessor &processor, NSURL *directory) {
    const auto status = processor.Load(file(directory, @"sentencepiece.bpe.model"));
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

NSString *contextTag(NSString *text, NSURL *directory) {
    NSRegularExpression *punctuation = [NSRegularExpression regularExpressionWithPattern:@"[!?.,;:~…\"'()\\[\\]]+"
                                                                                   options:0 error:nil];
    NSString *normalized = trim([[punctuation stringByReplacingMatchesInString:text options:0
                                                                         range:NSMakeRange(0, text.length)
                                                                  withTemplate:@" "] lowercaseString]);
    sentencepiece::SentencePieceProcessor processor;
    loadPieces(processor, directory);
    NSArray<NSNumber *> *lookup = json(directory, @"piece_ids.json");
    IDs ids{0};
    for (int piece : processor.EncodeAsIds(normalized.UTF8String)) {
        if (ids.size() == 63) break;
        ids.push_back(lookup[piece].longLongValue);
    }
    ids.push_back(2);
    IDs mask(ids.size(), 1);
    ids.resize(64, 1);
    mask.resize(64, 0);
    IDs type(64, 0);
    auto idsTensor = tensor(ids);
    auto maskTensor = tensor(mask);
    auto typeTensor = tensor(type);
    Graph model(directory, @"model.onnx");
    auto result = model.run({{"input_ids", idsTensor}, {"attention_mask", maskTensor}, {"token_type_ids", typeTensor}});
    const auto *scores = result.at("logits").GetTensorData<float>();
    return @[@"SOS", @"BUSY", @"ERROR", @"ROUTINE"][std::max_element(scores, scores + 4) - scores];
}

NSString *translation(NSString *text, NSString *source, NSString *tag, NSURL *directory) {
    NSDictionary *config = json(directory, @"mobile_config.json");
    NSDictionary *model = json(directory, @"config.json");
    NSDictionary<NSString *, NSNumber *> *vocab = json(directory, @"vocab.json");
    NSDictionary *tags = json(directory, @"tags.json");
    NSNumber *tagID = tags[[NSString stringWithFormat:@"[%@]", tag]];
    if (!tagID) throw std::runtime_error("Unrecognized translation context tag");
    const int64_t eos = [model[@"eos_token_id"] longLongValue];
    const int64_t start = [model[@"decoder_start_token_id"] longLongValue];
    const int64_t targetID = [config[@"languages"][[source isEqualToString:@"vi"] ? @"ko" : @"vi"] longLongValue];
    sentencepiece::SentencePieceProcessor processor;
    loadPieces(processor, directory);

    IDs sourceIDs{[config[@"languages"][source] longLongValue], tagID.longLongValue};
    for (const auto &piece : processor.EncodeAsPieces(text.UTF8String)) {
        NSNumber *id = vocab[unicode(piece)] ?: vocab[@"<unk>"];
        sourceIDs.push_back(id.longLongValue);
    }
    sourceIDs.push_back(eos);
    if (sourceIDs.size() > [model[@"max_position_embeddings"] unsignedLongValue]) {
        throw std::runtime_error("Transcript is too long for translation. Try a shorter recording.");
    }
    // Run the released INT8 graphs as exported for consistent Mac/iOS decoding.
    constexpr auto optimization = ORT_DISABLE_ALL;
    IDs mask(sourceIDs.size(), 1);
    auto input = tensor(sourceIDs);
    auto attention = tensor(mask);
    Ort::Value hidden{nullptr};
    {
        Graph encoder(directory, @"encoder_model_quantized.onnx", 1, optimization);
        auto outputs = encoder.run({{"input_ids", input}, {"attention_mask", attention}});
        hidden = std::move(outputs.at("last_hidden_state"));
    }
    DecoderCache cache;
    IDs generated{start, targetID};
    {
        Graph decoder(directory, @"decoder_model_quantized.onnx", 1, optimization);
        IDs first{start};
        cache.step(decoder, first, {{"encoder_hidden_states", hidden}, {"encoder_attention_mask", attention}});
    }
    hidden = Ort::Value(nullptr);
    {
        Graph decoder(directory, @"decoder_with_past_model_quantized.onnx", 1, optimization);
        const auto suppressed = tokenSet(config[@"suppressed"]);
        const auto limit = std::min<size_t>(256, 2 * sourceIDs.size() + 10);
        while (generated.size() - 1 < limit && generated.back() != eos) {
            IDs current{generated.back()};
            auto step = cache.step(decoder, current, {{"encoder_attention_mask", attention}});
            generated.push_back(bestToken(step.at("logits"), suppressed, repeatedTrigrams(generated)));
        }
    }
    auto special = tokenSet(config[@"language_ids"]);
    special.insert({0, 1, 2, 3});
    NSMutableDictionary<NSNumber *, NSString *> *reverse = [NSMutableDictionary dictionaryWithCapacity:vocab.count];
    [vocab enumerateKeysAndObjectsUsingBlock:^(NSString *piece, NSNumber *id, BOOL *stop) {
        reverse[id] = piece;
    }];
    std::vector<std::string> pieces;
    for (int64_t id : generated) {
        if (special.count(id)) continue;
        NSString *piece = reverse[@(id)];
        if (piece) pieces.emplace_back(piece.UTF8String);
    }
    return trim(unicode(processor.DecodePieces(pieces)));
}

template <typename Operation>
NSString *perform(NSError **error, Operation operation) {
    try {
        return operation();
    } catch (const std::exception &failure) {
        if (error) {
            *error = [NSError errorWithDomain:@"OneVoice.Inference" code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @(failure.what())}];
        }
        return nil;
    }
}
} // namespace

@implementation OVInference
+ (NSString *)transcribe:(NSData *)pcm language:(NSString *)language resources:(NSURL *)resources error:(NSError **)error {
    return perform(error, [&] {
        if (!pcm.length || pcm.length % sizeof(float) || pcm.length > 480000 * sizeof(float)) {
            throw std::runtime_error("Audio must contain 1 to 480,000 mono float samples");
        }
        const auto *first = static_cast<const float *>(pcm.bytes);
        std::vector<float> samples(first, first + pcm.length / sizeof(float));
        if ([language isEqualToString:@"ko"]) {
            return zipformer(std::move(samples), [resources URLByAppendingPathComponent:@"Zipformer"]);
        }
        return whisper(std::move(samples), [resources URLByAppendingPathComponent:@"Whisper"]);
    });
}

+ (NSString *)tag:(NSString *)transcript resources:(NSURL *)resources error:(NSError **)error {
    return perform(error, [&] { return contextTag(transcript, [resources URLByAppendingPathComponent:@"Tagger"]); });
}

+ (NSString *)translate:(NSString *)transcript language:(NSString *)language tag:(NSString *)tag
              resources:(NSURL *)resources error:(NSError **)error {
    return perform(error, [&] {
        return translation(transcript, language, tag, [resources URLByAppendingPathComponent:@"Translation"]);
    });
}
@end
