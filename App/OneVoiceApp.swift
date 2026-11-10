import AVFoundation
import Darwin
import SwiftUI

enum Direction: String, CaseIterable, Identifiable, Sendable {
    case vietnamese = "vi"
    case korean = "ko"

    var id: String { rawValue }
    var label: String {
        switch self {
        case .vietnamese: "Vietnamese → Korean"
        case .korean: "Korean → Vietnamese"
        }
    }

    var speechModel: String {
        switch self {
        case .vietnamese: "Whisper (Vietnamese)"
        case .korean: "Zipformer (Korean)"
        }
    }
}

struct TranslationResult: Sendable {
    let transcript: String
    let tag: String
    let translation: String
    let performance: [StagePerformance]
}

struct StagePerformance: Identifiable, Sendable {
    let id: String
    let seconds: Double
    let peakRAM: UInt64?

    var duration: String {
        seconds.formatted(.number.precision(.fractionLength(2))) + " s"
    }

    var memory: String {
        guard let peakRAM else { return "RAM unavailable" }
        let mib = Double(peakRAM) / 1_048_576
        return mib.formatted(.number.precision(.fractionLength(0))) + " MiB RAM"
    }
}

private final class RAMSampler: @unchecked Sendable {
    private let queue = DispatchQueue(label: "OneVoiceAI.RAMSampler")
    private var peak: UInt64?

    // The queue owns peak. Read the task's physical footprint, which includes native inference allocations.
    private func sample() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if status == KERN_SUCCESS, info.phys_footprint > 0 {
            peak = max(peak ?? 0, info.phys_footprint)
        }
    }

    static func measure<T>(_ model: String, operation: () throws -> T) rethrows -> (T, StagePerformance) {
        let sampler = RAMSampler()
        sampler.queue.sync { sampler.sample() }
        let timer = DispatchSource.makeTimerSource(queue: sampler.queue)
        timer.setEventHandler { sampler.sample() }
        timer.schedule(deadline: .now() + .milliseconds(20), repeating: .milliseconds(20), leeway: .milliseconds(5))
        timer.resume()
        defer { timer.cancel() }
        let start = DispatchTime.now()
        let result = try operation()
        let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
        let peakRAM = sampler.queue.sync {
            sampler.sample()
            return sampler.peak
        }
        return (result, StagePerformance(id: model, seconds: Double(elapsed) / 1_000_000_000, peakRAM: peakRAM))
    }
}

enum AppError: LocalizedError {
    case microphoneDenied
    case recordingFailed
    case interrupted
    case invalidAudio
    case missingResources
    case emptyTranscript

    var errorDescription: String? {
        switch self {
        case .microphoneDenied: "Enable microphone access in Settings to record. You can still try a sample."
        case .recordingFailed: "The recording could not be saved. Try recording again."
        case .interrupted: "Recording was interrupted. Try again when the microphone is available."
        case .invalidAudio: "The audio is empty or could not be read as 16 kHz mono audio."
        case .missingResources: "Model assets are missing. Run iOS/prepare.sh on the Mac and rebuild."
        case .emptyTranscript: "No speech was detected. Try a clearer recording."
        }
    }
}

enum Pipeline {
    static func resourceDirectory() throws -> URL {
        guard let url = Bundle.main.url(forResource: "ModelAssets", withExtension: nil),
              FileManager.default.fileExists(atPath: url.appendingPathComponent("checksums.json").path) else {
            throw AppError.missingResources
        }
        return url
    }

    static func readPCM(_ url: URL) throws -> Data {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.processingFormat.sampleRate == 16_000,
              file.processingFormat.channelCount == 1,
              file.length > 0,
              let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(min(file.length, 480_000))
              ) else {
            throw AppError.invalidAudio
        }
        try file.read(into: buffer)
        guard let samples = buffer.floatChannelData?[0], buffer.frameLength > 0 else {
            throw AppError.invalidAudio
        }
        return Data(bytes: samples, count: Int(buffer.frameLength) * MemoryLayout<Float>.stride)
    }

    static func run(
        audio: URL,
        direction: Direction,
        progress: @Sendable (String) async -> Void
    ) async throws -> TranslationResult {
        let directory = try resourceDirectory()
        let pcm = try readPCM(audio)
        await progress("Transcribing…")
        let (transcript, speech) = try RAMSampler.measure(direction.speechModel) {
            try autoreleasepool {
                try OVInference.transcribe(pcm, language: direction.rawValue, resources: directory)
            }
        }
        guard !transcript.isEmpty else { throw AppError.emptyTranscript }
        await progress("Identifying context…")
        let (tag, context) = try RAMSampler.measure("MiniLM") {
            try autoreleasepool { try OVInference.tag(transcript, resources: directory) }
        }
        await progress("Translating…")
        let (translation, mt) = try RAMSampler.measure("M2M v8 INT8") {
            try autoreleasepool {
                try OVInference.translate(transcript, language: direction.rawValue, tag: tag, resources: directory)
            }
        }
        return TranslationResult(transcript: transcript, tag: tag, translation: translation,
                                 performance: [speech, context, mt])
    }
}

@MainActor
final class TranslationViewModel: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var direction = Direction.vietnamese {
        didSet {
            if direction != oldValue {
                result = nil
                error = nil
            }
        }
    }
    @Published private(set) var isRecording = false
    @Published private(set) var isWorking = false
    @Published private(set) var status: String?
    @Published private(set) var error: String?
    @Published private(set) var result: TranslationResult?

    private var recorder: AVAudioRecorder?

    func recordOrStop() {
        if let recorder {
            finishRecording(recorder, success: true)
        } else if !isWorking {
            startRecording()
        }
    }

    func sample() {
        guard !isRecording, !isWorking else { return }
        do {
            let url = try Pipeline.resourceDirectory()
                .appendingPathComponent("Samples/\(direction.rawValue)_sample.wav")
            run(url, temporary: false)
        } catch {
            self.error = error.localizedDescription
        }
    }

    func interruptRecording() {
        if let recorder {
            finishRecording(recorder, success: false)
            error = AppError.interrupted.localizedDescription
        }
    }

    private func startRecording() {
        error = nil
        isWorking = true
        status = "Requesting microphone access…"
        Task {
            defer {
                isWorking = false
                status = nil
            }
            guard await AVAudioApplication.requestRecordPermission() else {
                error = AppError.microphoneDenied.localizedDescription
                return
            }
            let session = AVAudioSession.sharedInstance()
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".caf")
            do {
                try session.setCategory(.record, mode: .default)
                try session.setActive(true)
                let recording = try AVAudioRecorder(url: url, settings: [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: 16_000,
                    AVNumberOfChannelsKey: 1,
                    AVLinearPCMBitDepthKey: 16,
                    AVLinearPCMIsFloatKey: false,
                ])
                recording.delegate = self
                guard recording.record(forDuration: 30) else { throw AppError.recordingFailed }
                recorder = recording
                isRecording = true
                result = nil
            } catch {
                try? session.setActive(false, options: .notifyOthersOnDeactivation)
                try? FileManager.default.removeItem(at: url)
                self.error = error.localizedDescription
            }
        }
    }

    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let id = ObjectIdentifier(recorder)
        Task { @MainActor in
            guard let current = self.recorder, ObjectIdentifier(current) == id else { return }
            finishRecording(current, success: flag)
        }
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: (any Error)?) {
        audioRecorderDidFinishRecording(recorder, successfully: false)
    }

    private func finishRecording(_ recording: AVAudioRecorder, success: Bool) {
        recorder = nil
        isRecording = false
        recording.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        if success {
            run(recording.url, temporary: true)
        } else {
            try? FileManager.default.removeItem(at: recording.url)
            error = AppError.recordingFailed.localizedDescription
        }
    }

    private func run(_ url: URL, temporary: Bool) {
        let selectedDirection = direction
        error = nil
        result = nil
        isWorking = true
        status = "Preparing audio…"
        Task {
            defer {
                isWorking = false
                status = nil
                if temporary { try? FileManager.default.removeItem(at: url) }
            }
            do {
                result = try await Task.detached(priority: .userInitiated) { [self] in
                    try await Pipeline.run(audio: url, direction: selectedDirection) { stage in
                        await self.updateStatus(stage)
                    }
                }.value
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    private func updateStatus(_ stage: String) {
        status = stage
    }
}

@main
struct OneVoiceApp: App {
    var body: some Scene {
        WindowGroup { TranslationScreen() }
    }
}

struct TranslationScreen: View {
    @StateObject private var model = TranslationViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            List {
                Section("Direction") {
                    Picker("Direction", selection: $model.direction) {
                        ForEach(Direction.allCases) { direction in
                            Text(direction.label).tag(direction)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(model.isWorking || model.isRecording)
                }

                Section("Audio") {
                    Button(action: model.recordOrStop) {
                        Label(model.isRecording ? "Stop" : "Record", systemImage: model.isRecording ? "stop.circle" : "mic")
                    }
                    .disabled(model.isWorking)
                    Button(action: model.sample) {
                        Label("Try Sample", systemImage: "play.circle")
                    }
                    .disabled(model.isWorking || model.isRecording)
                    if let status = model.status {
                        ProgressView(status)
                    } else if model.isRecording {
                        Text("Recording for up to 30 seconds…")
                            .foregroundStyle(.secondary)
                    }
                    if let error = model.error {
                        Text(error).foregroundStyle(.red)
                    }
                }

                if let result = model.result {
                    Section("Translation") {
                        Text(result.translation).textSelection(.enabled)
                    }
                    Section("Transcript") {
                        Text(result.transcript).textSelection(.enabled)
                    }
                    Section("Context tag") {
                        Text(result.tag)
                    }
                    Section {
                        ForEach(result.performance) { stage in
                            LabeledContent(stage.id) {
                                VStack(alignment: .trailing) {
                                    Text(stage.duration).monospacedDigit()
                                    Text(stage.memory).font(.caption)
                                }
                            }
                        }
                    } header: {
                        Text("Model performance")
                    } footer: {
                        Text("Time includes model loading and inference. RAM is the highest sampled app memory footprint during each stage.")
                    }
                }
            }
            .navigationTitle("OneVoiceAI")
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { model.interruptRecording() }
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.interruptionNotification)) { notice in
                if notice.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt == AVAudioSession.InterruptionType.began.rawValue {
                    model.interruptRecording()
                }
            }
        }
    }
}
