import XCTest
@testable import OneVoiceAI

final class PipelineTests: XCTestCase {
    func testVietnameseSample() async throws {
        let result = try await translate(.vietnamese)
        XCTAssertEqual(result.transcript, "hàng đường này đang gấp em làm nhanh tay hơn một chút được không.")
        XCTAssertEqual(result.tag, "BUSY")
        XCTAssertEqual(result.translation, "이 화물이 급해요. 좀 더 빨리 할 수 있어요.")
        checkPerformance(result, speechModel: "Whisper (Vietnamese)")
    }

    func testKoreanSample() async throws {
        let result = try await translate(.korean)
        XCTAssertEqual(result.transcript, "문양이 급해서 그러는데 조금만 더 서둘러 주실래요")
        XCTAssertEqual(result.tag, "BUSY")
        XCTAssertEqual(result.translation, "Mẫu này đang bị vội vã, em xin nhanh lên một chút ạ.")
        checkPerformance(result, speechModel: "Zipformer (Korean)")
    }

    private func checkPerformance(_ result: TranslationResult, speechModel: String) {
        XCTAssertEqual(result.performance.map(\.id), [speechModel, "MiniLM", "M2M v8 INT8"])
        for stage in result.performance {
            XCTAssertGreaterThan(stage.seconds, 0, stage.id)
            XCTAssertGreaterThan(stage.peakRAM ?? 0, 0, stage.id)
        }
    }

    private func translate(_ direction: Direction) async throws -> TranslationResult {
        let audio = try Pipeline.resourceDirectory()
            .appendingPathComponent("Samples/\(direction.rawValue)_sample.wav")
        return try await Pipeline.run(audio: audio, direction: direction) { _ in }
    }
}
