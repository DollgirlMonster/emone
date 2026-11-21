import Foundation
import Emone

/// `POST /v1/systemone` (Emone-lmw.5): TypeSafe's System One API over a
/// decision adapter laid on the side model's weights. A request is a state and
/// named Noul / Choice / Score questions; the answer is calibrated
/// probabilities from one pointer-head readout per question, no decoding.
/// Request rendering and tokenization are Kev's (`SystemOneRequest`), checked
/// token for token against Kev's Python by `EmoneCLI decide`.
///
/// One temperature serves every question: the adapter head's (Rev: 2.52),
/// unless the request names its own `temperature` -- a Emone extension.
/// Temperature never changes which option wins, only how sure the answer is.
public final class SystemOneService: Sendable {
    public enum Failure: Error, CustomStringConvertible {
        /// The request is malformed or does not fit (HTTP 422, as Kev answers).
        case invalid(String)

        public var description: String {
            switch self {
            case .invalid(let reason): reason
            }
        }
    }

    public let modelID: String
    public let adapterDirectory: URL
    public let baseDirectory: URL
    public let headTemperature: Float
    let host: DecisionModelHost

    public init(package: ModelCatalog.SystemOnePackage, idleTimeout: Duration?) throws {
        self.modelID = package.modelID
        self.adapterDirectory = package.adapter
        self.baseDirectory = package.base.path
        self.headTemperature = try Self.temperature(
            of: package.adapter.appendingPathComponent(LoRAAdapter.headFile))
        self.host = DecisionModelHost(
            adapter: package.adapter, base: package.base.path, idleTimeout: idleTimeout)
    }

    /// The head's shipped temperature, from its safetensors metadata alone.
    static func temperature(of headURL: URL) throws -> Float {
        let handle = try FileHandle(forReadingFrom: headURL)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw Failure.invalid("decision head is unreadable")
        }
        let length = prefix.withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
        }
        guard length > 0, length < 1 << 24,
            let header = try handle.read(upToCount: length),
            let json = try JSONSerialization.jsonObject(with: header) as? [String: Any],
            let meta = json["__metadata__"] as? [String: String],
            let t = meta["temperature"].flatMap(Float.init), t > 0
        else { throw Failure.invalid("decision head has no temperature") }
        return t
    }

    /// The response body for one request, and the request id Kev's server
    /// returns in `x-typesafe-request-id`.
    public func answer(_ body: Data) async throws -> (body: Data, requestID: String) {
        let started = ContinuousClock.now
        let request: SystemOneRequest
        do {
            request = try SystemOneRequest(json: try OrderedJSON.parse(body))
        } catch let error as OrderedJSON.ParseError {
            throw Failure.invalid(error.description)
        } catch let error as SystemOneRequest.Invalid {
            throw Failure.invalid(error.description)
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: baseDirectory)
        let (state, questions) = request.encoded { tokenizer.encode($0) }
        let scored: [DecisionModel.Answer]
        do {
            scored = try await host.score(state: state, questions: questions).answers
        } catch let error as DecisionReadoutError {
            throw Failure.invalid("\(error)")
        }
        let temperature = request.temperature ?? headTemperature
        var answers: [(String, OrderedJSON)] = []
        for (question, answer) in zip(request.questions, scored) {
            let p = SystemOneAnswers.probabilities(
                logits: answer.logits, headTemperature: headTemperature, temperature: temperature)
            answers.append((question.id, SystemOneAnswers.answer(question, p)))
        }
        let answersJSON = OrderedJSON.object(answers)
        // Kev's usage: every token the record packs (the state once, each
        // question's branch), and the tokens of the serialised answers.
        let inputTokens =
            1 + state.count
            + questions.reduce(0) { total, q in
                total + 2 + q.instruction.count + q.options.reduce(0) { $0 + $1.count + 2 }
            }
        let outputTokens = tokenizer.encode(answersJSON.pythonDumps()).count
        let elapsed = started.duration(to: .now)
        let milliseconds =
            Double(elapsed.components.seconds) * 1_000 + Double(elapsed.components.attoseconds)
            / 1e15
        let response = OrderedJSON.object([
            ("model", .string(request.model)),
            ("answers", answersJSON),
            (
                "usage",
                .object([
                    ("input_tokens", .integer("\(inputTokens)")),
                    ("output_tokens", .integer("\(outputTokens)")),
                ])
            ),
            ("latency_ms", .number((milliseconds * 10).rounded() / 10)),
            // Through the Float's own shortest digits, so 2.52 reads 2.52.
            ("temperature", .number(Double("\(temperature)") ?? Double(temperature))),
        ])
        return (Data(response.serialized().utf8), "req_" + UUID().uuidString.lowercased())
    }

    public func shutdown() async {
        await host.shutdown()
    }
}
