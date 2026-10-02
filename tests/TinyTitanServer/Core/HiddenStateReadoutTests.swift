import Foundation
import NIOCore
import Testing
import TinyTitan
import TinyTitanMemory

@testable import TinyTitanServerCore

/// `x_hidden_states`: request decoding, validation, the response encoding and
/// the HTTP surface, against scripted backends. Nothing here loads a model.
@Suite("Hidden-state readout", .serialized)
struct HiddenStateReadoutTests {
    private static let flash = ServerReasoningProfile(
        family: .qwen38flash, thinkingMode: .off, reasoningEffort: nil)

    private func decode(_ json: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
    }

    private func chat(_ extra: String) -> String {
        #"{"model":"m","messages":[{"role":"user","content":"x"}]"# + extra + "}"
    }

    private func validate(
        _ extra: String, profile: ServerReasoningProfile = flash
    ) throws -> ValidatedChatRequest {
        try OpenAIRequestValidator.validate(
            try decode(chat(extra)), modelID: "m", reasoningProfile: profile)
    }

    /// The refusal's field, so a test asserts which rule fired and not just
    /// that something threw.
    private func refusedParam(_ extra: String) -> String? {
        do {
            _ = try validate(extra)
            return nil
        } catch let ServerRequestError.invalid(_, param, _) {
            return param
        } catch {
            return "unexpected \(error)"
        }
    }

    // MARK: decoding

    @Test func absentFieldDecodesToNil() throws {
        #expect(try decode(chat("")).hiddenStates == nil)
        #expect(try decode(chat(#","x_hidden_states":null"#)).hiddenStates == nil)
        #expect(try validate("").hiddenStates == nil)
    }

    @Test func presentFieldDecodesEveryKey() throws {
        let request = try decode(
            chat(
                """
                ,"x_hidden_states":{"layers":[12,24,36,47],"stream_mode":"mean_streams",
                "prefill_only":true,"cache":"bypass","encoding":"base64_f32","positions":"last"}
                """))
        let hidden = try #require(request.hiddenStates)
        #expect(hidden.layers == [12, 24, 36, 47])
        #expect(hidden.streamMode == "mean_streams")
        #expect(hidden.prefillOnly == true)
        #expect(hidden.cache == "bypass")
        #expect(hidden.encoding == "base64_f32")
        #expect(hidden.positions == "last")
    }

    @Test func aRequestWithTheFieldRoundTripsThroughCodable() throws {
        let request = try decode(chat(#","x_hidden_states":{"layers":[1]}"#))
        let again = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: try JSONEncoder().encode(request))
        #expect(again == request)
        #expect(again.hiddenStates?.layers == [1])
    }

    // MARK: validation

    @Test func acceptsTheDocumentedRequestAndDefaultsTheRest() throws {
        let plan = try #require(
            try validate(#","x_hidden_states":{"layers":[47,12,24,36,12]}"#).hiddenStates)
        #expect(plan.layers == [12, 24, 36, 47])
        #expect(plan.streamMode == .residual)
    }

    @Test func acceptsEveryNamedValue() throws {
        let validated = try validate(
            """
            ,"x_hidden_states":{"layers":[0],"stream_mode":"mean_streams","prefill_only":true,
            "cache":"bypass","encoding":"base64_f32","positions":"last"}
            """)
        #expect(validated.hiddenStates?.streamMode == .meanStreams)
    }

    @Test func maxTokensIsIgnoredForAReadout() throws {
        let hidden = #","x_hidden_states":{"layers":[3]}"#
        #expect(try validate(#","max_tokens":0"# + hidden).hiddenStates != nil)
        #expect(try validate(#","max_completion_tokens":5"# + hidden).hiddenStates != nil)
        // Without a readout the same value is still refused.
        #expect(throws: ServerRequestError.self) { try validate(#","max_tokens":0"#) }
    }

    @Test func refusesTheLayerRules() {
        #expect(refusedParam(#","x_hidden_states":{}"#) == "x_hidden_states.layers")
        #expect(refusedParam(#","x_hidden_states":{"layers":[]}"#) == "x_hidden_states.layers")
        #expect(refusedParam(#","x_hidden_states":{"layers":[-1]}"#) == "x_hidden_states.layers")
        #expect(refusedParam(#","x_hidden_states":{"layers":[48]}"#) == "x_hidden_states.layers")
        let seventeen = (0..<17).map(String.init).joined(separator: ",")
        #expect(
            refusedParam(#","x_hidden_states":{"layers":[\#(seventeen)]}"#)
                == "x_hidden_states.layers")
        // The last valid layer of a 48-block model, and the cap exactly.
        #expect(refusedParam(#","x_hidden_states":{"layers":[47]}"#) == nil)
        let sixteen = (0..<16).map(String.init).joined(separator: ",")
        #expect(refusedParam(#","x_hidden_states":{"layers":[\#(sixteen)]}"#) == nil)
    }

    @Test func refusesTheUnsupportedModes() {
        let layers = #""layers":[1]"#
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"stream_mode":"mean"}"#)
                == "x_hidden_states.stream_mode")
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"cache":"use"}"#)
                == "x_hidden_states.cache")
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"encoding":"json"}"#)
                == "x_hidden_states.encoding")
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"positions":"all"}"#)
                == "x_hidden_states.positions")
    }

    @Test func refusesStreamingAndMoreThanOneChoice() {
        let hidden = #","x_hidden_states":{"layers":[1]}"#
        #expect(refusedParam(#","stream":true"# + hidden) == "stream")
        #expect(refusedParam(#","n":2"# + hidden) == "n")
        // stream=false is an ordinary request.
        #expect(refusedParam(#","stream":false"# + hidden) == nil)
        // A capture during a generation may stream; n is still 1.
        let capture = #","x_hidden_states":{"layers":[1],"prefill_only":false}"#
        #expect(refusedParam(#","stream":true"# + capture) == nil)
        #expect(refusedParam(#","n":2"# + capture) == "n")
    }

    // MARK: capture during a generation (prefill_only: false)

    @Test func captureDuringAGenerationDefaultsToReuseAndKeepsMaxTokens() throws {
        let validated = try validate(
            #","max_tokens":9,"x_hidden_states":{"layers":[2,1],"prefill_only":false}"#)
        let plan = try #require(validated.hiddenStates)
        #expect(plan.prefillOnly == false)
        #expect(plan.layers == [1, 2])
        #expect(validated.maximumCompletionTokens == 9)
        // max_tokens keeps its meaning: zero is still refused.
        #expect(
            refusedParam(#","max_tokens":0,"x_hidden_states":{"layers":[1],"prefill_only":false}"#)
                == "max_tokens")
        // A probe still ignores it.
        #expect(
            try validate(#","max_tokens":0,"x_hidden_states":{"layers":[1]}"#)
                .maximumCompletionTokens == 1)
    }

    @Test func theCachePairingFollowsPrefillOnly() {
        let layers = #""layers":[1]"#
        // Capture: reuse (explicit or default) only.
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"prefill_only":false,"cache":"reuse"}"#)
                == nil)
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"prefill_only":false,"cache":"bypass"}"#)
                == "x_hidden_states.cache")
        // Probe: bypass only.
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"prefill_only":true,"cache":"bypass"}"#)
                == nil)
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"prefill_only":true,"cache":"reuse"}"#)
                == "x_hidden_states.cache")
        #expect(
            refusedParam(#","x_hidden_states":{\#(layers),"cache":"reuse"}"#)
                == "x_hidden_states.cache")
    }

    @Test func aFamilyWithoutAFixedGeometryDefersTheUpperBoundToTheSession() throws {
        // Dense Qwen 3.5 takes its layer count from the install's manifest, so
        // the validator cannot bound it; the session checks the loaded model.
        let dense = ServerReasoningProfile(
            family: .qwen35Dense, thinkingMode: .off, reasoningEffort: nil)
        let validated = try validate(#","x_hidden_states":{"layers":[200]}"#, profile: dense)
        #expect(validated.hiddenStates?.layers == [200])
    }

    @Test func derivedRequestsKeepTheReadout() throws {
        let validated = try validate(#","x_hidden_states":{"layers":[7]}"#)
        let plan = try #require(validated.hiddenStates)
        #expect(validated.withModel("other").hiddenStates == plan)
        #expect(validated.withWorkspace("w").hiddenStates == plan)
        #expect(validated.replacingMessages([], tools: []).hiddenStates == plan)
    }

    // MARK: response encoding

    private func decodeFloats(_ base64: String) throws -> [Float] {
        let data = try #require(Data(base64Encoded: base64))
        #expect(data.count % 4 == 0)
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = (0..<4).reduce(UInt32(0)) { acc, byte in
                acc | UInt32(data[data.startIndex + offset + byte]) << (8 * UInt32(byte))
            }
            return Float(bitPattern: bits)
        }
    }

    @Test func base64Float32RoundTripsBitExactlyAndLittleEndian() throws {
        let values: [Float] = [0, -0.0, 1, -1.5, 3.14159, .infinity, 1e-38, 65504, -123456.78]
        let encoded = HiddenStatesPayload.base64Float32(values)
        let decoded = try decodeFloats(encoded)
        #expect(decoded.map(\.bitPattern) == values.map(\.bitPattern))
        // 1.0f is 0x3F800000: little-endian bytes 00 00 80 3F.
        let one = try #require(Data(base64Encoded: HiddenStatesPayload.base64Float32([1])))
        #expect([UInt8](one) == [0x00, 0x00, 0x80, 0x3F])
    }

    @Test func jsonObjectCarriesShapeDtypeDataAndPosition() throws {
        let wide = (0..<8).map { Float($0) * 0.5 }
        let payload = HiddenStatesPayload(
            HiddenReadoutResult(
                position: 41,
                layers: [
                    HiddenReadoutLayer(layer: 12, values: wide),
                    HiddenReadoutLayer(layer: 24, values: [1, 2]),
                ],
                captureNanos: 1_500_000))
        let object = payload.jsonObject()
        #expect(object["capture_ms"] as? Double == 1.5)
        #expect(Set(object.keys) == ["12", "24", "position", "capture_path", "capture_ms"])
        #expect(object["position"] as? Int == 41)
        let path = try #require(object["capture_path"] as? [String: Any])
        #expect(path["runner"] as? String == "plain")
        #expect(path["prefill"] as? String == "gpu")
        #expect(path["early_stop"] as? Bool == true)
        let twelve = try #require(object["12"] as? [String: Any])
        #expect(twelve["dim"] as? Int == 8)
        #expect(twelve["dtype"] as? String == "float32")
        #expect(twelve["shape"] as? [Int] == [1, 8])
        #expect(try decodeFloats(try #require(twelve["data"] as? String)) == wide)
        let twentyFour = try #require(object["24"] as? [String: Any])
        #expect(twentyFour["shape"] as? [Int] == [1, 2])
    }

    // MARK: HTTP and decorators

    private actor ReadoutBackend: ServerInferenceBackend {
        private(set) var seen: [ValidatedChatRequest] = []

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws -> ServerCompletion {
            seen.append(request)
            guard let plan = request.hiddenStates else {
                return ServerCompletion(
                    content: "hi", toolCalls: [], finishReason: "stop",
                    usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
            }
            let payload = HiddenStatesPayload(
                HiddenReadoutResult(
                    position: 41,
                    layers: plan.layers.map {
                        HiddenReadoutLayer(layer: $0, values: [Float($0), 0.25])
                    },
                    path: HiddenCapturePath(usedANE: false, earlyStop: plan.prefillOnly)))
            guard plan.prefillOnly else {
                // A capture during a generation: the reply, then the rows.
                onEvent(.content("ok"))
                return ServerCompletion(
                    content: "ok", toolCalls: [], finishReason: "stop",
                    usage: OpenAIUsage(promptTokens: 42, completionTokens: 1, totalTokens: 43),
                    hiddenStates: payload)
            }
            return ServerCompletion(
                content: "", toolCalls: [], finishReason: "hidden_states",
                usage: OpenAIUsage(promptTokens: 42, completionTokens: 0, totalTokens: 42),
                hiddenStates: payload)
        }
    }

    private func post(_ port: Int, _ json: String) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: try #require(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(json.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, try #require(response as? HTTPURLResponse))
    }

    private func withServer<T>(
        _ backend: any ServerInferenceBackend, _ body: (Int) async throws -> T
    ) async throws -> T {
        let server = TinyTitanHTTPServer(modelID: "test-model", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let result = try await body(port)
            try await server.shutdown()
            return result
        } catch {
            try await server.shutdown()
            throw error
        }
    }

    @Test func theResponseCarriesHiddenStatesBesideChoices() async throws {
        let backend = ReadoutBackend()
        try await withServer(backend) { port in
            let (data, response) = try await post(
                port,
                #"""
                {"model":"test-model","messages":[{"role":"user","content":"hi"}],
                 "x_hidden_states":{"layers":[24,12]}}
                """#)
            #expect(response.statusCode == 200)
            let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let choices = try #require(object["choices"] as? [[String: Any]])
            #expect(choices[0]["finish_reason"] as? String == "hidden_states")
            let message = try #require(choices[0]["message"] as? [String: Any])
            #expect(message["content"] as? String == "")
            let hidden = try #require(object["hidden_states"] as? [String: Any])
            #expect(Set(hidden.keys) == ["12", "24", "position", "capture_path", "capture_ms"])
            #expect(hidden["position"] as? Int == 41)
            let layer = try #require(hidden["24"] as? [String: Any])
            #expect(layer["shape"] as? [Int] == [1, 2])
            #expect(try decodeFloats(try #require(layer["data"] as? String)) == [24, 0.25])
            let usage = try #require(object["usage"] as? [String: Any])
            #expect(usage["prompt_tokens"] as? Int == 42)
            #expect(usage["completion_tokens"] as? Int == 0)
            let seen = await backend.seen
            #expect(seen.first?.hiddenStates?.layers == [12, 24])
        }
    }

    @Test func aCaptureDuringAGenerationReturnsTheReplyAndTheRows() async throws {
        try await withServer(ReadoutBackend()) { port in
            let (data, response) = try await post(
                port,
                #"""
                {"model":"test-model","messages":[{"role":"user","content":"hi"}],
                 "x_hidden_states":{"layers":[3],"prefill_only":false}}
                """#)
            #expect(response.statusCode == 200)
            let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let choices = try #require(object["choices"] as? [[String: Any]])
            #expect(choices[0]["finish_reason"] as? String == "stop")
            let message = try #require(choices[0]["message"] as? [String: Any])
            #expect(message["content"] as? String == "ok")
            let hidden = try #require(object["hidden_states"] as? [String: Any])
            #expect(hidden["position"] as? Int == 41)
            let path = try #require(hidden["capture_path"] as? [String: Any])
            #expect(path["early_stop"] as? Bool == false)
        }
    }

    /// The data chunks of an SSE body, in order, without the [DONE] marker.
    private func dataChunks(_ body: String) throws -> [[String: Any]] {
        var chunks: [[String: Any]] = []
        for line in body.split(separator: "\n") where line.hasPrefix("data: ") {
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { continue }
            chunks.append(
                try #require(
                    JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any]))
        }
        return chunks
    }

    @Test func aStreamedCaptureCarriesHiddenStatesOnTheLastDataChunk() async throws {
        try await withServer(ReadoutBackend()) { port in
            // Without include_usage the finish chunk is the last data chunk.
            let (data, response) = try await post(
                port,
                #"""
                {"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}],
                 "x_hidden_states":{"layers":[12,3],"prefill_only":false}}
                """#)
            #expect(response.statusCode == 200)
            let body = String(decoding: data, as: UTF8.self)
            #expect(body.hasSuffix("data: [DONE]\n\n"))
            let chunks = try dataChunks(body)
            let withRows = chunks.indices.filter { chunks[$0]["hidden_states"] != nil }
            #expect(withRows == [chunks.count - 1])
            let last = chunks[chunks.count - 1]
            let choices = try #require(last["choices"] as? [[String: Any]])
            #expect(choices[0]["finish_reason"] as? String == "stop")
            let hidden = try #require(last["hidden_states"] as? [String: Any])
            #expect(Set(hidden.keys) == ["3", "12", "position", "capture_path", "capture_ms"])
            let layer = try #require(hidden["12"] as? [String: Any])
            #expect(try decodeFloats(try #require(layer["data"] as? String)) == [12, 0.25])
        }
    }

    @Test func aStreamedCaptureWithUsageCarriesHiddenStatesOnTheUsageChunk() async throws {
        try await withServer(ReadoutBackend()) { port in
            let (data, response) = try await post(
                port,
                #"""
                {"model":"test-model","stream":true,"stream_options":{"include_usage":true},
                 "messages":[{"role":"user","content":"hi"}],
                 "x_hidden_states":{"layers":[3],"prefill_only":false}}
                """#)
            #expect(response.statusCode == 200)
            let chunks = try dataChunks(String(decoding: data, as: UTF8.self))
            let last = try #require(chunks.last)
            #expect((last["choices"] as? [Any])?.isEmpty == true)
            #expect(last["usage"] != nil)
            #expect(last["hidden_states"] != nil)
            #expect(chunks.dropLast().allSatisfy { $0["hidden_states"] == nil })
        }
    }

    @Test func aStreamedProbeIsABadRequest() async throws {
        try await withServer(ReadoutBackend()) { port in
            let (_, response) = try await post(
                port,
                #"""
                {"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}],
                 "x_hidden_states":{"layers":[3]}}
                """#)
            #expect(response.statusCode == 400)
        }
    }

    @Test func anOrdinaryResponseHasNoHiddenStatesKey() async throws {
        try await withServer(ReadoutBackend()) { port in
            let (data, response) = try await post(
                port, #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#)
            #expect(response.statusCode == 200)
            let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["hidden_states"] == nil)
        }
    }

    @Test func aRefusedReadoutIsABadRequestNamingTheField() async throws {
        try await withServer(ReadoutBackend()) { port in
            let (data, response) = try await post(
                port,
                #"""
                {"model":"test-model","messages":[{"role":"user","content":"hi"}],
                 "stream":true,"x_hidden_states":{"layers":[1]}}
                """#)
            #expect(response.statusCode == 400)
            let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let error = try #require(object["error"] as? [String: Any])
            #expect(error["param"] as? String == "stream")
        }
    }

    @Test func memoryDoesNotRewriteAReadoutRequest() async throws {
        let inner = ReadoutBackend()
        var configuration = MemoryConfiguration()
        configuration.isEnabled = true
        configuration.workspace = "repo-a"
        configuration.user = "local"
        configuration.toolSurface = .full
        let service = MemoryService(configuration: configuration, durableStore: InMemoryStore())
        let backend = MemoryBackend(wrapping: inner, service: service, configuration: configuration)
        let validated = try validate(#","x_hidden_states":{"layers":[3]}"#)
        let completion = try await backend.generate(validated) { _ in }
        #expect(completion.hiddenStates != nil)
        let seen = await inner.seen
        #expect(seen.count == 1)
        // The prompt as sent: no memory instructions, no memory tools.
        #expect(seen[0].messages == validated.messages)
        #expect(seen[0].tools.isEmpty)
    }
}
