import Foundation
import Testing

@testable import TinyTitan

/// The tool-result continuation bridge against every bundled chat template.
///
/// The bridge is what gets prefilled after a cached tool-calling turn, so it
/// must equal the tail a cold render of the same conversation would put there;
/// anything else resumes the model into a prompt it would never have seen.
@Suite("Tool-result continuation")
struct ToolResultContinuationTests {
    private typealias Message = GFTokenizer.Message

    static let fixtures = [
        "ChatMLTokenizer", "AgentWorldChatMLTokenizer", "OrnithChatMLTokenizer",
        "Qwen38ChatMLTokenizer", "Qwen35ChatMLTokenizer",
    ]

    private static func folder(_ fixture: String) throws -> URL {
        try #require(
            Bundle.module.url(
                forResource: fixture, withExtension: nil, subdirectory: "Fixtures"))
    }

    private static let assistant = Message(
        role: .assistant,
        content: "<think>\nI should write the file\n</think>\n\nWriting it.",
        toolCalls: [
            .init(
                id: "call_2", name: "create_file",
                arguments: .object(["path": .string("recipe.py")]))
        ])

    /// A session some turns in: a system prompt, an earlier tool round, then
    /// the cached turn. The template's `<think>` stripping keys on the last
    /// user query, which only a history like this one exercises.
    private static let history: [Message] = [
        Message(role: .system, content: "You are a coding agent."),
        Message(role: .user, content: "Add a recipe toolbelt."),
        Message(
            role: .assistant, content: "Looking first.",
            toolCalls: [.init(id: "call_1", name: "ls", arguments: .object([:]))]),
        Message(role: .tool, content: "agent.py\nconfig.py", toolCallID: "call_1", name: "ls"),
    ]

    private static let result = Message(
        role: .tool, content: "Error: File already exists",
        toolCallID: "call_2", name: "create_file")

    private static let continuations: [String: [Message]] = [
        "results only": [result],
        "user interjects": [result, Message(role: .user, content: "how's it coming?")],
        "two appended turns": [
            result,
            Message(role: .user, content: "how's it coming?"),
            Message(role: .user, content: "also add tests"),
        ],
    ]

    @Test(
        "The bridge is the tail of a cold render",
        arguments: fixtures, [ModelThinkingMode.off, .on])
    func bridgeMatchesColdRender(_ fixture: String, _ thinking: ModelThinkingMode) async throws {
        let tok = try await GFTokenizer.load(
            from: Self.folder(fixture), thinkingMode: thinking)
        for (name, continuation) in Self.continuations {
            let incoming = Self.history + [Self.assistant] + continuation
            let cold = try tok.encodeToolChat(messages: incoming, tools: [])
            let bridge = try tok.encodeToolResultContinuation(
                cachedMessages: Self.history,
                assistant: Self.assistant,
                incomingMessages: incoming,
                tools: [])
            #expect(bridge.first == tok.endOfTurnID, "\(name)")
            #expect(
                cold.count > bridge.count && cold.suffix(bridge.count).elementsEqual(bridge),
                "\(name): bridge \(tok.decode(bridge, skipSpecialTokens: false))")
            let text = tok.decode(bridge, skipSpecialTokens: false)
            #expect(text.hasPrefix("<|im_end|>\n<|im_start|>user\n<tool_response>"), "\(name)")
            for message in continuation where message.role == .user {
                #expect(text.contains(message.content ?? ""), "\(name)")
            }
        }
    }
}
