import Testing

@testable import TinyTitanLiveTrace

@Suite struct LiveTraceTextTests {
    @Test func wrapsAtWordsAndKeepsEveryLineWithinWidth() {
        let lines = LiveTraceText.wrap("hello world foo bar baz", width: 10)
        #expect(lines == ["hello", "world foo", "bar baz"])
        #expect(lines.allSatisfy { LiveTraceText.width(of: $0) <= 10 })
    }

    @Test func breaksAtNewlinesAndKeepsBlankLines() {
        #expect(LiveTraceText.wrap("a\n\nb", width: 10) == ["a", "", "b"])
        #expect(LiveTraceText.wrap("a\r\nb", width: 10) == ["a", "b"])
        #expect(LiveTraceText.wrap("", width: 10) == [""])
    }

    @Test func cutsAWordLongerThanALine() {
        #expect(LiveTraceText.wrap("abcdefghijkl", width: 5) == ["abcde", "fghij", "kl"])
    }

    @Test func wideCharactersCountTwoCells() {
        #expect(LiveTraceText.width(of: "日本") == 4)
        let lines = LiveTraceText.wrap("日本語日本語", width: 6)
        #expect(lines == ["日本語", "日本語"])
    }

    @Test func untrustedLogTextCannotInjectEscapes() {
        #expect(LiveTraceText.sanitize("a\u{1B}[2Jb\u{1B}[31mc\u{7}d\te") == "abcd    e")
        #expect(LiveTraceText.sanitize("\u{1B}]0;title\u{7}x") == "x")
        #expect(LiveTraceText.sanitize("\u{1B}]8;;http://a\u{1B}\\link") == "link")
    }

    @Test func truncateAndHardWrap() {
        #expect(LiveTraceText.truncate("abcdef", width: 4) == "abcd")
        #expect(LiveTraceText.hardWrap("abcdef", width: 4) == ["abcd", "ef"])
        #expect(LiveTraceText.hardWrap("", width: 4) == [""])
    }
}
