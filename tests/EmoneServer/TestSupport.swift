import Foundation

/// Failures the server test fixtures raise instead of trapping.
///
/// These fixtures used to force unwrap `URL(string:)`, `HTTPURLResponse` and
/// tokenizer output. A typo in a test URL crashed the whole test process, which
/// hides the other tests' results; a thrown error fails only the test that
/// needed the value, and names it.
enum TestFixtureError: Error, CustomStringConvertible {
    case missing(String)

    var description: String {
        switch self {
        case .missing(let what):
            return "the test fixture could not build \(what)"
        }
    }
}

/// The value, or a thrown error naming what was missing.
func requireFixture<T>(_ value: T?, _ what: String) throws -> T {
    guard let value else { throw TestFixtureError.missing(what) }
    return value
}

/// A URL to the local test server, or a thrown error naming the path.
func localURL(port: Int, _ path: String) throws -> URL {
    try requireFixture(
        URL(string: "http://127.0.0.1:\(port)\(path)"),
        "a URL for \(path) on port \(port)")
}

/// A request to the local test server.
func localRequest(port: Int, _ path: String) throws -> URLRequest {
    URLRequest(url: try localURL(port: port, path))
}
