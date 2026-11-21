import Foundation

/// What one member's management API answers with.
///
/// These mirror `plugins/dsh-lan-manager`'s JSON exactly; the plugin is the
/// contract and this file is the reader. Fields the plugin may omit are
/// optional here, so a member running an older plugin degrades to "unknown"
/// rather than failing the whole read.

/// The answering instance's own identity.
public struct FleetSelf: Codable, Sendable, Equatable {
    public let id: String?
    public let name: String?
    public let port: Int?
    public let addresses: [String]?
    /// The plugin's own version.
    public let version: String?
    /// The harness the plugin is running inside — what a fleet view wants to show.
    public let dshVersion: String?

    public init(
        id: String? = nil, name: String? = nil, port: Int? = nil,
        addresses: [String]? = nil, version: String? = nil, dshVersion: String? = nil
    ) {
        self.id = id
        self.name = name
        self.port = port
        self.addresses = addresses
        self.version = version
        self.dshVersion = dshVersion
    }
}

/// A workspace as the plugin projects it: the page-visible sessions only.
public struct FleetWorkspace: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let path: String
    public let title: String?
    public let sessionCount: Int?
    public let hiddenSessionCount: Int?
    public let sessionIds: [String]?

    public init(
        id: String, path: String, title: String? = nil,
        sessionCount: Int? = nil, hiddenSessionCount: Int? = nil,
        sessionIds: [String]? = nil
    ) {
        self.id = id
        self.path = path
        self.title = title
        self.sessionCount = sessionCount
        self.hiddenSessionCount = hiddenSessionCount
        self.sessionIds = sessionIds
    }
}

/// One visible session, tagged with the workspace it belongs to.
public struct FleetSession: Codable, Sendable, Equatable, Identifiable {
    public let sessionId: String
    public let workspaceId: String?
    public let workspacePath: String?
    public let workspaceTitle: String?
    public let title: String?
    public let turns: Int?

    public var id: String { sessionId }

    public init(
        sessionId: String, workspaceId: String? = nil, workspacePath: String? = nil,
        workspaceTitle: String? = nil, title: String? = nil, turns: Int? = nil
    ) {
        self.sessionId = sessionId
        self.workspaceId = workspaceId
        self.workspacePath = workspacePath
        self.workspaceTitle = workspaceTitle
        self.title = title
        self.turns = turns
    }
}

/// Another member of the group, with the inventory this instance holds for it.
public struct FleetPeer: Codable, Sendable, Equatable {
    public let id: String
    public let address: String
    public let port: Int
    public let name: String?
    public let version: String?
    /// How this member was found: `tailscale`, `bonjour`, `seed`, `subnet`, or
    /// `gossip:<peer>` when another member told us about it.
    public let source: String?
    /// The addresses the member reports for itself, so the view can show v4 and v6.
    public let addresses: [String]?
    public let dshVersion: String?
    public let lastSeen: Double?
    public let workspaceCount: Int?
    public let sessionCount: Int?
    public let workspaces: [FleetWorkspace]?
    public let sessions: [FleetSession]?

    public init(
        id: String, address: String, port: Int, name: String? = nil,
        version: String? = nil, source: String? = nil, addresses: [String]? = nil,
        dshVersion: String? = nil, lastSeen: Double? = nil,
        workspaceCount: Int? = nil, sessionCount: Int? = nil,
        workspaces: [FleetWorkspace]? = nil, sessions: [FleetSession]? = nil
    ) {
        self.id = id
        self.address = address
        self.port = port
        self.name = name
        self.version = version
        self.source = source
        self.addresses = addresses
        self.dshVersion = dshVersion
        self.lastSeen = lastSeen
        self.workspaceCount = workspaceCount
        self.sessionCount = sessionCount
        self.workspaces = workspaces
        self.sessions = sessions
    }
}

/// The aggregate a manager reads: this instance, and every member it knows.
public struct FleetInventory: Codable, Sendable, Equatable {
    public let ok: Bool?
    public let group: String?
    public let node: FleetSelf?
    public let lastDiscovery: Double?
    public let workspaces: [FleetWorkspace]?
    public let sessions: [FleetSession]?
    public let peers: [FleetPeer]?

    /// `self` is a Swift keyword, so the JSON key is mapped rather than renamed
    /// in the wire format the plugin owns.
    private enum CodingKeys: String, CodingKey {
        case ok, group, lastDiscovery, workspaces, sessions, peers
        case node = "self"
    }

    public init(
        ok: Bool? = nil, group: String? = nil, node: FleetSelf? = nil,
        lastDiscovery: Double? = nil, workspaces: [FleetWorkspace]? = nil,
        sessions: [FleetSession]? = nil, peers: [FleetPeer]? = nil
    ) {
        self.ok = ok
        self.group = group
        self.node = node
        self.lastDiscovery = lastDiscovery
        self.workspaces = workspaces
        self.sessions = sessions
        self.peers = peers
    }
}

/// One machine in the group, reduced to what the manager has to act on: where it
/// answers, and what it holds.
public struct FleetNode: Sendable, Equatable, Identifiable {
    public let id: String
    public let name: String
    public let host: String
    public let port: Int
    public let isSelf: Bool
    public let source: String
    public let addresses: [String]
    public let dshVersion: String?
    public let lastSeen: Double?
    public let workspaces: [FleetWorkspace]
    public let sessions: [FleetSession]

    public init(
        id: String, name: String, host: String, port: Int, isSelf: Bool,
        source: String = "local", addresses: [String] = [], dshVersion: String? = nil,
        lastSeen: Double? = nil, workspaces: [FleetWorkspace] = [],
        sessions: [FleetSession] = []
    ) {
        self.id = id
        self.name = name
        self.host = host
        self.port = port
        self.isSelf = isSelf
        self.source = source
        self.addresses = addresses
        self.dshVersion = dshVersion
        self.lastSeen = lastSeen
        self.workspaces = workspaces
        self.sessions = sessions
    }

    /// How the member was found, in the words a person reads.
    public var sourceLabel: String {
        if isSelf { return "this Mac" }
        if source.hasPrefix("gossip:") { return "mesh" }
        switch source {
        case "tailscale": return "Tailscale"
        case "bonjour": return "Bonjour"
        case "seed": return "seed"
        case "subnet": return "LAN scan"
        default: return source.isEmpty ? "unknown" : source
        }
    }

    /// The IPv6 address this member reports, when it has one.
    public var ipv6: String? {
        addresses.first { $0.contains(":") }
    }

    /// The IPv4 address this member reports, when it has one.
    public var ipv4: String? {
        if !host.contains(":") { return host }
        return addresses.first { !$0.contains(":") }
    }

    /// Where to reach this node's management API.
    public var target: FleetTarget { FleetTarget(host: host, port: port) }
}

/// The whole group, assembled from one instance's answer.
public struct FleetGroup: Sendable, Equatable {
    public let group: String?
    public let nodes: [FleetNode]

    public init(group: String? = nil, nodes: [FleetNode] = []) {
        self.group = group
        self.nodes = nodes
    }

    public var workspaces: Int { nodes.reduce(0) { $0 + $1.workspaces.count } }
    public var sessions: Int { nodes.reduce(0) { $0 + $1.sessions.count } }

    /// The node that owns a session, or `nil` when nobody claims it.
    public func owner(ofSession sessionId: String) -> FleetNode? {
        nodes.first { node in node.sessions.contains { $0.sessionId == sessionId } }
    }

    /// The node that owns a workspace.
    public func owner(ofWorkspace workspaceId: String) -> FleetNode? {
        nodes.first { node in node.workspaces.contains { $0.id == workspaceId } }
    }

    /// Every session in the group, paired with the node that owns it.
    public var sessionOwners: [(node: FleetNode, session: FleetSession)] {
        nodes.flatMap { node in node.sessions.map { (node, $0) } }
    }
}

/// Where a member answers.
public struct FleetTarget: Sendable, Hashable, CustomStringConvertible {
    public let host: String
    public let port: Int

    public init(host: String, port: Int) {
        self.host = host
        self.port = port
    }

    /// Parse `host`, `host:port`, or `[v6]:port`.
    /// - Returns: `nil` when the text names no usable host.
    public init?(text: String, defaultPort: Int) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("[") {
            guard let close = trimmed.firstIndex(of: "]") else { return nil }
            let host = String(trimmed[trimmed.index(after: trimmed.startIndex)..<close])
            let rest = trimmed[trimmed.index(after: close)...]
            let port = rest.hasPrefix(":") ? Int(rest.dropFirst()) : defaultPort
            guard let port, !host.isEmpty, (1...65535).contains(port) else { return nil }
            self.init(host: host, port: port)
            return
        }
        let parts = trimmed.split(separator: ":", omittingEmptySubsequences: false)
        if parts.count == 2, let port = Int(parts[1]) {
            guard !parts[0].isEmpty, (1...65535).contains(port) else { return nil }
            self.init(host: String(parts[0]), port: port)
            return
        }
        guard parts.count == 1 else { return nil }
        self.init(host: trimmed, port: defaultPort)
    }

    public var description: String { "\(host):\(port)" }
}
