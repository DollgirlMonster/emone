import Foundation

/// A selectable row in the dashboard: a member, one of its workspaces, or one of
/// its sessions. Flattened from the group so a single selection index drives
/// navigation, and so the rows are testable without a terminal.
public enum FleetRow: Sendable, Equatable {
    case node(index: Int)
    case workspace(node: Int, index: Int)
    case session(node: Int, index: Int)

    /// Stable identity, so a refresh can keep the cursor on the same thing.
    public var id: String {
        switch self {
        case .node(let index): return "node:\(index)"
        case .workspace(let node, let index): return "ws:\(node):\(index)"
        case .session(let node, let index): return "s:\(node):\(index)"
        }
    }

    public var depth: Int {
        switch self {
        case .node: return 0
        case .workspace: return 1
        case .session: return 2
        }
    }
}

/// One line of the table, before it is cut to a width.
public struct FleetLine: Sendable, Equatable {
    public let row: FleetRow?
    public let name: String
    public let kind: String
    public let address: String
    public let version: String
    public let workspaces: String
    public let sessions: String
    public let lastSeen: String
    public let ipv6: String

    public init(
        row: FleetRow?, name: String, kind: String = "", address: String = "",
        version: String = "", workspaces: String = "", sessions: String = "",
        lastSeen: String = "", ipv6: String = ""
    ) {
        self.row = row
        self.name = name
        self.kind = kind
        self.address = address
        self.version = version
        self.workspaces = workspaces
        self.sessions = sessions
        self.lastSeen = lastSeen
        self.ipv6 = ipv6
    }
}

/// What the dashboard is currently doing.
public enum FleetMode: Sendable, Equatable {
    case browse
    case help
    case input(field: FleetField)
    case confirm(question: String, action: FleetAction)
}

/// A field the dashboard can ask for.
public enum FleetField: Sendable, Equatable {
    case promptText(target: FleetPromptTarget)
    case workspacePath(nodeID: String)
    case workspaceTitle(nodeID: String, path: String)
}

/// What a prompt is addressed to.
public enum FleetPromptTarget: Sendable, Equatable {
    case session(nodeID: String, sessionID: String)
    case workspace(nodeID: String, workspaceID: String)
    case node(nodeID: String)
    case group

    public var label: String {
        switch self {
        case .session(_, let sessionID): return "session \(sessionID)"
        case .workspace(_, let workspaceID): return "workspace \(workspaceID)"
        case .node(let nodeID): return "every session on \(nodeID)"
        case .group: return "every session in the group"
        }
    }
}

/// Something the user asked for that needs the network. The dashboard produces
/// these; `ttlanmanager top` performs them, so the model stays pure.
public enum FleetAction: Sendable, Equatable {
    case prompt(target: FleetPromptTarget, text: String)
    case archiveSession(nodeID: String, sessionID: String)
    case deleteWorkspace(nodeID: String, workspaceID: String)
    case createWorkspace(nodeID: String, path: String, title: String?)
    case refresh
    case quit
}

/// The dashboard's whole state. Value type, no I/O — every key press and every
/// refresh is a pure transition, which is what makes the layout and the key map
/// testable without a terminal.
public struct FleetDashboard: Sendable {
    public private(set) var group: FleetGroup
    public let seed: FleetTarget
    public private(set) var expanded: Set<String>
    public private(set) var selection: Int
    public private(set) var mode: FleetMode
    public private(set) var input: String
    public private(set) var status: String
    public private(set) var refreshedAt: Date?
    public private(set) var failure: String?

    public init(
        group: FleetGroup = FleetGroup(),
        seed: FleetTarget = FleetTarget(host: "127.0.0.1", port: 3080)
    ) {
        self.group = group
        self.seed = seed
        self.expanded = Set(group.nodes.map(\.id))
        self.selection = 0
        self.mode = .browse
        self.input = ""
        self.status = "reading the group…"
        self.refreshedAt = nil
        self.failure = nil
    }

    // MARK: - rows

    /// The flattened, visible rows: members, and — when expanded — their
    /// workspaces and the sessions inside each.
    public var rows: [FleetRow] {
        var rows: [FleetRow] = []
        for (nodeIndex, node) in group.nodes.enumerated() {
            rows.append(.node(index: nodeIndex))
            guard expanded.contains(node.id) else { continue }
            for (workspaceIndex, workspace) in node.workspaces.enumerated() {
                rows.append(.workspace(node: nodeIndex, index: workspaceIndex))
                let ids = Set(workspace.sessionIds ?? [])
                for (sessionIndex, session) in node.sessions.enumerated()
                where ids.isEmpty || ids.contains(session.sessionId) {
                    rows.append(.session(node: nodeIndex, index: sessionIndex))
                }
            }
        }
        return rows
    }

    public var selectedRow: FleetRow? {
        rows.indices.contains(selection) ? rows[selection] : rows.last
    }

    public var selectedNode: FleetNode? {
        guard let row = selectedRow else { return group.nodes.first }
        switch row {
        case .node(let index): return group.nodes.indices.contains(index) ? group.nodes[index] : nil
        case .workspace(let node, _), .session(let node, _):
            return group.nodes.indices.contains(node) ? group.nodes[node] : nil
        }
    }

    // MARK: - transitions

    /// A refresh landed.
    public mutating func apply(group: FleetGroup, at date: Date = Date()) {
        let keep = selectedRow?.id
        // A member seen for the first time opens; one the user collapsed stays
        // collapsed across refreshes.
        let known = Set(self.group.nodes.map(\.id))
        for node in group.nodes where !known.contains(node.id) {
            expanded.insert(node.id)
        }
        self.group = group
        self.refreshedAt = date
        self.failure = nil
        self.status =
            "\(group.nodes.count) member(s), \(group.workspaces) workspace(s), \(group.sessions) session(s)"
        // Keep the cursor on the same row where it still exists.
        if let keep, let index = rows.firstIndex(where: { $0.id == keep }) {
            selection = index
        } else {
            selection = min(selection, max(0, rows.count - 1))
        }
    }

    /// A refresh failed. The previous group stays on screen: a fleet view that
    /// blanks when one member is unreachable is worse than one that says so.
    public mutating func apply(failure message: String) {
        self.failure = message
        self.status = message
    }

    /// Replace the status line — used after an action, so the message survives
    /// the refresh that follows it.
    public mutating func setStatus(_ message: String) {
        status = message
        failure = nil
    }

    public mutating func move(by delta: Int) {
        let count = rows.count
        guard count > 0 else { return }
        selection = max(0, min(count - 1, selection + delta))
    }

    public mutating func toggleExpanded() {
        guard let node = selectedNode else { return }
        if expanded.contains(node.id) {
            expanded.remove(node.id)
        } else {
            expanded.insert(node.id)
        }
        if let index = rows.firstIndex(where: {
            $0.id == "node:\(group.nodes.firstIndex { $0.id == node.id } ?? 0)"
        }) {
            selection = index
        }
    }

    public mutating func beginPrompt(_ target: FleetPromptTarget) {
        input = ""
        mode = .input(field: .promptText(target: target))
    }

    public mutating func beginCreateWorkspace(nodeID: String) {
        input = ""
        mode = .input(field: .workspacePath(nodeID: nodeID))
    }

    public mutating func append(_ text: String) { input.append(text) }

    public mutating func backspace() {
        if !input.isEmpty { input.removeLast() }
    }

    public mutating func cancel() {
        mode = .browse
        input = ""
    }

    /// The action a row's primary key does.
    public func promptTarget(for row: FleetRow?) -> FleetPromptTarget? {
        switch row {
        case .session(let nodeIndex, let sessionIndex):
            guard group.nodes.indices.contains(nodeIndex),
                group.nodes[nodeIndex].sessions.indices.contains(sessionIndex)
            else { return nil }
            let node = group.nodes[nodeIndex]
            return .session(nodeID: node.id, sessionID: node.sessions[sessionIndex].sessionId)
        case .workspace(let nodeIndex, let workspaceIndex):
            guard group.nodes.indices.contains(nodeIndex),
                group.nodes[nodeIndex].workspaces.indices.contains(workspaceIndex)
            else { return nil }
            let node = group.nodes[nodeIndex]
            return .workspace(nodeID: node.id, workspaceID: node.workspaces[workspaceIndex].id)
        case .node(let index):
            guard group.nodes.indices.contains(index) else { return nil }
            return .node(nodeID: group.nodes[index].id)
        case nil:
            return group.nodes.isEmpty ? nil : .group
        }
    }

    /// A key press in browse mode, as an action or a mode change. Pure, so the
    /// whole key map is testable.
    public mutating func press(_ key: FleetKey) {
        switch mode {
        case .help:
            mode = .browse
            return
        case .input(let field):
            pressInInput(key, field: field)
            return
        case .confirm(_, let action):
            if key == .character("y") || key == .character("Y") || key == .enter {
                mode = .browse
                pendingAction = action
            } else {
                cancel()
                status = "cancelled"
            }
            return
        case .browse:
            break
        }
        pressInBrowse(key)
    }

    /// An action confirmed in the UI and not yet performed.
    public private(set) var pendingAction: FleetAction?

    /// Take the action the user confirmed.
    public mutating func takePendingAction() -> FleetAction? {
        defer { pendingAction = nil }
        return pendingAction
    }

    private mutating func pressInInput(_ key: FleetKey, field: FleetField) {
        switch key {
        case .enter:
            finishInput(field: field)
        case .backspace:
            backspace()
        case .escape:
            cancel()
            status = "cancelled"
        case .character(let text):
            append(text)
        default:
            break
        }
    }

    private mutating func finishInput(field: FleetField) {
        let value = input
        cancel()
        switch field {
        case .promptText(let target):
            guard !value.isEmpty else {
                status = "empty prompt — nothing sent"
                return
            }
            pendingAction = .prompt(target: target, text: value)
        case .workspacePath(let nodeID):
            guard !value.isEmpty else {
                status = "empty path — nothing registered"
                return
            }
            mode = .input(field: .workspaceTitle(nodeID: nodeID, path: value))
        case .workspaceTitle(let nodeID, let path):
            pendingAction = .createWorkspace(
                nodeID: nodeID, path: path, title: value.isEmpty ? nil : value)
        }
    }

    private mutating func pressInBrowse(_ key: FleetKey) {
        switch key {
        case .up:
            move(by: -1)
        case .down:
            move(by: 1)
        case .left:
            if let node = selectedNode, expanded.contains(node.id) { toggleExpanded() }
        case .right, .enter:
            if let node = selectedNode, !expanded.contains(node.id) { toggleExpanded() }
        case .character("j"):
            move(by: 1)
        case .character("k"):
            move(by: -1)
        case .character("p"):
            if let target = promptTarget(for: selectedRow) { beginPrompt(target) }
        case .character("P"):
            if let node = selectedNode { beginPrompt(.node(nodeID: node.id)) }
        case .character("A"):
            beginPrompt(.group)
        case .character("c"):
            if let node = selectedNode { beginCreateWorkspace(nodeID: node.id) }
        case .character("a"):
            if case .session(let nodeIndex, let sessionIndex) = selectedRow,
                group.nodes.indices.contains(nodeIndex),
                group.nodes[nodeIndex].sessions.indices.contains(sessionIndex)
            {
                let node = group.nodes[nodeIndex]
                let session = node.sessions[sessionIndex]
                mode = .confirm(
                    question: "Archive session \(session.sessionId)?",
                    action: .archiveSession(nodeID: node.id, sessionID: session.sessionId))
            } else {
                status = "select a session to archive"
            }
        case .character("d"):
            if case .workspace(let nodeIndex, let workspaceIndex) = selectedRow,
                group.nodes.indices.contains(nodeIndex),
                group.nodes[nodeIndex].workspaces.indices.contains(workspaceIndex)
            {
                let node = group.nodes[nodeIndex]
                let workspace = node.workspaces[workspaceIndex]
                mode = .confirm(
                    question: "Delete workspace \(workspace.id)? Its sessions are archived first.",
                    action: .deleteWorkspace(nodeID: node.id, workspaceID: workspace.id))
            } else {
                status = "select a workspace to delete"
            }
        case .character("r"):
            pendingAction = .refresh
        case .character("?"):
            mode = .help
        case .character("q"):
            pendingAction = .quit
        case .escape:
            pendingAction = .quit
        default:
            break
        }
    }
}

/// A key the terminal reported, reduced to what the dashboard cares about.
public enum FleetKey: Sendable, Equatable {
    case up, down, left, right, enter, escape, backspace
    case character(String)
}
