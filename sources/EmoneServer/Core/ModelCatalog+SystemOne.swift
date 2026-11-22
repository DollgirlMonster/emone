import Foundation
import Emone

extension ModelCatalog {
    /// A System One adapter package (Emone-lmw.11) and the install it is
    /// laid over.
    public struct SystemOnePackage: Sendable, Equatable {
        public let adapter: URL
        public let modelID: String
        public let base: Entry
    }

    /// The decision adapter `/v1/systemone` serves (Emone-lmw.5): a
    /// package in the models folder (`decision_adapter.json`) whose base
    /// install is in this catalog. `EMONE_SYSTEM_ONE` names one by its
    /// folder or model id, or turns System One `off`; unset, the first such
    /// package by folder name is served.
    public func systemOnePackage(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> SystemOnePackage? {
        let wanted = environment["EMONE_SYSTEM_ONE"]?.trimmingCharacters(in: .whitespaces)
        if wanted?.lowercased() == "off" { return nil }
        let children =
            (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let folder = child.resolvingSymlinksInPath()
            let manifestURL = folder.appendingPathComponent(LoRAAdapter.manifestFile)
            guard let data = try? Data(contentsOf: manifestURL),
                let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let modelID = manifest["model_id"] as? String,
                let base = (manifest["base"] as? [String: Any])?["model_id"] as? String
            else { continue }
            if let wanted, !wanted.isEmpty, wanted != child.lastPathComponent, wanted != modelID {
                continue
            }
            let install = entries.first { entry in
                entry.backend == .gpu
                    && (entry.id == base
                        || entry.id.split(separator: "_").first.map(String.init) == base)
            }
            guard let install else { continue }
            return SystemOnePackage(adapter: folder, modelID: modelID, base: install)
        }
        return nil
    }
}
