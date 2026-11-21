// swift-tools-version: 6.4
import PackageDescription

/// The language standard this package is written to.
///
/// Swift 6 language mode is set below (`swiftLanguageModes: [.v6]`); these are
/// the upcoming features that are not yet default in that mode and that the
/// tree is clean under. Enforced here rather than documented, so a target
/// added later cannot quietly opt out. The ones deliberately *not* adopted
/// (and why, with their measured diagnostic counts) are recorded in
/// `docs/swift-language-standard.md`.
///
/// `-warnings-as-errors` is part of the standard, not a preference: a warning
/// that only appears in a build log is a check nobody runs, and the release
/// script's log scan did not cover `swift test` at all. Every target carries
/// this array (24 of 24 at the time of writing), so the flag cannot be dodged
/// by a new target either. The tree builds and tests clean with it
/// (`swift build --build-tests`, `swift test --no-parallel`).
let emoneLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .unsafeFlags(["-warnings-as-errors"]),
]

let package = Package(
    name: "Emone",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Emone", targets: ["Emone"]),
        .library(name: "EmoneFormat", targets: ["EmoneFormat"]),
        .library(name: "ContinuityCore", targets: ["ContinuityCore"]),
        .executable(name: "EmoneRepack", targets: ["EmoneRepack"]),
        .executable(name: "EmoneCLI", targets: ["EmoneCLI"]),
        .executable(name: "EmoneServer", targets: ["EmoneServer"]),
        .executable(name: "EmoneBench", targets: ["EmoneBench"]),
        .executable(name: "ContinuityDemo", targets: ["ContinuityDemo"]),
        .executable(name: "emone-memory", targets: ["EmoneMemoryTool"]),
        // Emone DSH LAN Manager: the fleet manager for
        // `plugins/dsh-lan-manager`. The command is `ttlanmanager` while the
        // targets keep the long name, as `emone-memory` sits on
        // EmoneMemoryTool. Deliberately *not* in `release.sh`'s PRODUCTS: it
        // drives a group of DeepSeek Harness instances, so it is an operator tool
        // and not part of the engine users install.
        .executable(name: "ttlanmanager", targets: ["EmoneFleet"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.100.0"),
    ],
    targets: [
        .target(
            name: "EmoneFormat",
            path: "sources/EmoneFormat",
            swiftSettings: emoneLanguageStandard
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced.
        //
        // `-O2` for these kernels, which is *not* the build system's default:
        // SwiftPM's `swiftbuild` system compiles every C target at `-Os` in
        // release (the older native planner used `-O2`), and the difference on
        // the CPU int8 GEMV is 1.24x -- 2.06 against 1.66 ms per pass, min of
        // six interleaved rounds an arm, checksum identical
        // (390266.62). `.unsafeFlags` is the only way to set it, which is why
        // this package cannot be consumed as a dependency; Emone is an
        // application package and nothing depends on it. Raising Swift to
        // `-O3` was tried and rejected for the same constraint, having measured
        // the same as the release default (0.675 vs 0.680 ms).
        //
        // The language standard and the hardening warnings are enforced here,
        // not merely declared: `-std=c99` comes from `cLanguageStandard` above,
        // and these settings add the flags `-Wall -Wextra` does not imply plus
        // `-Werror`, so a new kernel cannot land with a shadowed variable, a
        // narrowing conversion, a dropped qualifier, a non-literal format or a
        // missing prototype. All three C files compile clean under the full set
        // (AUDIT/tool-coverage.md, proof L8), and `-pedantic-errors` rejects the
        // implicit declarations and GNU extensions C99 does not have.
        .target(
            name: "EmoneKernelsC",
            path: "sources/EmoneKernelsC",
            cSettings: [
                .unsafeFlags([
                    "-O2",
                    "-pedantic-errors",
                    "-Wall", "-Wextra",
                    "-Wshadow", "-Wconversion", "-Wsign-conversion", "-Wcast-qual",
                    "-Wwrite-strings", "-Wformat=2", "-Wstrict-prototypes",
                    "-Wmissing-prototypes",
                    "-Werror",
                ])
            ],
            swiftSettings: emoneLanguageStandard
        ),
        .target(
            name: "Emone",
            dependencies: [
                "EmoneFormat",
                "EmoneKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "sources/Emone",
            resources: [
                .copy("Metal")
            ],
            swiftSettings: emoneLanguageStandard
        ),
        .target(
            name: "EmoneRepackCore",
            dependencies: ["EmoneFormat"],
            path: "sources/EmoneRepack/Core",
            swiftSettings: emoneLanguageStandard
        ),
        .executableTarget(
            name: "EmoneRepack",
            dependencies: ["EmoneRepackCore"],
            path: "sources/EmoneRepack/Command",
            swiftSettings: emoneLanguageStandard
        ),
        // The live expert-grid view: the renderer, the terminal handling and the
        // trace model, shared by the CLI's `--live-trace` and the server's. A
        // library of its own so neither front end depends on the other; it
        // links the runtime only for the trace ring and the runner hook.
        .target(
            name: "EmoneLiveTrace",
            dependencies: ["Emone"],
            path: "sources/EmoneLiveTrace",
            swiftSettings: emoneLanguageStandard
        ),
        .target(
            name: "EmoneCLICore",
            dependencies: ["Emone", "EmoneLiveTrace"],
            path: "sources/EmoneCLI",
            exclude: ["Command"],
            swiftSettings: emoneLanguageStandard
        ),
        .executableTarget(
            name: "EmoneCLI",
            dependencies: ["EmoneCLICore"],
            path: "sources/EmoneCLI/Command",
            swiftSettings: emoneLanguageStandard
        ),
        // Continuity: sessions, task memory and context assembly, in this
        // process. Depends on nothing at all, not even NIO, so it cannot
        // reach the network and cannot be reached from one.
        .target(
            name: "ContinuityCore",
            path: "sources/ContinuityCore",
            // Documentation that lives next to the code it describes. SwiftPM
            // treats any undeclared file under a target path as unhandled and
            // warns on every clean plan; excluding it says so explicitly and
            // leaves the file where it is. (`sources/EmoneCLICore`'s
            // `exclude: ["Command"]` is the same mechanism.)
            exclude: ["README.md"],
            swiftSettings: emoneLanguageStandard
        ),
        // Worked examples and a scale check for ContinuityCore. Not part of
        // the server; it exists so the package's claims can be run.
        .executableTarget(
            name: "ContinuityDemo",
            dependencies: ["ContinuityCore"],
            path: "sources/ContinuityDemo",
            swiftSettings: emoneLanguageStandard
        ),
        // Agent memory: the model-facing surface (keys, tools, prompt
        // fragment, journal filter) over ContinuityCore. Depends on nothing
        // in the engine, so the serving path can use it without the memory
        // subsystem being able to reach back into inference, and on no
        // networking, so it cannot reach off the machine.
        .target(
            name: "EmoneMemory",
            dependencies: ["ContinuityCore"],
            path: "sources/EmoneMemory",
            swiftSettings: emoneLanguageStandard
        ),
        // See and correct what the server remembers: list, show, delete.
        // Reads take no lock; writes need the workspace.
        .executableTarget(
            name: "EmoneMemoryTool",
            dependencies: ["EmoneMemory", "ContinuityCore"],
            path: "sources/EmoneMemoryTool",
            swiftSettings: emoneLanguageStandard
        ),
        .target(
            name: "EmoneServerCore",
            dependencies: [
                "Emone",
                "EmoneLiveTrace",
                "EmoneMemory",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/EmoneServer/Core",
            swiftSettings: emoneLanguageStandard
        ),
        .executableTarget(
            name: "EmoneServer",
            dependencies: ["EmoneServerCore"],
            path: "sources/EmoneServer/Command",
            swiftSettings: emoneLanguageStandard
        ),
        .executableTarget(
            name: "EmoneBench",
            dependencies: ["Emone"],
            path: "sources/EmoneBench",
            swiftSettings: emoneLanguageStandard
        ),
        .target(
            name: "EmoneValidationSupport",
            dependencies: ["Emone"],
            path: "sources/EmoneValidation/Support",
            swiftSettings: emoneLanguageStandard
        ),
        // The fleet manager: Core is a library so its selection, aggregation and
        // request-building logic is testable without a network or a live fleet;
        // Command is the thin CLI over it.
        .target(
            name: "EmoneFleetCore",
            path: "sources/EmoneFleet/Core",
            swiftSettings: emoneLanguageStandard
        ),
        .executableTarget(
            name: "EmoneFleet",
            dependencies: ["EmoneFleetCore"],
            path: "sources/EmoneFleet/Command",
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "EmoneFleetTests",
            dependencies: ["EmoneFleetCore"],
            path: "tests/EmoneFleet",
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "EmoneTests",
            dependencies: [
                "Emone", "EmoneValidationSupport", "EmoneRepackCore",
                "EmoneCLICore", "EmoneLiveTrace",
            ],
            path: "tests/Emone",
            resources: [
                .copy("Tokenization/Fixtures"),
                .copy("Runtime/qwen38_tensor_names.txt"),
                .copy("Runtime/ple_golden.json"),
            ],
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "EmoneRepackTests",
            // `EmoneFormat` directly: the manifest and resident-index validation
            // tests assert on those types rather than on JSON dictionaries.
            dependencies: ["EmoneRepackCore", "EmoneFormat"],
            path: "tests/EmoneRepack/Core",
            resources: [.copy("Support/qwen38_tensor_names.txt")],
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "ContinuityCoreTests",
            dependencies: ["ContinuityCore"],
            path: "tests/ContinuityCore",
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "EmoneMemoryTests",
            dependencies: ["EmoneMemory", "ContinuityCore"],
            path: "tests/EmoneMemory",
            swiftSettings: emoneLanguageStandard
        ),
        .testTarget(
            name: "EmoneServerTests",
            dependencies: [
                "EmoneServerCore",
                "EmoneMemory",
                // The live-trace view's types, for the server's `--live-trace`
                // tests that read what it drew.
                "EmoneLiveTrace",
                // `GenerationDefaults.Sampling`, so the mapper tests can pin
                // that an omitted field follows the served model's profile
                // rather than a hardcoded house default.
                "Emone",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/EmoneServer",
            resources: [.copy("Fixtures")],
            swiftSettings: emoneLanguageStandard
        ),
    ],
    swiftLanguageModes: [.v6],
    // The C in this package is written to strict C99; declaring it here makes
    // the compiler enforce it instead of documenting an intention. Together
    // with the EmoneKernelsC cSettings below this is the C language
    // standard in force, and AUDIT/tool-coverage.md proves a violation fails.
    cLanguageStandard: .c99
)
