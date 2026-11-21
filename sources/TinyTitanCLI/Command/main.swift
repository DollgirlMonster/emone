import Darwin
import Foundation
import TinyTitanCLICore

let rawArgv = Array(CommandLine.arguments.dropFirst())
// `decide` is its own mode: a System One decision model scores, it does not
// generate, so none of the generation flags apply to it.
if rawArgv.first == "decide" {
    let argv = Array(rawArgv.dropFirst())
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    // Detached: top-level code is main-actor isolated, and the main thread is
    // about to block on the semaphore.
    Task.detached {
        box.code = await DecideCommand.run(argv)
        sem.signal()
    }
    sem.wait()
    exit(box.code)
}
// `features` extracts frozen-backbone decision features (TinyTitan-lmw.15).
if rawArgv.first == "features" {
    let argv = Array(rawArgv.dropFirst())
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        box.code = await FeaturesCommand.run(argv)
        sem.signal()
    }
    sem.wait()
    exit(box.code)
}
// `fastpath` is the TinyTitan-lmw.13 timing spike: System One on the MoE vs Rev.
if rawArgv.first == "fastpath" {
    let argv = Array(rawArgv.dropFirst())
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    Task.detached {
        box.code = await FastPathCommand.run(argv)
        sem.signal()
    }
    sem.wait()
    exit(box.code)
}
let parsedArgs: Args
do {
    parsedArgs = try Args.parse(rawArgv)
} catch ArgsError.helpRequested {
    print(Args.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n".utf8))
    FileHandle.standardError.write(Data(Args.usage.utf8))
    FileHandle.standardError.write(Data("\n".utf8))
    exit(2)
}

// Keep the cancellable task off the top-level executor so the blocking signal
// bridge cannot prevent it from starting.
/// unchecked-invariant: a hand-off box between the top-level task and the
/// blocking signal wait, published exactly once before the semaphore is
/// signalled and read only after -- the semaphore is the ordering.
final class RunBox: @unchecked Sendable {
    var code: Int32 = 0
    var task: Task<Void, Never>?
}

func drive(_ args: Args) -> Int32 {
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    box.task = Task {
        let result = await run(args: args)
        box.code = result.exitCode
        sem.signal()
    }

    signal(SIGINT, SIG_IGN)
    let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    sigintSource.setEventHandler { box.task?.cancel() }
    sigintSource.resume()

    sem.wait()
    return box.code
}

exit(drive(parsedArgs))
