import Foundation
import NIOCore
import NIOHTTP1

extension ServerHTTPHandler {
    /// `POST /v1/systemone` (TinyTitan-lmw.5): one System One request in, its
    /// answers out (`SystemOneService`). Malformed or oversized requests are
    /// 422, as Kev's server answers them; no decision adapter installed is 404.
    func handleSystemOne(body: ByteBuffer, context: ChannelHandlerContext) {
        guard let systemOne else {
            writeRequestError(
                context,
                .notFound(
                    message: "System One is not available: no decision adapter "
                        + "(decision_adapter.json) over an installed model in the catalog",
                    param: nil),
                status: .notFound, surface: .chat)
            return
        }
        let data = Data(body.readableBytesView)
        let contextBox = SendableContext(context)
        activeTask = childChannels.startTask {
            do {
                let (answer, requestID) = try await systemOne.answer(data)
                self.writeData(
                    contextBox.value, status: .ok, data: answer,
                    extraHeaders: [("x-typesafe-request-id", requestID)])
            } catch let failure as SystemOneService.Failure {
                self.writeRequestError(
                    contextBox.value,
                    .invalid(message: failure.description, param: nil, code: "invalid_request"),
                    status: .unprocessableEntity, surface: .chat)
            } catch {
                self.writeRequestError(
                    contextBox.value,
                    .invalid(
                        message: "System One failed: \(error)", param: nil, code: "server_error"),
                    status: .internalServerError, surface: .chat)
            }
        }
    }
}
