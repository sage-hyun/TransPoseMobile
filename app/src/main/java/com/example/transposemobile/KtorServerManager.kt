package com.example.transposemobile

import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.server.application.install
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.Frame
import kotlinx.coroutines.delay

class KtorServerManager(private val imuDataProducer: ImuDataProducer,
                        private val onnxManager: OnnxManager) {

    private var server: ApplicationEngine? = null

    fun startServer() {
        server = embeddedServer(CIO, port = 5559) {
            install(WebSockets) // WebSocket 플러그인 설치
            routing {
                staticResources("/unityWebGL", "static", index = "index.html") {
                    modify { resource, call ->
                        if (resource.path.endsWith(".gz")) {
                            call.response.headers.append(HttpHeaders.ContentEncoding, "gzip")
                        }
                    }
                    contentType { resource ->
                        if (resource.path.contains("wasm.gz")) {
                            ContentType.Application.Wasm
                        } else null
                    }
                }

                // WebSocket connection to handle socket communication
                webSocket("/ws") {
                    onnxManager.resetState()
                    while (true) {
                        val msg = onnxManager.getInferenceResult()
                        if(msg != null) {
                            send(Frame.Text(msg))
                        } else if (imuDataProducer.isRunning) {
                            continue
                        } else break

                        delay(10) // 메시지 전송 간격 (밀리초)
                    }
                }
            }
        }.start(wait = false)
    }

    fun stopServer() {
        server?.stop(1000, 2000) // 1초 대기 후 중지
        server = null
    }
}