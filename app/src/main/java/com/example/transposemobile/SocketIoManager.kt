package com.example.transposemobile

import android.util.Log
import io.socket.client.IO
import io.socket.client.Socket
import java.net.URISyntaxException
import kotlin.concurrent.fixedRateTimer

class SocketIoManager(private val imuDataProducer: ImuDataProducer,
                      private val onnxManager: OnnxManager) {

    private var isRunning: Boolean = true
    private lateinit var socket: Socket // Socket.IO 클라이언트

    init {
        try {
            socket = IO.socket("http://143.248.143.65:5555/")
            socket.connect()
            Log.d("SocketIO", "Connected to Socket.IO server")
        } catch (e: URISyntaxException) {
            e.printStackTrace()
            Log.e("SocketIO", "Failed to connect to Socket.IO server")
        }
    }

    fun startInference() {
        onnxManager.resetState()
        isRunning = true

        fixedRateTimer("InferenceTimer", false, 0L, 1000/100L) {
            try {
                val msg = onnxManager.getInferenceResult()
                if(msg != null) {
                    socket.emit("animation_data", msg)
                }
                else if (!imuDataProducer.isRunning) {
                    isRunning = false
                    cancel()
                }
                if (!isRunning) {
                    cancel()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun stopInference() {
        isRunning = false
    }
}