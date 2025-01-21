package com.example.transposemobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.util.Log
import java.util.Optional


class OnnxManager(private val dataBuffer: ImuDataBuffer,
                  assets: AssetManager, modelPath: String) {

    private lateinit var session: OrtSession
    private val onnxEnv: OrtEnvironment = OrtEnvironment.getEnvironment()

    var batchSize = 1 // input batch size

    // Stateful input/outputs
    private lateinit var pastFramesTensor: OnnxTensor
    private lateinit var hStateTensor: OnnxTensor
    private lateinit var cStateTensor: OnnxTensor
    private lateinit var rootYTensor: OnnxTensor
    private lateinit var lFootPosTensor: OnnxTensor
    private lateinit var rFootPosTensor: OnnxTensor
    private lateinit var tranTensor: OnnxTensor

    // 클래스 인스턴스 생성
    private val inferenceStats = InferenceStats()

    init {
        // 모델 파일 로드
        val modelBytes = assets.open(modelPath).readBytes()
        session = onnxEnv.createSession(modelBytes)
    }

    fun resetState() {
        // 혹시 이미 Tensor가 있다면 메모리 해제
        if (::tranTensor.isInitialized) { tranTensor.close() }
        if (::pastFramesTensor.isInitialized) { pastFramesTensor.close() }
        if (::hStateTensor.isInitialized) { hStateTensor.close() }
        if (::cStateTensor.isInitialized) { cStateTensor.close() }
        if (::rootYTensor.isInitialized) { rootYTensor.close() }
        if (::lFootPosTensor.isInitialized) { lFootPosTensor.close() }
        if (::rFootPosTensor.isInitialized) { rFootPosTensor.close() }

        // 초기 값 설정
        val tran = FloatArray(3) { 0f }             // 3D 벡터, 모두 0으로 초기화
        val pastFrames = Array(26) { FloatArray(72) { 0f } } // 26x72 크기의 배열, 모두 0으로 초기화
        val hState = Array(2) { FloatArray(256) { 0f } }     // 2x256 크기의 배열, 모두 0으로 초기화
        val cState = Array(2) { FloatArray(256) { 0f } }     // 2x256 크기의 배열, 모두 0으로 초기화
        val rootY = floatArrayOf(0.0f)              // 단일 값
        val lFootPos = floatArrayOf(0.1283f, -0.9559f, 0.0750f) // 3D 벡터
        val rFootPos = floatArrayOf(-0.1194f, -0.9564f, 0.0774f) // 3D 벡터

        // ONNX Tensor로 변환
        tranTensor = OnnxTensor.createTensor(onnxEnv, tran)
        pastFramesTensor = OnnxTensor.createTensor(onnxEnv, pastFrames)
        hStateTensor = OnnxTensor.createTensor(onnxEnv, hState)
        cStateTensor = OnnxTensor.createTensor(onnxEnv, cState)
        rootYTensor = OnnxTensor.createTensor(onnxEnv, rootY)
        lFootPosTensor = OnnxTensor.createTensor(onnxEnv, lFootPos)
        rFootPosTensor = OnnxTensor.createTensor(onnxEnv, rFootPos)
    }


    fun getInferenceResult(): String? {
        try {

            // dynamic batch size 방식
            val acc2D = mutableListOf<FloatArray>()
            val ori2D = mutableListOf<FloatArray>()
            while(acc2D.size < batchSize) {
                val accData = dataBuffer.accQueue.poll()
                if (accData != null) acc2D.add(accData) // null 확인 후 추가
            }
            while(ori2D.size < batchSize) {
                val oriData = dataBuffer.oriQueue.poll()
                if (oriData != null) ori2D.add(oriData) // null 확인 후 추가
            }

            // ONNX Tensor로 변환
            val accTensor = OnnxTensor.createTensor(onnxEnv, acc2D.toTypedArray())
            val oriTensor = OnnxTensor.createTensor(onnxEnv, ori2D.toTypedArray())

            // 모델 입력 설정
            val inputs = mapOf(
                "acc" to accTensor,
                "ori" to oriTensor,
                "tran_in" to tranTensor,
                "past_frames_in" to pastFramesTensor,
                "h_state_in" to hStateTensor,
                "c_state_in" to cStateTensor,
                "root_y_in" to rootYTensor,
                "lfoot_pos_in" to lFootPosTensor,
                "rfoot_pos_in" to rFootPosTensor
            )

            // 모델 추론 실행
            val startTime = System.currentTimeMillis() // 시작 시간 기록
            val results = session.run(inputs)          // 추론 실행
            val endTime = System.currentTimeMillis()   // 종료 시간 기록

            // 실행 시간 계산 및 로그 출력
            val duration = endTime - startTime
            Log.d("InferenceTime", "Model inference took $duration ms")
            // 실행 시간 저장
            inferenceStats.addDuration(duration)
            // 통계 출력
            val (avg, min, max) = inferenceStats.getStats()
            Log.d("InferenceStats", "Average: ${"%.2f".format(avg)} ms, Min: $min ms, Max: $max ms")


            // input Tensor 리소스 정리 (특히 global 변수들은 새로운 값 받기 전에 메모리 해제 필수)
            accTensor.close()
            oriTensor.close()
            tranTensor.close()
            pastFramesTensor.close()
            hStateTensor.close()
            cStateTensor.close()
            rootYTensor.close()
            lFootPosTensor.close()
            rFootPosTensor.close()


            // 결과 데이터 처리 - Optional에서 값을 안전하게 추출
            val poseTensor = (results["pose"] as Optional<OnnxTensor>).orElse(null)
            tranTensor = (results["tran_out"] as Optional<OnnxTensor>).orElse(null)
            pastFramesTensor = (results["past_frames_out"] as Optional<OnnxTensor>).orElse(null)
            hStateTensor = (results["h_state_out"] as Optional<OnnxTensor>).orElse(null)
            cStateTensor = (results["c_state_out"] as Optional<OnnxTensor>).orElse(null)
            rootYTensor = (results["root_y_out"] as Optional<OnnxTensor>).orElse(null)
            lFootPosTensor = (results["lfoot_pos_out"] as Optional<OnnxTensor>).orElse(null)
            rFootPosTensor = (results["rfoot_pos_out"] as Optional<OnnxTensor>).orElse(null)


            // poseTensor와 tranTensor가 null이 아닌 경우만 처리
            if (poseTensor != null && tranTensor != null) {
                // 텐서를 배열로 변환
                val poseMatrix = poseTensor.floatBuffer.array()
                val pose = MathUtils.rotationMatrixToRodriguesOpenCV(poseMatrix)

                val tran = tranTensor.floatBuffer.array()

                // Tensor 리소스 정리
                poseTensor.close()
//                        tranTensor.close()

                // Socket.IO로 데이터 전송
                val s = pose.joinToString(",") + "#" + tran.joinToString(",") + "$"
                return s
//                    socket.emit("animation_data", s)
//                    Log.d("SocketIO", "Sent data: $s")



                // UI 업데이트
//                    val poseText = pose.joinToString(", ")
//                    val tranText = tran.joinToString(", ")
//                        runOnUiThread {
//                            textView.text = "Pose: $poseText\n\nTran: $tranText\n\n"
//                        }
//                    Log.d("output", "Pose: $poseText Tran: $tranText")

            } else {
                // Optional 값이 없는 경우 처리
//                    runOnUiThread {
//                        textView.text = "Pose or Tran output is empty."
//                    }
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
        return null
    }

    fun closeSession() {
        if (::session.isInitialized) session.close()
    }
}
