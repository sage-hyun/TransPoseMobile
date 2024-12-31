package com.example.transposemobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import io.socket.client.IO
import io.socket.client.Socket
import org.json.JSONArray
import java.io.File
import java.net.URISyntaxException
import java.util.Optional
import kotlin.concurrent.fixedRateTimer

class MainActivity : AppCompatActivity() {

    private lateinit var textView: TextView
    private lateinit var session: ai.onnxruntime.OrtSession
    private lateinit var onnxEnv: OrtEnvironment

    private lateinit var accData: List<FloatArray>
    private lateinit var oriData: List<FloatArray>
    private var currentIndex = 0 // 현재 반복 인덱스
    private lateinit var socket: Socket // Socket.IO 클라이언트

    // 클래스 인스턴스 생성
    private val inferenceStats = InferenceStats()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // UI TextView 연결
        textView = findViewById(R.id.textView)

        try {
            // Socket.IO 초기화
            initializeSocket()

            // ONNX Runtime 환경 초기화
            onnxEnv = OrtEnvironment.getEnvironment()

            // 모델 파일 로드
            val assetManager = assets
//            val modelPath = "transpose_net_241217.onnx"
            val modelPath = "simplified_model_241217.onnx"
            val modelBytes = assetManager.open(modelPath).readBytes()
            session = onnxEnv.createSession(modelBytes)

            // JSON 데이터 로드
            accData = loadJsonArray("acc_240521.json")
            oriData = loadJsonArray("ori_240521.json")

            // 두 데이터의 길이가 다르면 예외 처리
            if (accData.size != oriData.size) {
                throw IllegalArgumentException("acc.json과 ori.json의 shape[0] 값이 다릅니다.")
            }

            // 1초 간격으로 추론 실행
            startInferenceLoop()

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun initializeSocket() {
        try {
            socket = IO.socket("http://143.248.143.65:5555/")
            socket.connect()
            Log.d("SocketIO", "Connected to Socket.IO server")
        } catch (e: URISyntaxException) {
            e.printStackTrace()
            Log.e("SocketIO", "Failed to connect to Socket.IO server")
        }
    }

    private fun startInferenceLoop() {
        fixedRateTimer("InferenceTimer", false, 1000L, 1000/100L) {
            try {
                // 현재 인덱스의 데이터를 가져옴
                if (currentIndex < accData.size) {
                    val acc = accData[currentIndex]
                    val ori = oriData[currentIndex]

                    // 1차원 배열을 2차원 배열로 변환 (1 x feature_size) 예: (18,) 대신 (1,18)
                    val acc2D = arrayOf(acc)
                    val ori2D = arrayOf(ori)

                    // ONNX Tensor로 변환
                    val accTensor = OnnxTensor.createTensor(onnxEnv, acc2D)
                    val oriTensor = OnnxTensor.createTensor(onnxEnv, ori2D)

                    // 모델 입력 설정
                    val inputs = mapOf(
                        "acc" to accTensor,
                        "ori" to oriTensor
                    )

                    // 모델 추론 실행
                    val startTime = System.currentTimeMillis() // 시작 시간 기록
                    val results = session.run(inputs)          // 추론 실행
                    val endTime = System.currentTimeMillis()   // 종료 시간 기록

                    // 실행 시간 계산 및 로그 출력
                    val duration = endTime - startTime
//                    Log.d("InferenceTime", "Model inference took $duration ms")
                    // 실행 시간 저장
                    inferenceStats.addDuration(duration)
                    // 통계 출력
                    val (avg, min, max) = inferenceStats.getStats()
                    Log.d("InferenceStats", "Average: ${"%.2f".format(avg)} ms, Min: $min ms, Max: $max ms")



                    // 결과 데이터 처리
                    val poseValue = results["pose"]
                    val tranValue = results["tran"]

                    // Optional에서 값을 안전하게 추출
                    val poseTensor = (poseValue as Optional<OnnxTensor>).orElse(null)
                    val tranTensor = (tranValue as Optional<OnnxTensor>).orElse(null)

                    // poseTensor와 tranTensor가 null이 아닌 경우만 처리
                    if (poseTensor != null && tranTensor != null) {
                        // 텐서를 배열로 변환
                        val pose = poseTensor.floatBuffer.array()
                        val tran = tranTensor.floatBuffer.array()


                        // Socket.IO로 데이터 전송
                        val s = pose.joinToString(",") + "#" + tran.joinToString(",") + "$"
                        socket.emit("animation_data", s)
                        Log.d("SocketIO", "Sent data: $s")


                        // UI 업데이트
                        val poseText = pose.joinToString(", ")
                        val tranText = tran.joinToString(", ")

                        // UI 업데이트
                        runOnUiThread {
                            textView.text = "Pose: $poseText\n\nTrans: $tranText\n\n"
                        }
//                        Log.d("output", "Pose: $poseText Trans: $tranText")

                        // Tensor 리소스 정리
                        poseTensor.close()
                        tranTensor.close()

                    } else {
                        // Optional 값이 없는 경우 처리
                        runOnUiThread {
                            textView.text = "Pose or Tran output is empty."
                        }
                    }

                    // 다음 인덱스로 이동
                    currentIndex++
                } else {
                    // 반복 종료
                    cancel()
                }

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }


    // Helper: JSON 파일을 읽고 FloatArray 리스트로 변환
    private fun loadJsonArray(fileName: String): List<FloatArray> {
        val file = File(filesDir, fileName)
        if (!file.exists()) {
            assets.open(fileName).use { inputStream ->
                file.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        // JSON 파일 읽기
        val jsonData = file.readText()
        val jsonArray = JSONArray(jsonData)
        val resultList = mutableListOf<FloatArray>()

        // 각 row를 FloatArray로 변환하여 리스트에 추가
        for (i in 0 until jsonArray.length()) {
            val rowArray = jsonArray.getJSONArray(i)
            val row = FloatArray(rowArray.length()) { j -> rowArray.getDouble(j).toFloat() }
            resultList.add(row)
        }
        return resultList
    }

    override fun onDestroy() {
        super.onDestroy()
        session.close()
    }
}

class InferenceStats {
    private val durations = mutableListOf<Long>() // 실행 시간 저장

    // 실행 시간 추가
    fun addDuration(duration: Long) {
        durations.add(duration)
    }

    // 평균, 최소, 최대 계산
    fun getStats(): Triple<Double, Long, Long> {
        if (durations.isEmpty()) return Triple(0.0, 0, 0) // 데이터가 없을 경우 처리

        val avg = durations.average() // 평균 계산
        val min = durations.minOrNull() ?: 0 // 최소값 계산
        val max = durations.maxOrNull() ?: 0 // 최대값 계산
        return Triple(avg, min, max)
    }
}