package com.example.transposemobile

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import android.webkit.WebView
import android.widget.Button
import android.widget.TextView
import androidx.lifecycle.Observer
import com.google.android.material.slider.Slider
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlin.concurrent.fixedRateTimer


class MainActivity : AppCompatActivity() {

    private lateinit var textView: TextView

    // 클래스 인스턴스 생성
    private var imuDataBuffer: ImuDataBuffer = ImuDataBuffer()
    private lateinit var imuDataProducer: ImuDataProducer
    private lateinit var onnxManager: OnnxManager

    private lateinit var socketIoManager: SocketIoManager
    private lateinit var ktorServerManager: KtorServerManager


    @SuppressLint("SetJavaScriptEnabled", "SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // UI elements
            val slider1: Slider = findViewById(R.id.slider1)
            val sliderValue1: TextView = findViewById(R.id.sliderValue1)
            val slider2: Slider = findViewById(R.id.slider2)
            val sliderValue2: TextView = findViewById(R.id.sliderValue2)

            val startBtn: Button = findViewById(R.id.startButton)
            val stopBtn: Button = findViewById(R.id.stopButton)
            val switchToggle: SwitchMaterial = findViewById(R.id.switchToggle)
            val webView: WebView = findViewById(R.id.webView)

            textView = findViewById(R.id.textView)


            // imuDataProducer 초기화
            imuDataProducer = ImuDataProducer(imuDataBuffer, assets, filesDir)

            // ONNX 초기화
            val modelPath = "transpose_net_250103_dynamic_batch.onnx"
            onnxManager = OnnxManager(imuDataBuffer, assets, modelPath)

            // socketIO 초기화
            socketIoManager = SocketIoManager(imuDataProducer, onnxManager)

            // Ktor 초기화
            ktorServerManager = KtorServerManager(imuDataProducer, onnxManager)


            fun startBtnCommonAction() {
                startBtn.visibility = View.GONE
                stopBtn.visibility = View.VISIBLE
                switchToggle.isEnabled = false // Switch 비활성화
                switchToggle.alpha = 0.7f      // 투명도를 줄여 비활성화 표시
                imuDataProducer.startBuffering()
            }
            fun stopBtnCommonAction() {
                stopBtn.visibility = View.GONE
                startBtn.visibility = View.VISIBLE
                switchToggle.isEnabled = true
                switchToggle.alpha = 1f
                imuDataProducer.stopBuffering()
            }

            // 서버에서 unity WebGL을 확인하는 방식
            fun switchOffAction() {
                webView.visibility = View.GONE
                ktorServerManager.stopServer()

                startBtn.setOnClickListener {
                    startBtnCommonAction()
                    socketIoManager.startInference()
                }
                stopBtn.setOnClickListener {
                    stopBtnCommonAction()
                    socketIoManager.stopInference()
                }
            }

            // 모바일에서 unity WebGL을 확인하는 방식
            @SuppressLint("ClickableViewAccessibility")
            fun switchOnAction() {
                ktorServerManager.startServer()

                // Setup WebView
                webView.visibility = View.VISIBLE
                webView.settings.javaScriptEnabled = true
                webView.setOnTouchListener { _, _ -> true }
                webView.loadUrl("http://localhost:5559/unityWebGL")

                startBtn.setOnClickListener {
                    startBtnCommonAction()
                    webView.loadUrl("javascript:document.getElementById('connectBtn').click();")
                }
                stopBtn.setOnClickListener {
                    stopBtnCommonAction()
                    webView.loadUrl("javascript:document.getElementById('disconnectBtn').click();")
                }
            }

            // 토글 액션
            switchToggle.setOnCheckedChangeListener { _, isChecked ->
                if (!isChecked) {
                    switchOffAction()
                } else {
                    switchOnAction()
                }
            }
            switchOffAction() // 최초 기본 상태

            // 슬라이더 조절
            slider1.addOnChangeListener { _, value, _ ->
                imuDataProducer.frequency = value.toLong()
                sliderValue1.text = "${value.toInt()} Hz"
            }
            slider2.addOnChangeListener { _, value, _ ->
                onnxManager.batchSize = value.toInt()
                sliderValue2.text = "${value.toInt()} Frame"
            }

            // LiveData 관찰
            imuDataProducer.eventLiveData.observe(this, Observer { eventOccurred ->
                if (eventOccurred) {
                    stopBtn.performClick() // Producer 실행이 끝나면 stop 버튼 자동으로 누르기
                }
            })

            fixedRateTimer("InferenceTimer", false, 0L, 2000L) {
                val (avg, min, max) = onnxManager.inferenceStats.getStats()
                if (avg >= 0) {
                    textView.text = "Average: ${"%.2f".format(avg)} ms, Min: $min ms, Max: $max ms"
                }
            }

        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        onnxManager.closeSession()
    }
}