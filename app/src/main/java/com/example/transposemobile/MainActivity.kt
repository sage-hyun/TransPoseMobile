package com.example.transposemobile

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import android.webkit.WebView
import android.widget.Button
import android.widget.TextView
import androidx.lifecycle.Observer
import com.google.android.material.switchmaterial.SwitchMaterial


class MainActivity : AppCompatActivity() {

    private lateinit var textView: TextView

    // 클래스 인스턴스 생성
    private var imuDataBuffer: ImuDataBuffer = ImuDataBuffer()
    private lateinit var imuDataProducer: ImuDataProducer
    private lateinit var onnxManager: OnnxManager

    private lateinit var socketIoManager: SocketIoManager
    private lateinit var ktorServerManager: KtorServerManager


    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            // UI elements
            val startBtn: Button = findViewById(R.id.startButton)
            val stopBtn: Button = findViewById(R.id.stopButton)
            val switchToggle: SwitchMaterial = findViewById(R.id.switchToggle)
            val webView: WebView = findViewById(R.id.webView)

            // imuDataProducer 초기화
            imuDataProducer = ImuDataProducer(imuDataBuffer, assets, filesDir)

            // ONNX 초기화
            val modelPath = "transpose_net_250103_dynamic_batch.onnx"
            onnxManager = OnnxManager(imuDataBuffer, assets, modelPath)
            onnxManager.batchSize = 1

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
            fun switchOnAction() {
                ktorServerManager.startServer()

                // Setup WebView
                webView.visibility = View.VISIBLE
                webView.settings.javaScriptEnabled = true
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

            // LiveData 관찰 - Producer 실행이 끝나면 stop 버튼 자동으로 누르기
            imuDataProducer.eventLiveData.observe(this, Observer { eventOccurred ->
                if (eventOccurred) {
                    stopBtn.performClick()
                }
            })

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