package com.example.transposemobile

import android.content.res.AssetManager
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File

class ImuDataProducer(private val dataBuffer: ImuDataBuffer,
                      private val assets: AssetManager, private val filesDir: File) {

    private lateinit var accData: List<FloatArray>
    private lateinit var oriData: List<FloatArray>

    var isRunning = true // 실행 여부를 제어하는 플래그
    private var currentIndex = 0 // 현재 반복 인덱스

    // LiveData를 통해 이벤트 전달
    private val _eventLiveData = MutableLiveData<Boolean>()
    val eventLiveData: LiveData<Boolean> get() = _eventLiveData

    init {
        try {
            // JSON 데이터 로드
            accData = loadJsonArray("acc_240521.json")
            oriData = loadJsonArray("ori_240521.json")

            // 두 데이터의 길이가 다르면 예외 처리
            if (accData.size != oriData.size) {
                throw IllegalArgumentException("acc.json과 ori.json의 shape[0] 값이 다릅니다.")
            }

        } catch (e: Exception) {
            e.printStackTrace()


        }
    }

    fun startBuffering() {
        GlobalScope.launch {
            // 초기화
            currentIndex = 0
            isRunning = true
            dataBuffer.accQueue.clear()
            dataBuffer.oriQueue.clear()

            while(isRunning && (currentIndex < accData.size)) {

                // 현재 인덱스의 데이터를 버퍼에 추가
                dataBuffer.accQueue.add(accData[currentIndex])
                dataBuffer.oriQueue.add(oriData[currentIndex])
                currentIndex++

                // 버퍼 크기 제한
                dataBuffer.limitBufferSize(100)

                // 10ms 대기
                delay(10)
            }

            // 반복 종료
            Log.d("getImuData", "finished reading.")
            isRunning = false

            // 작업이 끝났다는 사실을 MainThread로 LiveData 전달
            withContext(Dispatchers.Main) {
                _eventLiveData.value = true // 작업 완료 이벤트 발생
            }
        }
    }

    fun stopBuffering() {
        isRunning = false
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
}