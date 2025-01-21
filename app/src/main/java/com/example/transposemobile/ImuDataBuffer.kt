package com.example.transposemobile

import java.util.concurrent.ConcurrentLinkedQueue

// 공유 데이터 버퍼 클래스
class ImuDataBuffer {
    val accQueue = ConcurrentLinkedQueue<FloatArray>()
    val oriQueue = ConcurrentLinkedQueue<FloatArray>()

    // 버퍼 크기 제한 함수
    fun limitBufferSize(maxSize: Int) {
        while (accQueue.size > maxSize) accQueue.poll()
        while (oriQueue.size > maxSize) oriQueue.poll()
    }
}

