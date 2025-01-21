package com.example.transposemobile

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