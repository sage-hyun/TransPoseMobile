package com.example.transposemobile

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.calib3d.Calib3d

object MathUtils {
    fun rotationMatrixToRodriguesOpenCV(matrices: FloatArray): FloatArray {
        // 총 24개의 3x3 행렬로 구성된 1D 리스트
        require(matrices.size == 216) { "Input must be a 1D list of size 216 (24 * 3 * 3)." }

        val rodriguesVectors = mutableListOf<Float>()

        for (i in 0 until 24) {
            // 각 3x3 행렬 추출
            val rotationMatrix = Mat(3, 3, CvType.CV_32F)
            for (row in 0 until 3) {
                for (col in 0 until 3) {
                    val index = i * 9 + row * 3 + col
                    rotationMatrix.put(row, col, matrices[index].toDouble())
                }
            }

            // Rodrigues 변환 수행
            val rodVector = Mat()
            Calib3d.Rodrigues(rotationMatrix, rodVector)

            // 결과를 1D 리스트로 변환하여 저장
            for (j in 0 until 3) {
                rodriguesVectors.add(rodVector[j, 0][0].toFloat())
            }
        }

        // 1D 결과 리스트로 변환
        return rodriguesVectors.toFloatArray()
    }
}
