package com.dualcam.capture.storage

import com.dualcam.capture.imu.IMUReading
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * Thread-safe CSV logger.
 * Schema: timestamp_ns,frame_id,primary,secondary,primary_filename,secondary_filename,
 *         accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
 *
 * One row per stereo pair. Buffers each camera's data until the partner arrives.
 */
class CsvWriter(sessionDir: File) {

    private data class HalfRow(
        val timestampNs: Long,
        val filename: String,
        val imu: IMUReading?
    )

    private val writer = BufferedWriter(FileWriter(File(sessionDir, "imu.csv")))
    private val pendingPrimary = HashMap<Int, HalfRow>()
    private val pendingSecondary = HashMap<Int, HalfRow>()
    private var rowCount = 0

    init {
        writer.write(
            "timestamp_ns,frame_id,primary,secondary," +
            "primary_filename,secondary_filename," +
            "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
        )
        writer.flush()
    }

    @Synchronized
    fun writeRow(
        timestampNs: Long,
        frameId: Int,
        camera: String,
        filename: String,
        imu: IMUReading?
    ) {
        val half = HalfRow(timestampNs, filename, imu)
        if (camera == "primary") {
            val partner = pendingSecondary.remove(frameId)
            if (partner != null) {
                emit(frameId, half, partner)
            } else {
                pendingPrimary[frameId] = half
            }
        } else {
            val partner = pendingPrimary.remove(frameId)
            if (partner != null) {
                emit(frameId, partner, half)
            } else {
                pendingSecondary[frameId] = half
            }
        }
    }

    private fun emit(frameId: Int, primary: HalfRow, secondary: HalfRow) {
        val imu = primary.imu ?: secondary.imu
        val ax = imu?.accelX ?: 0f
        val ay = imu?.accelY ?: 0f
        val az = imu?.accelZ ?: 0f
        val gx = imu?.gyroX ?: 0f
        val gy = imu?.gyroY ?: 0f
        val gz = imu?.gyroZ ?: 0f
        writer.write(
            "${primary.timestampNs},$frameId,primary,secondary," +
            "${primary.filename},${secondary.filename}," +
            "$ax,$ay,$az,$gx,$gy,$gz\n"
        )
        rowCount++
        if (rowCount % 30 == 0) writer.flush()
    }

    fun flush() = writer.flush()

    fun close() {
        writer.flush()
        writer.close()
    }
}
