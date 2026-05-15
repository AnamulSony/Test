package com.dualcam.capture.imu

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs

data class IMUReading(
    val timestampNs: Long,
    val accelX: Float, val accelY: Float, val accelZ: Float,
    val gyroX: Float, val gyroY: Float, val gyroZ: Float
)

class IMUController(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Lock-free circular buffer — sized for ~1 second at 200 Hz
    private val BUFFER_SIZE = 256
    private val timestamps = LongArray(BUFFER_SIZE)
    private val accelBuf = Array(BUFFER_SIZE) { FloatArray(3) }
    private val gyroBuf = Array(BUFFER_SIZE) { FloatArray(3) }
    private val writeIdx = AtomicInteger(0)

    private val latestAccel = FloatArray(3)
    private var imuCount = 0L
    private var startTimeNs = 0L
    var currentHz = 0f
        private set

    fun start() {
        imuCount = 0L
        startTimeNs = System.nanoTime()
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        updateHz()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAccel[0] = event.values[0]
                latestAccel[1] = event.values[1]
                latestAccel[2] = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                val idx = writeIdx.getAndIncrement() and (BUFFER_SIZE - 1) // fast modulo (power of 2)
                timestamps[idx] = event.timestamp
                accelBuf[idx][0] = latestAccel[0]; accelBuf[idx][1] = latestAccel[1]; accelBuf[idx][2] = latestAccel[2]
                gyroBuf[idx][0] = event.values[0]; gyroBuf[idx][1] = event.values[1]; gyroBuf[idx][2] = event.values[2]
                imuCount++
                if (imuCount % 100L == 0L) updateHz()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    private fun updateHz() {
        val elapsedSec = (System.nanoTime() - startTimeNs) / 1_000_000_000f
        if (elapsedSec > 0f) currentHz = imuCount / elapsedSec
    }

    /**
     * Returns the IMU reading whose timestamp is closest to [queryNs].
     * Searches up to 64 entries back from the current write position.
     */
    fun getReadingNearestTo(queryNs: Long): IMUReading? {
        val last = (writeIdx.get() - 1) and (BUFFER_SIZE - 1)
        if (timestamps[last] == 0L) return null

        var bestIdx = last
        var bestDelta = abs(timestamps[last] - queryNs)

        for (i in 1 until 64) {
            val idx = (last - i) and (BUFFER_SIZE - 1)
            if (timestamps[idx] == 0L) break
            val delta = abs(timestamps[idx] - queryNs)
            if (delta < bestDelta) { bestDelta = delta; bestIdx = idx } else break
        }

        return IMUReading(
            timestampNs = timestamps[bestIdx],
            accelX = accelBuf[bestIdx][0], accelY = accelBuf[bestIdx][1], accelZ = accelBuf[bestIdx][2],
            gyroX = gyroBuf[bestIdx][0], gyroY = gyroBuf[bestIdx][1], gyroZ = gyroBuf[bestIdx][2]
        )
    }
}
