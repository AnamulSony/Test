package com.dualcam.capture.storage

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import com.dualcam.capture.camera.CameraSetup
import com.dualcam.capture.camera.DualCameraController
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

data class SessionDirs(
    val sessionDir: File,
    val primaryDir: File,
    val secondaryDir: File
)

class StorageManager(private val context: Context) {

    private val baseDir = File(context.getExternalFilesDir(null), "DualCamCapture")

    fun createSession(): SessionDirs {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val sessionDir = File(baseDir, "session_$ts")
        val primaryDir = File(sessionDir, "primary")
        val secondaryDir = File(sessionDir, "secondary")
        primaryDir.mkdirs()
        secondaryDir.mkdirs()
        return SessionDirs(sessionDir, primaryDir, secondaryDir)
    }

    fun saveFrame(jpegBytes: ByteArray, dirs: SessionDirs, cameraIndex: Int, frameId: Int): String {
        val dir = if (cameraIndex == 0) dirs.primaryDir else dirs.secondaryDir
        val filename = "%06d.jpg".format(frameId)
        File(dir, filename).writeBytes(jpegBytes)
        return filename
    }

    /** Writes camera intrinsics metadata matching the POC calib.json format. */
    fun writeCalibJson(dirs: SessionDirs, setup: CameraSetup) {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        fun cameraJson(physId: String): String {
            val c = runCatching { manager.getCameraCharacteristics(physId) }.getOrNull()
            val focalLengths = c?.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList() ?: emptyList<Float>()
            val apertures = c?.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES)?.toList() ?: emptyList<Float>()
            val sensorSize = c?.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            val pixelArray = c?.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
            return """{"camera_id":"$physId","focal_lengths_mm":$focalLengths,"apertures":$apertures,"sensor_size_mm":[${sensorSize?.width},${sensorSize?.height}],"sensor_pixels":[${pixelArray?.width},${pixelArray?.height}]}"""
        }

        val json = """
{
  "capture_width": ${DualCameraController.CAPTURE_WIDTH},
  "capture_height": ${DualCameraController.CAPTURE_HEIGHT},
  "logical_camera_id": "${setup.logicalCameraId}",
  "primary": ${cameraJson(setup.primaryPhysicalId)},
  "secondary": ${cameraJson(setup.secondaryPhysicalId)}
}
""".trimIndent()

        File(dirs.sessionDir, "calib.json").writeText(json)
    }
}
