package com.dualcam.capture.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import java.util.concurrent.Executor

class DualCameraController(
    private val context: Context,
    private val setup: CameraSetup,
    private val onFrameCaptured: (cameraIndex: Int, jpegBytes: ByteArray, timestampNs: Long) -> Unit,
    private val onError: (String) -> Unit = {}
) {

    companion object {
        const val CAPTURE_WIDTH = 1280
        const val CAPTURE_HEIGHT = 720
        const val PREVIEW_WIDTH = 640
        const val PREVIEW_HEIGHT = 360
        private const val BUFFER_DEPTH = 3
    }

    private val cameraThread = HandlerThread("DualCamThread").also { it.start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private val cameraExecutor = Executor { cameraHandler.post(it) }

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var primaryReader: ImageReader? = null
    private var secondaryReader: ImageReader? = null

    @SuppressLint("MissingPermission")
    fun startCapture(primaryPreviewSurface: Surface?, secondaryPreviewSurface: Surface?) {
        primaryReader = makeReader(0)
        secondaryReader = makeReader(1)

        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        manager.openCamera(setup.logicalCameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createSession(camera, primaryPreviewSurface, secondaryPreviewSurface)
            }
            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                onError("Camera disconnected")
            }
            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                onError("Camera error: $error")
            }
        }, cameraHandler)
    }

    private fun makeReader(index: Int): ImageReader =
        ImageReader.newInstance(CAPTURE_WIDTH, CAPTURE_HEIGHT, ImageFormat.JPEG, BUFFER_DEPTH).also { reader ->
            reader.setOnImageAvailableListener({ r ->
                val image = r.acquireLatestImage() ?: return@setOnImageAvailableListener
                val ts = image.timestamp
                val buf = image.planes[0].buffer
                val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
                image.close()
                onFrameCaptured(index, bytes, ts)
            }, cameraHandler)
        }

    private fun createSession(
        camera: CameraDevice,
        primaryPreview: Surface?,
        secondaryPreview: Surface?
    ) {
        val outputConfigs = mutableListOf(
            OutputConfiguration(primaryReader!!.surface).also { it.setPhysicalCameraId(setup.primaryPhysicalId) },
            OutputConfiguration(secondaryReader!!.surface).also { it.setPhysicalCameraId(setup.secondaryPhysicalId) }
        )
        primaryPreview?.let {
            outputConfigs += OutputConfiguration(it).also { cfg -> cfg.setPhysicalCameraId(setup.primaryPhysicalId) }
        }
        secondaryPreview?.let {
            outputConfigs += OutputConfiguration(it).also { cfg -> cfg.setPhysicalCameraId(setup.secondaryPhysicalId) }
        }

        camera.createCaptureSession(SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR,
            outputConfigs,
            cameraExecutor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    startRepeating(session, camera, primaryPreview, secondaryPreview)
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    onError("Session configuration failed")
                }
            }
        ))
    }

    private fun startRepeating(
        session: CameraCaptureSession,
        camera: CameraDevice,
        primaryPreview: Surface?,
        secondaryPreview: Surface?
    ) {
        val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(primaryReader!!.surface)
            addTarget(secondaryReader!!.surface)
            primaryPreview?.let { addTarget(it) }
            secondaryPreview?.let { addTarget(it) }
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        }.build()

        session.setRepeatingRequest(request, null, cameraHandler)
    }

    fun stopCapture() {
        runCatching { captureSession?.stopRepeating() }
        runCatching { captureSession?.close() }
        runCatching { cameraDevice?.close() }
        runCatching { primaryReader?.close() }
        runCatching { secondaryReader?.close() }
        cameraThread.quitSafely()
    }
}
