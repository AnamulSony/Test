package com.dualcam.capture

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.Surface
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.dualcam.capture.camera.CameraDiscovery
import com.dualcam.capture.camera.CameraSetup
import com.dualcam.capture.camera.DualCameraController
import com.dualcam.capture.databinding.ActivityMainBinding
import com.dualcam.capture.imu.IMUController
import com.dualcam.capture.storage.CsvWriter
import com.dualcam.capture.storage.SessionDirs
import com.dualcam.capture.storage.StorageManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val storageManager by lazy { StorageManager(this) }
    private val imuController by lazy { IMUController(this) }

    private var dualCamera: DualCameraController? = null
    @Volatile private var csvWriter: CsvWriter? = null
    @Volatile private var sessionDirs: SessionDirs? = null
    @Volatile private var isCapturing = false

    private val primaryCount = AtomicInteger(0)
    private val secondaryCount = AtomicInteger(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnCapture.setOnClickListener {
            if (isCapturing) stopCapture() else requestPermissionThenStart()
        }
        binding.btnInfo.setOnClickListener { showCameraInfo() }
        binding.btnFiles.setOnClickListener { showFilesDialog() }
    }

    // ── Permission ────────────────────────────────────────────────────────────

    private fun requestPermissionThenStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCapture()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startCapture()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    // ── Start / Stop ──────────────────────────────────────────────────────────

    private fun startCapture() {
        binding.btnCapture.isEnabled = false
        binding.tvStatus.text = "Detecting cameras…"

        lifecycleScope.launch(Dispatchers.IO) {
            val setup = CameraDiscovery.findDualRearSetup(this@MainActivity)
            withContext(Dispatchers.Main) {
                binding.btnCapture.isEnabled = true
                if (setup == null) {
                    binding.tvStatus.text = "No dual rear camera found"
                    Toast.makeText(
                        this@MainActivity,
                        "No dual rear camera found. Tap Info for details.",
                        Toast.LENGTH_LONG
                    ).show()
                    return@withContext
                }
                initSession(setup)
            }
        }
    }

    private fun initSession(setup: CameraSetup) {
        val dirs = storageManager.createSession()
        sessionDirs = dirs
        csvWriter = CsvWriter(dirs.sessionDir)
        primaryCount.set(0)
        secondaryCount.set(0)

        lifecycleScope.launch(Dispatchers.IO) { storageManager.writeCalibJson(dirs, setup) }

        val primarySurface = binding.texturePrimary.surfaceTexture?.let {
            it.setDefaultBufferSize(DualCameraController.PREVIEW_WIDTH, DualCameraController.PREVIEW_HEIGHT)
            Surface(it)
        }
        val secondarySurface = binding.textureSecondary.surfaceTexture?.let {
            it.setDefaultBufferSize(DualCameraController.PREVIEW_WIDTH, DualCameraController.PREVIEW_HEIGHT)
            Surface(it)
        }

        dualCamera = DualCameraController(
            context = this,
            setup = setup,
            onFrameCaptured = ::onFrame,
            onError = { msg ->
                lifecycleScope.launch(Dispatchers.Main) {
                    binding.tvStatus.text = "Error: $msg"
                    if (isCapturing) stopCapture()
                }
            }
        ).also { it.startCapture(primarySurface, secondarySurface) }

        imuController.start()
        isCapturing = true

        binding.recordingDot.setBackgroundResource(R.drawable.dot_recording)
        binding.btnCapture.text = "STOP CAPTURE"
        binding.btnCapture.backgroundTintList =
            android.content.res.ColorStateList.valueOf(android.graphics.Color.parseColor("#C62828"))
        binding.tvSetup.text = "Wide ${setup.primaryFocalMm}mm  +  UW ${setup.secondaryFocalMm}mm"
        binding.tvStatus.text = "Recording → ${dirs.sessionDir.name.takeLast(15)}"
        binding.tvFrameCount.text = "Primary: 0   Secondary: 0"
        binding.tvImuHz.text = "IMU: --"
    }

    private fun stopCapture() {
        dualCamera?.stopCapture()
        dualCamera = null
        imuController.stop()
        csvWriter?.close()
        csvWriter = null
        isCapturing = false

        val p = primaryCount.get(); val s = secondaryCount.get()
        binding.recordingDot.setBackgroundResource(R.drawable.dot_idle)
        binding.btnCapture.text = "START CAPTURE"
        binding.btnCapture.backgroundTintList =
            android.content.res.ColorStateList.valueOf(android.graphics.Color.parseColor("#1976D2"))
        binding.tvStatus.text = "Saved — $p primary, $s secondary frames"
        binding.tvFrameCount.text = "Primary: $p   Secondary: $s"
    }

    // ── Frame callback ────────────────────────────────────────────────────────

    private fun onFrame(cameraIndex: Int, jpegBytes: ByteArray, timestampNs: Long) {
        val dirs = sessionDirs ?: return
        val imu = imuController.getReadingNearestTo(timestampNs)
        val camera = if (cameraIndex == 0) "primary" else "secondary"
        val frameId = if (cameraIndex == 0) primaryCount.getAndIncrement() else secondaryCount.getAndIncrement()

        lifecycleScope.launch(Dispatchers.IO) {
            runCatching {
                val filename = storageManager.saveFrame(jpegBytes, dirs, cameraIndex, frameId)
                csvWriter?.writeRow(timestampNs, frameId, camera, filename, imu)
            }
            val p = primaryCount.get(); val s = secondaryCount.get()
            if ((p + s) % 5 == 0) {
                withContext(Dispatchers.Main) {
                    binding.tvFrameCount.text = "Primary: $p   Secondary: $s"
                    binding.tvImuHz.text = "IMU: ${"%.0f".format(imuController.currentHz)} Hz"
                }
            }
        }
    }

    // ── Files dialog ──────────────────────────────────────────────────────────

    private fun showFilesDialog() {
        lifecycleScope.launch(Dispatchers.IO) {
            val baseDir = File(getExternalFilesDir(null), "DualCamCapture")
            val sessions = baseDir.listFiles()
                ?.filter { it.isDirectory }
                ?.sortedByDescending { it.lastModified() }
                ?: emptyList()

            withContext(Dispatchers.Main) {
                if (sessions.isEmpty()) {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("No Sessions")
                        .setMessage("No captured sessions yet.\nTap START CAPTURE to begin.")
                        .setPositiveButton("OK", null)
                        .show()
                    return@withContext
                }
                showSessionList(sessions, baseDir)
            }
        }
    }

    private fun showSessionList(sessions: List<File>, baseDir: File) {
        val labels = sessions.map { session ->
            val p = File(session, "primary").listFiles()?.size ?: 0
            val s = File(session, "secondary").listFiles()?.size ?: 0
            val sizeMb = session.walkBottomUp().filter { it.isFile }.map { it.length() }.sum() / 1_048_576.0
            "📁 ${session.name.removePrefix("session_")}\n   P:$p  S:$s frames  •  ${"%.1f".format(sizeMb)} MB"
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle("Sessions  (${sessions.size})")
            .setItems(labels) { _, index ->
                showSessionOptions(sessions[index], baseDir)
            }
            .setNeutralButton("🗑 Clear All") { _, _ ->
                confirmClearAll(sessions, baseDir)
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showSessionOptions(session: File, baseDir: File) {
        val p = File(session, "primary").listFiles()?.size ?: 0
        val s = File(session, "secondary").listFiles()?.size ?: 0
        val hasCalib = File(session, "calib.json").exists()
        val hasCsv   = File(session, "imu.csv").exists()
        val sizeMb   = session.walkBottomUp().filter { it.isFile }.map { it.length() }.sum() / 1_048_576.0
        val adbCmd   = "adb pull \"${session.absolutePath}\""

        val detail = buildString {
            appendLine("Primary:   $p frames")
            appendLine("Secondary: $s frames")
            appendLine("Size:      ${"%.1f".format(sizeMb)} MB")
            appendLine("calib.json ${if (hasCalib) "✅" else "❌"}")
            appendLine("imu.csv    ${if (hasCsv)   "✅" else "❌"}")
            appendLine()
            appendLine("ADB command:")
            append(adbCmd)
        }

        AlertDialog.Builder(this)
            .setTitle(session.name.removePrefix("session_"))
            .setMessage(detail)
            .setPositiveButton("Copy ADB") { _, _ ->
                val cb = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cb.setPrimaryClip(ClipData.newPlainText("adb", adbCmd))
                Toast.makeText(this, "ADB command copied!", Toast.LENGTH_SHORT).show()
            }
            .setNeutralButton("🗑 Delete") { _, _ ->
                AlertDialog.Builder(this)
                    .setTitle("Delete Session?")
                    .setMessage("Delete ${session.name}?\n$p primary + $s secondary frames will be lost.")
                    .setPositiveButton("Delete") { _, _ ->
                        lifecycleScope.launch(Dispatchers.IO) {
                            session.deleteRecursively()
                            val remaining = baseDir.listFiles()
                                ?.filter { it.isDirectory }
                                ?.sortedByDescending { it.lastModified() }
                                ?: emptyList()
                            withContext(Dispatchers.Main) {
                                Toast.makeText(this@MainActivity, "Session deleted", Toast.LENGTH_SHORT).show()
                                if (remaining.isNotEmpty()) showSessionList(remaining, baseDir)
                            }
                        }
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
            .setNegativeButton("Back") { _, _ -> showSessionList(
                baseDir.listFiles()?.filter { it.isDirectory }?.sortedByDescending { it.lastModified() } ?: emptyList(),
                baseDir
            )}
            .show()
    }

    private fun confirmClearAll(sessions: List<File>, baseDir: File) {
        val totalFrames = sessions.sumOf {
            (File(it, "primary").listFiles()?.size ?: 0) +
            (File(it, "secondary").listFiles()?.size ?: 0)
        }
        AlertDialog.Builder(this)
            .setTitle("Clear All Sessions?")
            .setMessage("${sessions.size} session(s) • $totalFrames total frames will be permanently deleted.")
            .setPositiveButton("Delete All") { _, _ ->
                lifecycleScope.launch(Dispatchers.IO) {
                    sessions.forEach { it.deleteRecursively() }
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "All sessions cleared", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ── Camera info ───────────────────────────────────────────────────────────

    private fun showCameraInfo() {
        lifecycleScope.launch(Dispatchers.IO) {
            val info = CameraDiscovery.getAllCameraInfo(this@MainActivity)
            withContext(Dispatchers.Main) {
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Camera Info")
                    .setMessage(info)
                    .setPositiveButton("OK", null)
                    .show()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (isCapturing) stopCapture()
    }

    companion object {
        private const val REQ_CAMERA = 100
    }
}
