package com.dualcam.capture.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata

data class CameraSetup(
    val logicalCameraId: String,
    val primaryPhysicalId: String,   // wide (main)
    val secondaryPhysicalId: String, // ultrawide
    val primaryFocalMm: Float,
    val secondaryFocalMm: Float
)

object CameraDiscovery {

    /**
     * Finds a rear logical multi-camera and selects two physical sub-cameras:
     * primary = wide (second-shortest focal), secondary = ultrawide (shortest focal).
     * Returns null if no suitable logical multi-camera is found.
     */
    fun findDualRearSetup(context: Context): CameraSetup? {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        for (cameraId in manager.cameraIdList) {
            val chars = manager.getCameraCharacteristics(cameraId)

            if (chars.get(CameraCharacteristics.LENS_FACING) != CameraMetadata.LENS_FACING_BACK) continue

            val capabilities = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: continue
            if (!capabilities.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)) continue

            val physicalIds = chars.physicalCameraIds
            if (physicalIds.size < 2) continue

            // Map each physical camera to its shortest focal length, drop failures
            val physicalInfoList = physicalIds.mapNotNull { physId ->
                runCatching {
                    val pChars = manager.getCameraCharacteristics(physId)
                    val focal = pChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.minOrNull() ?: return@mapNotNull null
                    Pair(physId, focal)
                }.getOrNull()
            }.sortedBy { it.second } // ascending: ultrawide first, telephoto last

            if (physicalInfoList.size < 2) continue

            // index 0 = ultrawide (secondary), index 1 = wide/main (primary)
            return CameraSetup(
                logicalCameraId = cameraId,
                primaryPhysicalId = physicalInfoList[1].first,
                secondaryPhysicalId = physicalInfoList[0].first,
                primaryFocalMm = physicalInfoList[1].second,
                secondaryFocalMm = physicalInfoList[0].second
            )
        }
        return null
    }

    fun getAllCameraInfo(context: Context): String {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val sb = StringBuilder()

        manager.cameraIdList.forEach { id ->
            val chars = manager.getCameraCharacteristics(id)
            val facing = when (chars.get(CameraCharacteristics.LENS_FACING)) {
                CameraMetadata.LENS_FACING_BACK -> "BACK"
                CameraMetadata.LENS_FACING_FRONT -> "FRONT"
                else -> "EXTERNAL"
            }
            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList()
            val caps = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList() ?: emptyList()
            val isLogical = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)
            val physIds = if (isLogical) chars.physicalCameraIds.toString() else "-"
            sb.appendLine("[$id] $facing  focal=$focalLengths  logical=$isLogical  physicals=$physIds")
        }

        sb.appendLine("\nConcurrent sets: ${manager.concurrentCameraIds}")

        val setup = findDualRearSetup(context)
        if (setup != null) {
            sb.appendLine("\nSelected setup:")
            sb.appendLine("  Logical camera: ${setup.logicalCameraId}")
            sb.appendLine("  Primary  [${setup.primaryFocalMm}mm]: ${setup.primaryPhysicalId}")
            sb.appendLine("  Secondary [${setup.secondaryFocalMm}mm]: ${setup.secondaryPhysicalId}")
        } else {
            sb.appendLine("\nNo suitable dual rear camera setup found!")
        }

        return sb.toString()
    }
}
