// Swift wrapper for the PicoVDB STL importer.
//
// Integration (e.g. ../picopainterapp):
//   1. `zig build xcframework` -> zig-out/PicoVDBSTL.xcframework
//   2. Add the xcframework to the Xcode target (General > Frameworks; the
//      bundled module.modulemap makes `import PicoVDBSTL` work).
//   3. Copy this file into the app.
//
// Voxelization is CPU-bound (seconds for large meshes) — call off the main
// actor.

import Foundation
import PicoVDBSTL

public struct STLImportError: Error, CustomStringConvertible {
    public let code: Int32
    public var description: String { String(cString: pv_error_string(code)) }
}

public struct STLInfo {
    public let triangleCount: UInt32
    public let bboxMin: SIMD3<Float>
    public let bboxMax: SIMD3<Float>
}

public struct STLImportStats {
    public let activeVoxels: UInt64
    public let surfaceVoxels: UInt64
    public let leafCount: UInt32
    public let lowerCount: UInt32
    public let upperCount: UInt32
    /// Post-rotation mesh bounds in world units.
    public let worldMin: SIMD3<Float>
    public let worldMax: SIMD3<Float>
}

public enum PicoVDB {
    /// Grid value storage; raw values match pv_value_type in picovdb.h.
    public enum ValueType: UInt32 {
        case f32 = 0
        case u8 = 1
    }

    /// Triangle count and world bounds, without voxelizing.
    public static func stlInfo(_ stl: Data) throws -> STLInfo {
        var info = pv_stl_info()
        let rc = stl.withUnsafeBytes { raw in
            pv_stl_get_info(raw.bindMemory(to: UInt8.self).baseAddress, stl.count, &info)
        }
        guard rc == 0 else { throw STLImportError(code: rc) }
        return STLInfo(
            triangleCount: info.triangle_count,
            bboxMin: SIMD3(info.bbox_min.0, info.bbox_min.1, info.bbox_min.2),
            bboxMax: SIMD3(info.bbox_max.0, info.bbox_max.1, info.bbox_max.2)
        )
    }

    /// Convert an STL (binary or ASCII) to an encoded .pvdb.
    /// - Parameters:
    ///   - voxelsPerUnit: grid resolution in voxels per world unit.
    ///   - maxVoxels: fail instead of voxelizing when the voxel estimate (mesh
    ///     bbox dilated by the narrow band) exceeds this; peak memory is ~8
    ///     bytes per estimated voxel. 0 = unlimited.
    ///   - rotateDeg: applied about x, then y, then z (e.g. (-90, 0, 0)
    ///     re-orients a Z-up mesh to Y-up).
    public static func importSTL(
        _ stl: Data,
        voxelsPerUnit: Float,
        maxVoxels: UInt64 = 0,
        halfWidth: Float = 0, // 0 selects the default (3.0)
        valueType: ValueType = .f32,
        rotateDeg: SIMD3<Float> = .zero
    ) throws -> (pvdb: Data, stats: STLImportStats) {
        var options = pv_mesh_options(
            max_voxels: maxVoxels,
            voxels_per_unit: voxelsPerUnit,
            half_width: halfWidth,
            value_type: valueType.rawValue,
            rotate_deg: (rotateDeg.x, rotateDeg.y, rotateDeg.z)
        )
        var buffer = pv_buffer()
        let rc = stl.withUnsafeBytes { raw in
            pv_stl_to_pvdb(raw.bindMemory(to: UInt8.self).baseAddress, stl.count, &options, &buffer)
        }
        guard rc == 0, let data = buffer.data else { throw STLImportError(code: rc) }
        defer { pv_buffer_free(&buffer) }
        let stats = STLImportStats(
            activeVoxels: buffer.stats.active_voxels,
            surfaceVoxels: buffer.stats.surface_voxels,
            leafCount: buffer.stats.leaf_count,
            lowerCount: buffer.stats.lower_count,
            upperCount: buffer.stats.upper_count,
            worldMin: SIMD3(buffer.stats.world_min.0, buffer.stats.world_min.1, buffer.stats.world_min.2),
            worldMax: SIMD3(buffer.stats.world_max.0, buffer.stats.world_max.1, buffer.stats.world_max.2)
        )
        return (Data(bytes: data, count: buffer.len), stats)
    }
}
