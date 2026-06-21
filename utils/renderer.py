import os

import numpy as np
from pxr import Usd, UsdGeom


class USDMeshRenderer:
    def __init__(
        self,
        filepath,
        totalframes,
        fps,
        use_binary=True,
        enable_streaming=False,
        stream_chunk_size=100,
        root_prim_path="/root",
    ) -> None:
        if use_binary and not filepath.endswith(".usdc"):
            filepath = filepath.replace(".usda", ".usdc")

        self.stage = Usd.Stage.CreateNew(filepath)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        self.stage.SetStartTimeCode(1)
        self.stage.SetEndTimeCode(totalframes)
        self.stage.SetTimeCodesPerSecond(fps)

        self.totalframes = totalframes
        self.fps = fps
        self.frame = 0
        self.time = 0.0
        self.mesh_prims = []
        self.mesh_verts = []
        self.enable_streaming = enable_streaming
        self.stream_chunk_size = stream_chunk_size
        self.filepath = filepath

        self.root_prim_path = (
            root_prim_path if root_prim_path.startswith("/") else f"/{root_prim_path}"
        )
        self.root_xform = UsdGeom.Xform.Define(self.stage, self.root_prim_path)
        self.stage.SetDefaultPrim(self.root_xform.GetPrim())

    def render(self):
        if self.frame >= self.totalframes:
            return

        for prim_path, vertices in zip(self.mesh_prims, self.mesh_verts):
            mesh_geom = UsdGeom.Mesh(self.stage.GetPrimAtPath(prim_path))
            mesh_geom.GetPointsAttr().Set(value=vertices.to_numpy(), time=self.frame + 1)

        if self.enable_streaming and (self.frame + 1) % self.stream_chunk_size == 0:
            print(f"Streaming save at frame {self.frame + 1}/{self.totalframes}")
            self.stage.Save()

        self.frame += 1
        self.time = self.frame / self.fps

    def add_static_sphere(self, center, radius, name="sdf_sphere"):
        prim_path = f"{self.root_prim_path}/{name}"
        sphere_geom = UsdGeom.Sphere.Define(self.stage, prim_path)
        sphere_geom.GetRadiusAttr().Set(float(radius))
        sphere_geom.AddTranslateOp().Set(tuple(np.asarray(center, dtype=np.float32)))

    def add_static_cube(self, center, size, name="sdf_cube"):
        prim_path = f"{self.root_prim_path}/{name}"
        cube_geom = UsdGeom.Cube.Define(self.stage, prim_path)
        xform = UsdGeom.Xformable(cube_geom.GetPrim())
        xform.AddScaleOp().Set(tuple(np.asarray(size, dtype=np.float32)))
        xform.AddTranslateOp().Set(tuple(np.asarray(center, dtype=np.float32)))

    def add_static_plane(self, point, normal, extent=2.0, name="sdf_plane"):
        prim_path = f"{self.root_prim_path}/{name}"
        plane_geom = UsdGeom.Mesh.Define(self.stage, prim_path)

        n = np.asarray(normal, dtype=np.float32)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            n = n / n_norm

        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(ref, n)) > 0.9:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u = np.cross(n, ref)
        u = u / max(np.linalg.norm(u), 1e-8)
        v = np.cross(n, u)

        p = np.asarray(point, dtype=np.float32)
        h = float(extent) * 0.5
        points = np.array(
            [
                p - h * u - h * v,
                p - h * u + h * v,
                p + h * u + h * v,
                p + h * u - h * v,
            ],
            dtype=np.float32,
        )

        plane_geom.GetPointsAttr().Set(points)
        plane_geom.GetFaceVertexIndicesAttr().Set(
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        )
        plane_geom.GetFaceVertexCountsAttr().Set(np.array([3, 3], dtype=np.int32))
        plane_geom.GetSubdivisionSchemeAttr().Set("none")

    def add_dynamic_mesh(
        self,
        vert,
        face,
        meshname="mesh",
    ):
        prim_path = f"{self.root_prim_path}/{meshname}"
        self.mesh_prims.append(prim_path)
        self.mesh_verts.append(vert)

        mesh_geom = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh_geom.GetPointsAttr().Set(vert.to_numpy())
        mesh_geom.GetFaceVertexIndicesAttr().Set(face.to_numpy())
        mesh_geom.GetFaceVertexCountsAttr().Set(face.shape[0] // 3 * [3])
        mesh_geom.GetSubdivisionSchemeAttr().Set("none")

    def save(self):
        self.stage.Save()

    def get_root_prim_path(self):
        return self.root_prim_path

    def get_file_size_mb(self):
        if os.path.exists(self.filepath):
            return os.path.getsize(self.filepath) / (1024 * 1024)
        return 0.0


class OptimizedUSDMeshRenderer(USDMeshRenderer):
    def __init__(
        self,
        filepath,
        totalframes,
        fps,
        max_memory_frames=500,
        compression_level="high",
        root_prim_path="/root",
    ):
        if not filepath.endswith(".usdc"):
            filepath = filepath.replace(".usda", ".usdc")

        super().__init__(
            filepath,
            totalframes,
            fps,
            use_binary=True,
            enable_streaming=True,
            stream_chunk_size=50,
            root_prim_path=root_prim_path,
        )

        self.max_memory_frames = max_memory_frames
        self.compression_level = compression_level

    def render(self):
        if self.frame >= self.totalframes:
            return

        for prim_path, vertices in zip(self.mesh_prims, self.mesh_verts):
            mesh_geom = UsdGeom.Mesh(self.stage.GetPrimAtPath(prim_path))
            mesh_geom.GetPointsAttr().Set(value=vertices.to_numpy(), time=self.frame + 1)

        if self.frame % 100 == 0:
            file_size_mb = self.get_file_size_mb()
            if file_size_mb > 1000:
                self.stream_chunk_size = 25
            elif file_size_mb > 500:
                self.stream_chunk_size = 50

        if (self.frame + 1) % self.stream_chunk_size == 0:
            self.stage.Save()
            print(
                "Memory management save at frame "
                f"{self.frame + 1}, file size: {self.get_file_size_mb():.1f}MB"
            )

        self.frame += 1
        self.time = self.frame / self.fps

    def save(self):
        print("Performing final optimized save...")
        self.stage.Save()
        final_size = self.get_file_size_mb()
        print(f"Final USD file size: {final_size:.1f}MB")

        if final_size > 2000:
            print("WARNING: File is very large (>2GB). Consider:")
            print("- Reducing frame count or mesh resolution")
            print("- Using a sequence of smaller USD files")
        elif final_size > 1000:
            print("INFO: Large file (>1GB). USD viewers may need extra time to load it.")
