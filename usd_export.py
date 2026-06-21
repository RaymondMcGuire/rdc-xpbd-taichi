import os
from pathlib import Path

import numpy as np

from xpbd_gpu.sdf import CubeSdf, PlaneSdf, SphereSdf
from utils import renderer


def method_tag(block_neohookean, dynamic_cheby):
    model_tag = "block_neohookean" if block_neohookean else "split_neohookean"
    accel_tag = "dynamic_cheby" if dynamic_cheby else "no_cheby"
    return f"{model_tag}_{accel_tag}"


def young_tag(young):
    return f"{young:.3e}".replace("+", "").replace(".", "p")


def build_output_paths(model_path, block_neohookean, dynamic_cheby, young, use_binary):
    model_name = Path(model_path).name
    model_prefix = model_name.split(".")[0] if "." in model_name else Path(model_name).stem
    if not model_prefix:
        model_prefix = "model"

    output_dir = os.path.join(os.getcwd(), "output", model_prefix)
    os.makedirs(output_dir, exist_ok=True)

    usd_ext = "usdc" if use_binary else "usda"
    run_name = f"xpbd_{method_tag(block_neohookean, dynamic_cheby)}_young_{young_tag(young)}"
    usd_output_path = os.path.join(output_dir, f"{run_name}.{usd_ext}")
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    return output_dir, run_output_dir, usd_output_path


def create_renderer(args, usd_output_path, total_frames, fps):
    if args.no_usd:
        print("USD export disabled (--no-usd)")
        return None

    if args.optimize_usd:
        print("Using OptimizedUSDMeshRenderer for large file handling")
        return renderer.OptimizedUSDMeshRenderer(
            usd_output_path, total_frames, fps, root_prim_path=args.usd_root_prim
        )

    return renderer.USDMeshRenderer(
        usd_output_path,
        total_frames,
        fps,
        use_binary=args.use_binary,
        enable_streaming=args.streaming,
        root_prim_path=args.usd_root_prim,
    )


def add_solver_mesh(usd_renderer, solver):
    if usd_renderer is not None:
        usd_renderer.add_dynamic_mesh(solver.mesh.verts.x, solver.indices, "deformable_mesh")


def add_sdfs(usd_renderer, sdf_config):
    if usd_renderer is None:
        return

    if isinstance(sdf_config, (list, tuple)):
        for i, sdf_obj in enumerate(sdf_config):
            _add_sdf(usd_renderer, sdf_obj, name_suffix=f"_{i}")
    else:
        _add_sdf(usd_renderer, sdf_config)


def _add_sdf(usd_renderer, sdf_obj, name_suffix=""):
    if isinstance(sdf_obj, SphereSdf):
        center = sdf_obj.center[None].to_numpy()
        radius = float(sdf_obj.radius[None])
        usd_renderer.add_static_sphere(
            center=center, radius=radius, name=f"sdf_sphere{name_suffix}"
        )
    elif isinstance(sdf_obj, CubeSdf):
        center = sdf_obj.center[None].to_numpy()
        size = (sdf_obj.half_size[None].to_numpy() * 2.0).astype(np.float32)
        usd_renderer.add_static_cube(center=center, size=size, name=f"sdf_cube{name_suffix}")
    elif isinstance(sdf_obj, PlaneSdf):
        point = sdf_obj.point[None].to_numpy()
        normal = sdf_obj.dir[None].to_numpy()
        usd_renderer.add_static_plane(
            point=point, normal=normal, extent=4.0, name=f"sdf_plane{name_suffix}"
        )
    else:
        print(f"SDF type {type(sdf_obj).__name__} has no USD export hook yet.")
