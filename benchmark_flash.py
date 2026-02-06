import torch
import triton
from typing import List
import trimesh

from models.octformer import OctreeAttention, OctreeT
from dev import OctreeAttentionFlash
from ocnn.octree import Octree, Points, merge_octrees


def normalize_mesh(mesh):
    """将 Mesh 归一化到 [-1, 1] 空间"""
    vertices = mesh.vertices
    v_max = vertices.max(axis=0)
    v_min = vertices.min(axis=0)
    center = (v_max + v_min) / 2
    scale = (v_max - v_min).max() / 2
    mesh.vertices = (vertices - center) / scale
    return mesh


def build_octree():
    device = torch.device("cuda")

    # 1. 构建 Meshes
    meshes = [
        trimesh.creation.icosphere(subdivisions=2, radius=r) for r in [0.8, 0.6, 0.4]
    ]

    try:
        # 下载并归一化斯坦福兔子
        bunny_url = "https://github.com/mikedh/trimesh/raw/main/models/bunny.ply"
        bunny = trimesh.load_remote(bunny_url).to_mesh()
        meshes.append(normalize_mesh(bunny))
    except Exception as e:
        print(f"无法下载兔子模型，仅使用球体进行测试: {e}")

    # 2. 采样点云并构建 Octree
    pointclouds = [
        Points(torch.from_numpy(trimesh.sample.sample_surface(mesh, 10000)[0]))
        for mesh in meshes
    ]

    def build_octree(points):
        octree = Octree(depth=10, full_depth=2)
        octree.build_octree(points)
        return octree

    octrees = [build_octree(pc) for pc in pointclouds]
    octree = merge_octrees(octrees)
    octree = octree.to(device)

    return octree


configs = [
    triton.testing.Benchmark(
        x_names=["depth"],
        x_vals=list(range(3, 11)),
        line_arg="provider",  # 线条名称
        line_vals=["original", "flash"],
        line_names=["Original", "Flash"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Execution Time (ms)",
        plot_name=f"head_dim={head_dim}, num_heads={num_heads}, patch_size={patch_size}",
        args={"head_dim": head_dim, "num_heads": num_heads, "patch_size": patch_size},
    )
    for patch_size in [32, 64, 128, 256, 512, 1024]
    for head_dim in [256]
    for num_heads in [4]
]


# 配置 Benchmark 参数
@triton.testing.perf_report(configs)
def benchmark(depth, head_dim, num_heads, patch_size, provider):
    device = torch.device("cuda")
    dim = head_dim * num_heads
    octree = build_octree()
    octree = OctreeT(octree, patch_size, dilation=1, nempty=True)
    data = torch.randn(octree.nnum_nempty[depth], dim, device=device).type(
        torch.float16
    )

    if provider == "original":
        model = (
            OctreeAttention(
                dim=dim, patch_size=patch_size, num_heads=num_heads, use_rpe=False
            )
            .to(device)
            .type(torch.float16)
        )
    else:
        model = (
            OctreeAttentionFlash(
                dim=dim, patch_size=patch_size, num_heads=num_heads, use_rpe=False, chunking_mode='new'
            )
            .to(device)
            .type(torch.float16)
        )

    def fn():
        out = model(data, octree, depth)
        loss = out.sum()
        loss.backward()
        return

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


if __name__ == "__main__":
    benchmark.run(save_path="./benchmark_results", show_plots=False, print_data=True)
