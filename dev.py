import torch
from torch import nn
from typing import Optional
from models.octformer import RPE, OctreeT, OctreeAttention
from ocnn.octree import Octree, Points


import flash_attn


def get_qkv_seglen(octree: Octree, depth: int, patch_size: int, nempty: bool):
    batch_nnum = (
        octree.batch_nnum[depth] if not nempty else octree.batch_nnum_nempty[depth]
    ).type(torch.int32).to(octree.device)
    assert (batch_nnum != 0).all(), "batch_nnum is 0!"
    remainder = (batch_nnum - 1) % patch_size + 1
    num_patch_per_batch = (batch_nnum + patch_size - 1) // patch_size
    seglen = torch.full((num_patch_per_batch.sum().item(),), patch_size, dtype=torch.int32, device=octree.device)
    seglen[num_patch_per_batch.cumsum(0) - 1] = remainder
    return torch.cat([torch.tensor([0], device=octree.device, dtype=torch.int32), seglen.cumsum(0)], dim=0).int(), seglen.max().item()


def get_qkv_seglen_v1(octree: Octree, depth: int, patch_size: int, nempty: bool):
    batch_nnum = (
        octree.batch_nnum_nempty[depth] if nempty else octree.batch_nnum[depth]
    ).type(torch.int32).to(octree.device)
    
    total_nodes = batch_nnum.sum().item()
    
    # 1. 计算 Batch 边界 (Cumulative sum of nodes)
    batch_boundaries = batch_nnum.cumsum(0)
    
    # 2. 计算 Patch 边界 (K, 2K, 3K...)
    num_total_patches = (total_nodes + patch_size - 1) // patch_size
    patch_boundaries = torch.arange(1, num_total_patches + 1, device=octree.device) * patch_size
    
    # 3. 合并所有边界并取交集
    # 我们需要的是：只要到了 Patch 结尾，或者到了 Batch 结尾，都要切断 Attention
    combined = torch.cat([batch_boundaries, patch_boundaries])
    combined = combined[combined <= total_nodes] # 过滤掉超过总数的
    all_split_points = torch.unique(combined).sort()[0]
    
    # 4. 构造 cu_seqlens (必须以 0 开头)
    cu_seqlens = torch.zeros(len(all_split_points) + 1, dtype=torch.int32, device=octree.device)
    cu_seqlens[1:] = all_split_points
    
    # 5. 计算 seglen (用于计算 max_seq_len)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq_len = seqlens.max().item() if len(seqlens) > 0 else 0
    
    return cu_seqlens, max_seq_len


class OctreeAttentionSDPA(OctreeAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.use_rpe, "RPE is not supported!"
    
    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation

        # patch partition
        data = octree.patch_partition(data, depth)
        if D > 1:  # dilation
            mask = octree.dilate_mask[depth]
            data = data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        else:
            mask = octree.patch_mask[depth]
        data = data.view(-1, K, C)
        mask = mask.type(data.dtype)

        # qkv
        qkv = self.qkv(data).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (N, H, K, C')

        data = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), scale=self.scale, dropout_p=self.attn_drop.p, is_causal=False)
        data = data.transpose(1, 2).reshape(-1, C)

        # patch reverse
        if D > 1:  # dilation
            data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        data = octree.patch_reverse(data, depth)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data



class OctreeAttentionFlash(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dilation: int = 1,
        use_rpe: bool = True,
        chunking_mode: str = 'original'
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.chunk_fn = get_qkv_seglen_v1 if chunking_mode == 'original' else get_qkv_seglen
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop) if proj_drop > 0 else torch.nn.Identity()
        assert not use_rpe, "RPE is not supported!"
        assert dilation == 1, "dilation > 1 is not supported!"

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        H = self.num_heads
        C = self.head_dim

        # patch partition
        data = data.view(-1, self.dim)

        # qkv
        qkv = self.qkv(data).reshape(-1, 3, H, C)
        seglen, max_seq_len = self.chunk_fn(octree, depth, self.patch_size, octree.nempty)

        data = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, seglen, max_seq_len, self.attn_drop, self.scale)
        data = data.view(-1, self.dim)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data
