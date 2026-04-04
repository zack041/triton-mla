import torch
import triton
import triton.language as tl


def is_cuda():
    return torch.cuda.is_available()


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def decode_split_kernel(
    q,
    k_cache,
    block_table,
    seq_lens,
    o_accum,
    m_accum,
    l_accum,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap,
    q_stride_b,
    q_stride_h,
    block_table_stride_b,
    block_table_stride_n,
    num_q_heads: tl.constexpr,
    max_kv_splits: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    req_seq_len = tl.load(seq_lens + req_idx)
    offs_h = head_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_nope = tl.arange(0, D_NOPE)
    offs_d_rope = tl.arange(0, D_ROPE)
    head_mask = offs_h < num_q_heads

    q_nope_ptr = q + req_idx * q_stride_b + offs_h[:, None] * q_stride_h + offs_d_nope[None, :]
    q_rope_ptr = (
        q
        + req_idx * q_stride_b
        + offs_h[:, None] * q_stride_h
        + (D_NOPE + offs_d_rope)[None, :]
    )
    q_nope = tl.load(q_nope_ptr, mask=head_mask[:, None], other=0.0)
    q_rope = tl.load(q_rope_ptr, mask=head_mask[:, None], other=0.0)

    kscale = tl.load(k_scale).to(q_nope.dtype)
    vscale = tl.load(v_scale).to(q_nope.dtype)

    combined_scale = (sm_scale * kscale).to(q_nope.dtype)
    q_nope = q_nope * combined_scale
    q_rope = q_rope * combined_scale

    qk_dim = D_NOPE + D_ROPE
    page_stride = page_size * qk_dim
    req_block_table = block_table + req_idx * block_table_stride_b

    num_kv_tiles = tl.cdiv(req_seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_kv_tiles, max_kv_splits)
    if split_idx * tiles_per_split >= num_kv_tiles:
        return
    tile_start = split_idx * tiles_per_split
    tile_end = tl.minimum(tile_start + tiles_per_split, num_kv_tiles)

    M = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    L = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D_NOPE), dtype=tl.float32)

    for tile in tl.range(tile_start, tile_end):
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        causal_mask = offs_n < req_seq_len
        logical_page = offs_n // page_size
        offs_page = offs_n % page_size
        physical_page = tl.load(
            req_block_table + logical_page * block_table_stride_n,
            mask=causal_mask,
            other=0,
        ).to(tl.int64)

        kv_base = physical_page * page_stride + offs_page * qk_dim
        kv_offset = kv_base[:, None]

        kv_nope = tl.load(
            k_cache + kv_offset + offs_d_nope[None, :],
            mask=causal_mask[:, None],
            other=0.0,
        ).to(q_nope.dtype)

        k_nope = tl.trans(kv_nope)

        k_rope = tl.trans(tl.load(
            k_cache + kv_offset + (D_NOPE + offs_d_rope)[None, :],
            mask=causal_mask[:, None],
            other=0.0,
        )).to(q_rope.dtype)

        S = tl.dot(q_nope, k_nope) + tl.dot(q_rope, k_rope)
        if logit_cap > 0:
            S = logit_cap * tanh(S / logit_cap)
        S = tl.where(head_mask[:, None] & causal_mask[None, :], S, -float("inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        v = kv_nope
        acc += tl.dot(P.to(v.dtype), v)

        L = L * alpha + l_j
        M = m_j

    head_blocks = tl.cdiv(num_q_heads, BLOCK_M)
    accum_idx = (req_idx * head_blocks + head_block_idx) * max_kv_splits + split_idx
    o_accum_ptr = o_accum + accum_idx * BLOCK_M * D_NOPE + offs_h[:, None] % BLOCK_M * D_NOPE + offs_d_nope[None, :]
    m_accum_ptr = m_accum + accum_idx * BLOCK_M + offs_h % BLOCK_M
    l_accum_ptr = l_accum + accum_idx * BLOCK_M + offs_h % BLOCK_M

    tl.store(o_accum_ptr, acc * vscale, mask=head_mask[:, None])
    tl.store(m_accum_ptr, M, mask=head_mask)
    tl.store(l_accum_ptr, L, mask=head_mask)


@triton.jit
def reduce_split_kernel(
    o_accum,
    m_accum,
    l_accum,
    seq_lens,
    o,
    lse,
    o_stride_b,
    o_stride_h,
    lse_stride_b,
    lse_stride_h,
    num_q_heads: tl.constexpr,
    max_kv_splits: tl.constexpr,
    D_NOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_d = tl.arange(0, D_NOPE)
    offs_s = tl.arange(0, max_kv_splits)

    req_seq_len = tl.load(seq_lens + req_idx)
    num_kv_tiles = tl.cdiv(req_seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(num_kv_tiles, max_kv_splits)
    split_mask = offs_s * tiles_per_split < num_kv_tiles
    head_blocks = tl.cdiv(num_q_heads, BLOCK_M)
    head_block_idx = head_idx // BLOCK_M
    head_in_block = head_idx % BLOCK_M
    base_idx = (req_idx * head_blocks + head_block_idx) * max_kv_splits
    slot_idx = base_idx + offs_s
    scalars_offset = slot_idx * BLOCK_M + head_in_block
    m_parts = tl.load(m_accum + scalars_offset, mask=split_mask, other=-float("inf"))
    l_parts = tl.load(l_accum + scalars_offset, mask=split_mask, other=0.0)
    overall_m = tl.max(m_parts, axis=0)
    weights = tl.where(split_mask, tl.exp(m_parts - overall_m), 0.0)
    denom = tl.sum(l_parts * weights, axis=0)

    o_offset = (
        slot_idx[:, None] * (BLOCK_M * D_NOPE)
        + head_in_block * D_NOPE
        + offs_d[None, :]
    )
    o_parts = tl.load(o_accum + o_offset, mask=split_mask[:, None], other=0.0)
    weighted_o = o_parts * weights[:, None]
    o_sum = tl.sum(weighted_o, axis=0)
    o_out = o_sum / denom
    lse_out = overall_m + tl.log(denom)
    o_ptr = o + req_idx * o_stride_b + head_idx * o_stride_h + offs_d
    lse_ptr = lse + req_idx * lse_stride_b + head_idx * lse_stride_h
    tl.store(o_ptr, o_out)
    tl.store(lse_ptr, lse_out)

def fwd_mla(
    q,  # [batch, num_q_heads, D_NOPE + D_ROPE]
    k_cache,  # [num_blocks, page_size, D_NOPE + D_ROPE] or flattened
    o,  # [batch, num_q_heads, D_NOPE]
    lse,  # [batch, num_q_heads]
    block_table,  # [batch, max_blocks_per_seq]
    seq_lens,  # [batch]
    page_size,
    o_accum=None,  # [batch * head_block * splits, BLOCK_M, D_NOPE], float32
    lse_accum=None,  # [batch * head_block * splits, BLOCK_M], float32; stores M
    l_accum=None,  # [batch * head_block * splits, BLOCK_M], float32; stores L
    sm_scale=1.0,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    max_kv_splits=16,
    D_NOPE=512,
    D_ROPE=64,
):

    if k_scale is None:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    if v_scale is None:
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    BLOCK_M = 16
    BLOCK_N = 128
    batch_size = q.shape[0]
    max_seq_len = int(seq_lens.max().item())
    q_heads = q.shape[1]
    head_blocks = (q_heads + BLOCK_M - 1) // BLOCK_M

    grid1 = (batch_size, head_blocks, max_kv_splits)
    grid2 = (batch_size, q_heads)

    decode_split_kernel[grid1](
        q,
        k_cache,
        block_table,
        seq_lens,
        o_accum,
        lse_accum,
        l_accum,
        sm_scale,
        k_scale,
        v_scale,
        logit_cap,
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        q_heads,
        max_kv_splits,
        page_size,
        BLOCK_M,
        BLOCK_N,
        D_NOPE,
        D_ROPE,
        num_warps = 4,
        num_stages = 2,
    )
    reduce_split_kernel[grid2](
        o_accum,
        lse_accum,
        l_accum,
        seq_lens,
        o,
        lse,
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        lse.stride(1),
        q_heads,
        max_kv_splits,
        D_NOPE,
        BLOCK_M,
        BLOCK_N,
    )

    return o, lse
