import math
from dataclasses import dataclass
import importlib
import torch
import triton
import mla
import triton_decode_attention
importlib.reload(mla)
importlib.reload(triton_decode_attention)

from mla import fwd_mla
from triton_decode_attention import decode_attention_fwd


DTYPE = torch.float16
DEVICE = "cuda"
NUM_Q_HEADS = 128
D_NOPE = 512
D_ROPE = 64
HEAD_DIM = D_NOPE + D_ROPE
PAGE_SIZE = 16

MAX_KV_SPLITS = 16  # fixed for cuda graph assumption, value based on vllm's unified kernel
LOGIT_CAP = 0.0
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)
BLOCK_M = 16


@dataclass
class BenchCase:
    batch_size: int
    seq_len: int
    name: str

def setup_inputs(case: BenchCase):
    bs = case.batch_size
    seq_len = case.seq_len

    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device=DEVICE)
    q = torch.randn(bs, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1

    max_pages_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = bs * max_pages_per_seq + 16

    kv_data = torch.randn(total_pages, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.1
    block_table = torch.zeros(bs, max_pages_per_seq, dtype=torch.int32, device=DEVICE)
    page_counter = 0
    for b in range(bs):
        n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(n_pages):
            block_table[b, p] = page_counter
            page_counter += 1

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    return q, kv_data, block_table, seq_lens, k_scale, v_scale


def alloc_split_buffers(batch_size: int):
    head_blocks = (NUM_Q_HEADS + BLOCK_M - 1) // BLOCK_M
    total_slots = batch_size * head_blocks * MAX_KV_SPLITS
    o_accum = torch.zeros(total_slots, BLOCK_M, D_NOPE, dtype=torch.float32, device=DEVICE)
    m_accum = torch.zeros(total_slots, BLOCK_M, dtype=torch.float32, device=DEVICE)
    l_accum = torch.zeros(total_slots, BLOCK_M, dtype=torch.float32, device=DEVICE)
    return o_accum, m_accum, l_accum


def bench_one(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return min(times), sum(times) / len(times), max(times)


def estimate_decode_flops(bs: int, seq_len: int) -> float:
    # Per (head, key-token):
    # qk: 2 * (D_NOPE + D_ROPE), pv: 2 * D_NOPE
    # total: 2 * (2 * D_NOPE + D_ROPE)
    return (
        bs
        * NUM_Q_HEADS
        * seq_len
        * (2.0 * (2.0 * D_NOPE + D_ROPE))
    )


def reference_mla_decode(q, k_cache, block_table, seq_lens, k_scale_val=1.0, v_scale_val=1.0):
    batch, num_heads, head_dim = q.shape
    assert head_dim == HEAD_DIM

    o = torch.zeros(batch, num_heads, D_NOPE, dtype=torch.float32, device=q.device)
    lse = torch.zeros(batch, num_heads, dtype=torch.float32, device=q.device)

    for b in range(batch):
        seq_len = seq_lens[b].item()
        if seq_len == 0:
            lse[b] = -float("inf")
            continue

        token_indices = torch.arange(seq_len, device=q.device)
        logical_pages = token_indices // PAGE_SIZE
        in_page_offsets = token_indices % PAGE_SIZE
        physical_pages = block_table[b, logical_pages].long()

        kv_data = k_cache[physical_pages, in_page_offsets]
        k_nope = kv_data[:, :D_NOPE] * k_scale_val
        k_rope = kv_data[:, D_NOPE:] * k_scale_val
        v = kv_data[:, :D_NOPE] * v_scale_val

        q_nope = q[b, :, :D_NOPE]
        q_rope = q[b, :, D_NOPE:]

        scores = (q_nope @ k_nope.T + q_rope @ k_rope.T) * SM_SCALE
        if LOGIT_CAP > 0:
            scores = LOGIT_CAP * torch.tanh(scores / LOGIT_CAP)

        lse[b] = torch.logsumexp(scores, dim=-1)
        o[b] = torch.softmax(scores, dim=-1) @ v

    return o, lse


def run_ours(q, kv_data, block_table, seq_lens, k_scale, v_scale):
    bs = q.shape[0]
    o = torch.zeros(bs, NUM_Q_HEADS, D_NOPE, dtype=DTYPE, device=DEVICE)
    lse = torch.zeros(bs, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    o_accum, m_accum, l_accum = alloc_split_buffers(bs)

    def fn():
        fwd_mla(
            q=q,
            k_cache=kv_data,
            o=o,
            lse=lse,
            block_table=block_table,
            seq_lens=seq_lens,
            page_size=PAGE_SIZE,
            o_accum=o_accum,
            lse_accum=m_accum,
            l_accum=l_accum,
            sm_scale=SM_SCALE,
            logit_cap=LOGIT_CAP,
            k_scale=k_scale,
            v_scale=v_scale,
            max_kv_splits=MAX_KV_SPLITS,
            D_NOPE=D_NOPE,
            D_ROPE=D_ROPE,
        )

    return fn, o, lse


def run_vllm_kernel(q, kv_data, block_table, seq_lens, k_scale, v_scale):
    bs = q.shape[0]

    k_cache = kv_data.unsqueeze(2).contiguous()  # [pages, page_size, 1, HEAD_DIM]
    v_cache = kv_data[:, :, :D_NOPE].unsqueeze(2).contiguous()  # [pages, page_size, 1, D_NOPE]

    o = torch.zeros(bs, NUM_Q_HEADS, D_NOPE, dtype=DTYPE, device=DEVICE)
    lse = torch.zeros(bs, NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)

    attn_logits = torch.zeros(
        bs, NUM_Q_HEADS, MAX_KV_SPLITS, D_NOPE + 1,
        dtype=torch.float32,
        device=DEVICE,
    )

    def fn():
        decode_attention_fwd(
            q,
            k_cache,
            v_cache,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            MAX_KV_SPLITS,
            SM_SCALE,
            PAGE_SIZE,
            LOGIT_CAP,
            k_scale,
            v_scale,
        )

    return fn, o, lse


def test_correctness():
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    cases = [
        BenchCase(1, 512, "nosplit: bs=1, seq=512"),
        BenchCase(1, 2048, "nosplit: bs=1, seq=2048"),
        BenchCase(64, 4096, "nosplit: bs=64, seq=4096"),
        BenchCase(1, 4096, "split: bs=1, seq=4096"),
        BenchCase(8, 8192, "split: bs=8, seq=8192"),
        BenchCase(4, 16384, "split: bs=4, seq=16384"),
    ]

    all_passed = True
    for case in cases:
        q, kv_data, block_table, seq_lens, k_scale, v_scale = setup_inputs(case)

        run_fn, o, lse = run_ours(q, kv_data, block_table, seq_lens, k_scale, v_scale)
        run_fn()
        torch.cuda.synchronize()

        ref_o, ref_lse = reference_mla_decode(q, kv_data, block_table, seq_lens)

        o_f32 = o.float()
        o_atol = (o_f32 - ref_o).abs().max().item()
        o_rtol = ((o_f32 - ref_o).abs() / (ref_o.abs() + 1e-8)).max().item()
        lse_atol = (lse - ref_lse).abs().max().item()

        o_pass = o_atol < 0.05
        lse_pass = lse_atol < 0.1
        passed = o_pass and lse_pass
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {case.name}")
        print(f"  output max_atol={o_atol:.6f}, max_rtol={o_rtol:.6f} {'OK' if o_pass else 'FAIL'}")
        print(f"  lse    max_atol={lse_atol:.6f} {'OK' if lse_pass else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
    return all_passed


def test_performance():
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: fwd_mla vs triton_decode_attention")
    print("=" * 60)

    cases = [
        BenchCase(1, 2048, "bs=1, seq=2K"),
        BenchCase(1, 4096, "bs=1, seq=4K"),
        BenchCase(1, 8192, "bs=1, seq=8K"),
        BenchCase(1, 16384, "bs=1, seq=16K"),
        BenchCase(1, 32768, "bs=1, seq=32K"),
        BenchCase(1, 65536, "bs=1, seq=64K"),
        BenchCase(8, 2048, "bs=8, seq=2K"),
        BenchCase(8, 4096, "bs=8, seq=4K"),
        BenchCase(8, 8192, "bs=8, seq=8K"),
        BenchCase(8, 16384, "bs=8, seq=16K"),
        BenchCase(8, 32768, "bs=8, seq=32K"),
        BenchCase(8, 65536, "bs=8, seq=64K"),
        BenchCase(32, 2048, "bs=32, seq=2K"),
        BenchCase(32, 4096, "bs=32, seq=4K"),
        BenchCase(32, 8192, "bs=32, seq=8K"),
        BenchCase(32, 16384, "bs=32, seq=16K"),
        BenchCase(32, 32768, "bs=32, seq=32K"),
        BenchCase(32, 65536, "bs=32, seq=64K"),
    ]

    header = (
        f"{'Config':<22} {'fwd_mla (ms)':>10} {'vLLM (ms)':>10} "
        f"{'Speedup':>8} {'fwd_mla TFLOPS':>14} {'vLLM TFLOPS':>14}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for i, case in enumerate(cases):
        q, kv_data, block_table, seq_lens, k_scale, v_scale = setup_inputs(case)

        ours_fn, o_ours, _ = run_ours(q, kv_data, block_table, seq_lens, k_scale, v_scale)
        _, avg_ours, _ = bench_one(ours_fn)

        vllm_fn, o_vllm, _ = run_vllm_kernel(q, kv_data, block_table, seq_lens, k_scale, v_scale)
        _, avg_vllm, _ = bench_one(vllm_fn)

        flops = estimate_decode_flops(case.batch_size, case.seq_len)
        tflops_ours = flops / (avg_ours * 1e-3) / 1e12
        tflops_vllm = flops / (avg_vllm * 1e-3) / 1e12
        speedup = avg_vllm / avg_ours

        print(
            f"{case.name:<22} {avg_ours:>10.3f} {avg_vllm:>10.3f} "
            f"{speedup:>7.2f}x {tflops_ours:>13.2f} {tflops_vllm:>13.2f}"
        )


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"triton: {triton.__version__}")

    test_correctness()
    test_performance()
