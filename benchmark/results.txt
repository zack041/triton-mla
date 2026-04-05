(q head = 16, deepseek v2 lite)
Device: NVIDIA H100 80GB HBM3
triton: 3.6.0
============================================================
CORRECTNESS TEST
============================================================

[PASS] nosplit: bs=1, seq=512
  output max_atol=0.000008, max_rtol=2.231283 OK
  lse    max_atol=0.003629 OK

[PASS] nosplit: bs=1, seq=2048
  output max_atol=0.000004, max_rtol=0.509213 OK
  lse    max_atol=0.000784 OK

[PASS] nosplit: bs=64, seq=4096
  output max_atol=0.000004, max_rtol=59.604645 OK
  lse    max_atol=0.005713 OK

[PASS] split: bs=1, seq=4096
  output max_atol=0.000004, max_rtol=0.479873 OK
  lse    max_atol=0.005466 OK

[PASS] split: bs=8, seq=8192
  output max_atol=0.000004, max_rtol=7.706983 OK
  lse    max_atol=0.004836 OK

[PASS] split: bs=4, seq=16384
  output max_atol=0.000002, max_rtol=41.723251 OK
  lse    max_atol=0.006978 OK

============================================================
OVERALL: ALL PASSED
============================================================

============================================================
PERFORMANCE COMPARISON: fwd_mla vs triton_decode_attention
============================================================

Config                 fwd_mla (ms)  vLLM (ms)  Speedup fwd_mla TFLOPS    vLLM TFLOPS
-------------------------------------------------------------------------------------
bs=1, seq=2K                0.066      0.059    0.90x          1.08          1.20
bs=1, seq=4K                0.065      0.061    0.94x          2.18          2.32
bs=1, seq=8K                0.068      0.064    0.95x          4.22          4.45
bs=1, seq=16K               0.084      0.096    1.14x          6.77          5.94
bs=1, seq=32K               0.127      0.189    1.49x          9.00          6.03
bs=1, seq=64K               0.211      0.360    1.70x         10.80          6.33
bs=8, seq=2K                0.065      0.059    0.92x          8.82          9.61
bs=8, seq=4K                0.067      0.063    0.94x         16.96         18.11
bs=8, seq=8K                0.085      0.074    0.87x         26.79         30.72
bs=8, seq=16K               0.115      0.125    1.09x         39.60         36.39
bs=8, seq=32K               0.174      0.227    1.30x         52.32         40.25
bs=8, seq=64K               0.287      0.430    1.50x         63.58         42.47
bs=32, seq=2K               0.097      0.080    0.83x         23.49         28.40
bs=32, seq=4K               0.130      0.128    0.98x         35.03         35.76
bs=32, seq=8K               0.185      0.219    1.18x         49.31         41.68
bs=32, seq=16K              0.293      0.404    1.38x         62.39         45.21
bs=32, seq=32K              0.506      0.780    1.54x         72.19         46.78
bs=32, seq=64K              0.938      1.517    1.62x         77.87         48.14

(q head = 128, big deepseeks)
Device: NVIDIA H100 80GB HBM3
triton: 3.6.0
============================================================
CORRECTNESS TEST
============================================================

[PASS] nosplit: bs=1, seq=512
  output max_atol=0.000015, max_rtol=9.419645 OK
  lse    max_atol=0.004052 OK

[PASS] nosplit: bs=1, seq=2048
  output max_atol=0.000004, max_rtol=3.690425 OK
  lse    max_atol=0.003686 OK

[PASS] nosplit: bs=64, seq=4096
  output max_atol=0.000004, max_rtol=71.525574 OK
  lse    max_atol=0.005864 OK

[PASS] split: bs=1, seq=4096
  output max_atol=0.000004, max_rtol=53.644180 OK
  lse    max_atol=0.005691 OK

[PASS] split: bs=8, seq=8192
  output max_atol=0.000002, max_rtol=47.683716 OK
  lse    max_atol=0.004903 OK

[PASS] split: bs=4, seq=16384
  output max_atol=0.000002, max_rtol=17.881393 OK
  lse    max_atol=0.007005 OK

============================================================
OVERALL: ALL PASSED
============================================================

============================================================
PERFORMANCE COMPARISON: fwd_mla vs triton_decode_attention
============================================================

Config                 fwd_mla (ms)  vLLM (ms)  Speedup fwd_mla TFLOPS    vLLM TFLOPS
-------------------------------------------------------------------------------------
bs=1, seq=2K                0.065      0.060    0.93x          8.76          9.43
bs=1, seq=4K                0.064      0.062    0.96x         17.72         18.51
bs=1, seq=8K                0.068      0.065    0.96x         33.64         35.19
bs=1, seq=16K               0.085      0.104    1.23x         53.88         43.76
bs=1, seq=32K               0.132      0.197    1.50x         69.40         46.23
bs=1, seq=64K               0.211      0.372    1.77x         86.66         49.06
bs=8, seq=2K                0.128      0.097    0.76x         35.71         47.10
bs=8, seq=4K                0.168      0.147    0.88x         54.27         61.96
bs=8, seq=8K                0.253      0.248    0.98x         72.16         73.52
bs=8, seq=16K               0.410      0.451    1.10x         89.02         80.97
bs=8, seq=32K               0.743      0.851    1.15x         98.27         85.81
bs=8, seq=64K               1.392      1.628    1.17x        104.89         89.70
bs=32, seq=2K               0.364      0.327    0.90x         50.19         55.80
bs=32, seq=4K               0.536      0.545    1.02x         68.14         66.96
bs=32, seq=8K               0.870      0.951    1.09x         83.97         76.78
bs=32, seq=16K              1.575      1.780    1.13x         92.75         82.03
bs=32, seq=32K              2.918      3.430    1.18x        100.08         85.15
bs=32, seq=64K              5.560      8.031    1.44x        105.06         72.73


Note: performance may degrade in lower triton versions.
