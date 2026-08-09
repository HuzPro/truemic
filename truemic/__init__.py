import os

# audioseal's vendored moshi torch.compiles its encoder at first call; inductor
# needs MSVC ('cl'), absent on this machine. Eager is fast enough. Must be set
# before audioseal/torch import.
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# torch's OpenMP pool busy-waits ~200 ms after every parallel region — measured
# 188 ms of CPU burned per 2 s idle on a 2-core laptop, zero slowdown disabled.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
