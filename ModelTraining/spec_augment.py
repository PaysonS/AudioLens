
import torch, random

def spec_augment(x, max_t=30, max_f=8):
    # x: [B, 80, T]
    B, F, T = x.shape
    x = x.clone()
    for b in range(B):
        if T > 0:
            t = random.randint(0, min(max_t, max(0, T-1)))
            t0 = random.randint(0, max(0, T - t))
            x[b, :, t0:t0+t] = 0
        if F > 0:
            f = random.randint(0, min(max_f, max(0, F-1)))
            f0 = random.randint(0, max(0, F - f))
            x[b, f0:f0+f, :] = 0
    return x
