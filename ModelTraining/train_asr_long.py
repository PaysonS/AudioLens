
import os, math, time, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio, soundfile as sf
from itertools import chain
from sentencepiece import SentencePieceProcessor
from datasets import load_dataset, Audio

from tiny_conformer_ctc import TinyConformerCTC
from spec_augment import spec_augment

EPOCHS            = 75           # total passes
STEPS_PER_EPOCH   = 2500         # minibatches per epoch
BATCH_SIZE        = 4
ACCUM             = 8            # effective batch = BATCH_SIZE*ACCUM
BASE_LR           = 5e-4
WARMUP_STEPS      = 1000
VALID_SAMPLES     = 50
VAL_TIME_BUDGET_S = 60
USE_SPEC_AUG      = True
SPM_PATH          = os.path.join("models", "bpe2k.model")  # or "bpe1k.model"
SR                = 16000
N_MELS            = 80


os.makedirs("models", exist_ok=True)

spm = SentencePieceProcessor(model_file=SPM_PATH)
VOCAB = spm.get_piece_size()
BLANK = VOCAB  # CTC blank id is vocab size

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=640, hop_length=160, win_length=320,
    n_mels=N_MELS, f_min=20, f_max=7600, power=1.0
)


def pick_device():
    # NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple 
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    # Intel 
    if hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None)) and torch.xpu.is_available():
        return torch.device("xpu")
    # Windows
    try:
        import torch_directml
        return torch_directml.device()  # special device object
    except Exception:
        pass
    # CPU fallback
    return torch.device("cpu")



def cmvn_timewise(x: torch.Tensor, n_mels: int = N_MELS) -> torch.Tensor:

    if x.dim() != 2:
        raise RuntimeError(f"Expected 2D mel, got {tuple(x.shape)}")
    if x.shape[0] == n_mels:             # [n_mels, T]
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (x - m) / s
    if x.shape[1] == n_mels:             # [T, n_mels]
        m = x.mean(dim=0, keepdim=True); s = x.std(dim=0, keepdim=True).clamp_min(1e-5)
        return ((x - m) / s).transpose(0, 1)
    raise RuntimeError(f"Unexpected mel shape {tuple(x.shape)}; cannot find n_mels={n_mels} on any axis")

def to_logmel_tensor(wav_np: np.ndarray) -> torch.Tensor:

    with torch.no_grad():
        x = mel(torch.from_numpy(wav_np.astype("float32", copy=False)).unsqueeze(0)) + 1e-6
        x = x.squeeze(0)  # [n_mels,T] or [T,n_mels]
        x = x.log()
        x = cmvn_timewise(x, n_mels=N_MELS)  # -> [n_mels, T]
        return x.contiguous()

def _iter_hf_local_splits(splits):
    """Yield (wav_numpy, text) from HF dataset **only when local files exist**."""
    for split in splits:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=False)
        ds = ds.cast_column("audio", Audio(decode=False)).with_format("python")
        for ex in ds:
            a = ex.get("audio", None)
            path = a.get("path") if isinstance(a, dict) else None
            if not path or not os.path.exists(path):
                continue
            txt = ex["text"].lower().strip()
            try:
                wav, sr = sf.read(path, dtype="float32", always_2d=False)
            except Exception:
                try:
                    wav_t, sr = torchaudio.load(path)
                    wav = wav_t.mean(0).numpy()
                except Exception:
                    continue
            if isinstance(wav, np.ndarray) and wav.ndim > 1:
                wav = wav.mean(1)
            if int(sr) != SR:
                wav = torchaudio.functional.resample(torch.from_numpy(wav), int(sr), SR).numpy()
            yield wav, txt

def _iter_torchaudio_train():
    roots = "."
    # order doesn't matter; you can shuffle downstream
    ds1 = torchaudio.datasets.LIBRISPEECH(roots, url="train-clean-100", download=True)
    ds2 = torchaudio.datasets.LIBRISPEECH(roots, url="train-clean-360", download=True)
    for ds in (ds1, ds2):
        for i in range(len(ds)):
            wav_t, sr, txt = ds[i][0], int(ds[i][1]), ds[i][2].lower().strip()
            wav = wav_t.mean(0).numpy()
            if sr != SR:
                wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
            yield wav, txt

class TrainIterable(torch.utils.data.IterableDataset):
    def __init__(self):
        self._sources = []
        # Probe HF local (openslr/librispeech_asr exposes "train", not "train.100/360")
        hf_iter = list(_take_n(_iter_hf_local_splits(["train"]), 5))
        if len(hf_iter) > 0:
            self._mode = "hf"
        else:
            self._mode = "ta"  # torchaudio fallback

    def __iter__(self):
        if self._mode == "hf":
            # Use the single "train" split from openslr/librispeech_asr
            src = _iter_hf_local_splits(["train"])
        else:
            src = _iter_torchaudio_train()
        for wav, txt in src:
            ids = spm.encode(txt)
            if not ids:
                continue
            yield torch.from_numpy(wav), torch.tensor(ids, dtype=torch.int32)


def _take_n(iterable, n):
    cnt = 0
    for x in iterable:
        yield x
        cnt += 1
        if cnt >= n:
            break

def collate(batch):
    feats, tgts, in_lens, tgt_lens = [], [], [], []
    for wav_t, ids in batch:
        f = to_logmel_tensor(wav_t.numpy())        # [n_mels, T_i]
        feats.append(f)
        tgts.append(ids)
        in_lens.append(f.shape[-1])
        tgt_lens.append(int(ids.numel()))
    if len(in_lens) == 0:
        # Empty batch (shouldn't happen, but be safe)
        return torch.zeros(0, N_MELS, 1), torch.tensor([], dtype=torch.int32), torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)
    T = max(in_lens); B = len(batch); F = feats[0].shape[0]
    out = torch.zeros(B, F, T, dtype=torch.float32)
    for i, f in enumerate(feats):
        out[i, :, :f.shape[-1]] = f
    # int64 lengths for CTC
    return out, torch.cat(tgts), torch.tensor(in_lens, dtype=torch.int64), torch.tensor(tgt_lens, dtype=torch.int64)

def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    R, H = len(r), len(h)
    dp = np.zeros((R + 1, H + 1), dtype=np.int32)
    dp[:, 0] = np.arange(R + 1); dp[0, :] = np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[R, H] / max(1, R)

def greedy_ids_from_logp_np(logp_np, blank_id=BLANK):
    ids, prev = [], -1
    for t in logp_np.argmax(axis=-1).tolist():
        if t != prev and t != blank_id:
            ids.append(int(t))
        prev = t
    return ids

@torch.no_grad()
def validate(model: nn.Module, n_samples: int = VALID_SAMPLES) -> float:
    model.eval()
    device = next(model.parameters()).device

    scored = 0
    total = 0.0
    start = time.time()

  
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=False)
    ds = ds.cast_column("audio", Audio(decode=False)).with_format("python")

    for ex in ds:
        a = ex.get("audio", None); path = a.get("path") if isinstance(a, dict) else None
        if not path or not os.path.exists(path):
            continue
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            try:
                wav_t, sr = torchaudio.load(path); wav = wav_t.mean(0).numpy()
            except Exception:
                continue
        if isinstance(wav, np.ndarray) and wav.ndim > 1:
            wav = wav.mean(1)
        if int(sr) != SR:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), int(sr), SR).numpy()

        feat = to_logmel_tensor(wav).unsqueeze(0).to(device)
        logp = model(feat).squeeze(0).cpu().numpy()           # [T,V]
        ids = greedy_ids_from_logp_np(logp, blank_id=BLANK)
        hyp = spm.decode_ids(ids).lower().strip()
        ref = ex["text"].lower().strip()

        total += cer(ref, hyp); scored += 1
        if scored >= n_samples or (time.time() - start) > VAL_TIME_BUDGET_S:
            break

    if scored == 0:
        print("[val] HF local paths unusable - falling back to torchaudio test-clean for validation stubs")
        td = torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=True)
        for i in range(min(n_samples, len(td))):
            wav_t, sr, ref = td[i][0], int(td[i][1]), td[i][2]
            wav = wav_t.mean(0).numpy()
            if sr != SR:
                wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
            feat = to_logmel_tensor(wav).unsqueeze(0).to(device)
            logp = model(feat).squeeze(0).cpu().numpy()
            ids = greedy_ids_from_logp_np(logp, blank_id=BLANK)
            hyp = spm.decode_ids(ids).lower().strip()
            total += cer(ref.lower().strip(), hyp); scored += 1
            if scored >= n_samples or (time.time() - start) > VAL_TIME_BUDGET_S:
                break

    model.train()
    if scored == 0:
        raise RuntimeError("Validation scored 0 samples - cannot report CER.")
    print(f"[val] scored {scored} samples")
    return total / scored

def ensure_log_probs(x: torch.Tensor) -> torch.Tensor:
    """Ensure [B,T,V] are log-probs before feeding CTC."""
    if torch.any(x > 0):  # crude check; logits usually include positives
        return torch.log_softmax(x, dim=-1)
    return x

def main():
    device = pick_device()
    print("Using device:", device)



    loader = DataLoader(TrainIterable(), batch_size=BATCH_SIZE, collate_fn=collate)
    model = TinyConformerCTC(vocab=BLANK + 1).to(device)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    opt = optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.98), weight_decay=1e-2)
    total_effective_steps = (EPOCHS * STEPS_PER_EPOCH) // max(1, ACCUM)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / max(1, WARMUP_STEPS)
        t_total = max(1, total_effective_steps - WARMUP_STEPS)
        s = min(step - WARMUP_STEPS, t_total)
        return 0.5 * (1.0 + math.cos(math.pi * s / t_total))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    opt.zero_grad()
    global_step = 0
    best_cer = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        step_in_epoch = 0
        trained_items = 0

        for feat, y, in_l, y_l in loader:
            step_in_epoch += 1
            global_step += 1
            trained_items += int(feat.size(0))

            feat = spec_augment(feat).to(device) if USE_SPEC_AUG else feat.to(device)  # [B,80,T]
            logp = model(feat)                               # [B,T_out,V]
            logp = ensure_log_probs(logp)

           
            B, T_out, V = logp.shape
            T_in = feat.shape[-1]

            in_l_scaled = (in_l * T_out // T_in).clamp(max=T_out).to(torch.int64)

            y_l_capped = torch.minimum(y_l, in_l_scaled)


            loss = ctc(logp.permute(1, 0, 2), y.to(device), in_l_scaled, y_l_capped) / ACCUM

            loss.backward()

            if global_step % ACCUM == 0:
                with torch.no_grad():
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            g = p.grad.data
                            total_norm_sq += float(torch.sum(g*g))
                    grad_norm = (total_norm_sq ** 0.5)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                sched.step()
            else:
                grad_norm = float('nan')

            if step_in_epoch % 50 == 0:
                eff_loss = float(loss) * ACCUM
                print(f"step {step_in_epoch}/{STEPS_PER_EPOCH}  loss={eff_loss:.3f}  lr={sched.get_last_lr()[0]:.2e}  grad_norm={grad_norm if grad_norm==grad_norm else 0.0:.2f}")

            if step_in_epoch >= STEPS_PER_EPOCH:
                break

        print(f"[train] seen {trained_items} items this epoch")
        if trained_items == 0:
            print("[WARN] No training audio seen. Falling back to torchaudio likely failed - check internet or dataset availability.")
       
        ep_path = f"models/asr_tiny_fp32_ep{epoch}.pt"
        torch.save(model.state_dict(), ep_path)
        print(f"? Saved {ep_path}")


        val_cer = validate(model, n_samples=VALID_SAMPLES)
        print(f"[val] CER={val_cer:.3f}   (best so far: {best_cer:.3f})")
        if val_cer < best_cer:
            best_cer = val_cer
            best_path = "models/asr_tiny_fp32_best_newest.pt"
            torch.save(model.state_dict(), best_path)
            print(f"? Saved new best: {best_path}")


    torch.save(model.state_dict(), "models/asr_tiny_fp32.pt")
    print("Saved FP32 model to models/asr_tiny_fp32.pt")

if __name__ == "__main__":
    main()

