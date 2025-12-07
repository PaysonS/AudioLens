import torch
from tiny_conformer_ctc import TinyConformerCTC

SR = 16000
N_MELS = 80

CKPT_PATH = "models/asr_tiny_fp32_best_newest.pt"
ONNX_PATH = "tiny_conformer_conv_only_fp32_static.onnx"

# 1) Load state dict
state = torch.load(CKPT_PATH, map_location="cpu")

# If you ever wrap with "model": {...}, unwrap here
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
    state = state["model"]

# 2) Infer vocab from final linear layer
vocab_from_ckpt = state["ctc.weight"].shape[0]
print("Checkpoint vocab size:", vocab_from_ckpt)

model = TinyConformerCTC(vocab=vocab_from_ckpt)
model.load_state_dict(state)
model.eval()

# 3) Static dummy input, e.g. 400 frames
T = 400
dummy = torch.randn(1, N_MELS, T, dtype=torch.float32)

# 4) Export ONNX (no dynamic axes, no log-softmax)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["mel"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes=None
)

print(f"Saved {ONNX_PATH}")
