import torch
from tiny_conformer_ctc import TinyConformerCTC

SR = 16000
N_MELS = 80

CKPT_PATH = "models/asr_tiny_fp32_best_newest.pt"
ONNX_PATH = "tiny_conformer_conv_only_fp32_static.onnx"

state = torch.load(CKPT_PATH, map_location="cpu")

if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
    state = state["model"]
vocab_from_ckpt = state["ctc.weight"].shape[0]
print("Checkpoint vocab size:", vocab_from_ckpt)

model = TinyConformerCTC(vocab=vocab_from_ckpt)
model.load_state_dict(state)
model.eval()

T = 400
dummy = torch.randn(1, N_MELS, T, dtype=torch.float32)

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

