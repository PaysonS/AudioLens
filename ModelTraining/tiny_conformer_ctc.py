import torch, torch.nn as nn

class ConvSubsampler(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d, (3,3), stride=(2,2), padding=1), nn.ReLU(),  # 80->40
            nn.Conv2d(d, d, (3,3), stride=(2,1), padding=1), nn.ReLU())  # 40->20
        self.proj = nn.Linear(d*(80//4), d)  # 80//4 = 20

    def forward(self, x):  # x: [B,80,T]
        x = x.unsqueeze(1)              # [B,1,80,T]
        x = self.conv(x)                # [B,d,20,T']
        B,C,F,T = x.shape
        x = x.permute(0,3,1,2).reshape(B,T, C*F)
        return self.proj(x)             # [B,T,d]


class EncoderBlock(nn.Module):
    def __init__(self, d=192, ff=384, k=15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d*2, k, padding=k//2), nn.ReLU(),
            nn.Conv1d(d*2, d, 1)
        )
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x, attn_mask=None):
        # “fake attention” residual: just LN shortcut
        y = self.ln1(x)
        x = x + y

        y = self.conv(self.ln2(x).transpose(1, 2)).transpose(1, 2)
        x = x + y

        x = x + self.ffn(self.ln3(x))
        return x


class TinyConformerCTC(nn.Module):
    def __init__(self, vocab, d=192, layers=12):
        super().__init__()
        self.sub = ConvSubsampler(d)
        self.enc = nn.ModuleList([EncoderBlock(d) for _ in range(layers)])
        self.ctc = nn.Linear(d, vocab)

    def forward(self, feat, attn_mask=None):
        x = self.sub(feat)          # [B, T', d]
        for blk in self.enc:
            x = blk(x, attn_mask)
        return self.ctc(x)          # [B, T', vocab]  (LOGITS, not log-softmax)
