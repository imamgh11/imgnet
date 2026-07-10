# export_imgnet_onnx.py
# Export IMGNet Conv10 ke ONNX
# SW Block di-rewrite agar ONNX-compatible (slice instead of unfold+mask)
# Bobot tetap dari checkpoint asli — hasil embedding identik

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONFIG ─────────────────────────────────────────────────
CKPT_PATH = r"C:\PythonProj\img_bnn\checkpoints_sw357_conv10_imgsign\SW357_conv10_imgsign\best_model_epoch39_plateau.pth"
ONNX_PATH = r"C:\PythonProj\img_bnn\imgnet_conv10_epoch39.onnx"
OPSET     = 18


# ── SW BLOCK (ONNX-compatible version) ─────────────────────
class SWBlock(nn.Module):
    """
    SW Block — ONNX-compatible
    Reflect padding di-emulate dengan flip + concat
    Mathematically identik dengan training, 100% ONNX-safe
    """
    def __init__(self):
        super().__init__()
        self._off = {}
        for ws in [3, 5, 7]:
            mid = ws // 2
            self._off[ws] = [(dr, dc) for dr in range(ws) for dc in range(ws)
                             if not (dr == mid and dc == mid)]
        self.fc = nn.Sequential(
            nn.Linear(240, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def _reflect_pad(self, x, p):
        left   = x[:, :, :, 1:p+1].flip(dims=[3])
        right  = x[:, :, :, -(p+1):-1].flip(dims=[3])
        x = torch.cat([left, x, right], dim=3)
        top    = x[:, :, 1:p+1, :].flip(dims=[2])
        bottom = x[:, :, -(p+1):-1, :].flip(dims=[2])
        return torch.cat([top, x, bottom], dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        # Kumpulkan diff per window size, lalu per offset — sama persis dengan unfold+mask
        # unfold menghasilkan: (B,C,H,W,ws,ws) → flatten ws*ws → remove center
        # Ekuivalen: untuk tiap (dr,dc) dalam row-major order (skip center),
        #            diff[:,:,:,:,idx] = center - neighbor[dr,dc]
        all_diffs = []
        for ws in [3, 5, 7]:
            p   = ws // 2
            xp  = self._reflect_pad(x, p)
            mid = ws // 2
            # Kumpulkan per channel dulu, baru per offset — matching unfold layout
            # unfold layout: diff shape = (B, C, H, W, ws*ws-1)
            # FC input = (B*H*W, C*(ws*ws-1)*3windows) = (B*H*W, 240)
            ws_diffs = []
            for dr in range(ws):
                for dc in range(ws):
                    if dr == mid and dc == mid: continue
                    neighbor = xp[:, :, dr:dr+H, dc:dc+W]
                    ws_diffs.append(x - neighbor)  # (B, C, H, W)
            # Stack: (B, C*(ws²-1), H, W)
            all_diffs.extend(ws_diffs)

        # Susun sama dengan unfold: per window size, semua channel dulu baru offset
        # unfold: (B, C, H, W, N) → permute → (B*H*W, C*N)
        # kita: list of (B,C,H,W) dengan panjang 80 → cat dim=1 → (B, C*80, H, W)
        # Tapi FC expect (B*H*W, C*80) dengan susunan C bersebelahan untuk tiap offset
        # Perlu: untuk tiap posisi spatial, urutan input ke FC = [all_diffs_dim0, all_diffs_dim1, ...]
        # Ini sama dengan cat lalu permute

        # Reorder: unfold hasilkan (B, C, H, W, N_diff) → permute(0,2,3,1,4) → (B,H,W,C,N)
        # → reshape (B*H*W, C*N)
        # Kita punya list N_diff tensor (B,C,H,W) → stack dim=4 → (B,C,H,W,N)
        d = torch.stack(all_diffs, dim=4)  # (B, C, H, W, 80)
        B2, C2, H2, W2, N = d.shape
        d = d.permute(0, 2, 3, 1, 4).reshape(B2 * H2 * W2, C2 * N)  # (B*H*W, C*80=240)
        o = self.fc(d)
        return o.reshape(B2, H2, W2, -1).permute(0, 3, 1, 2)


# ── IMGNET MODEL ───────────────────────────────────────────
class IMGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sw1    = SWBlock();            self.bn1  = nn.BatchNorm2d(32)
        self.conv2  = nn.Conv2d(32,  64,  3, stride=1, padding=1, bias=False); self.bn2  = nn.BatchNorm2d(64)
        self.conv3  = nn.Conv2d(64,  64,  3, stride=2, padding=1, bias=False); self.bn3  = nn.BatchNorm2d(64)
        self.conv4  = nn.Conv2d(64,  128, 3, stride=1, padding=1, bias=False); self.bn4  = nn.BatchNorm2d(128)
        self.conv5  = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False); self.bn5  = nn.BatchNorm2d(128)
        self.conv6  = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False); self.bn6  = nn.BatchNorm2d(128)
        self.conv7  = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False); self.bn7  = nn.BatchNorm2d(256)
        self.conv8  = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False); self.bn8  = nn.BatchNorm2d(256)
        self.conv9  = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False); self.bn9  = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False); self.bn10 = nn.BatchNorm2d(256)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, 1024)
        self.bn     = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.sw1(x)))
        for i in range(2, 11):
            x = F.relu(getattr(self, f'bn{i}')(getattr(self, f'conv{i}')(x)))
        x = self.gap(x).view(x.size(0), -1)
        return self.bn(self.fc(x))


# ── EXPORT ─────────────────────────────────────────────────
def export():
    device = torch.device("cpu")  # export dari CPU untuk portabilitas
    print(f"Device : {device}")

    # Load model
    model = IMGNet().to(device)
    st = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "model" in st: st = st["model"]
    model.load_state_dict(st, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded — {n_params:,} params (~{n_params*4/1024/1024:.2f} MB FP32)")

    # Test forward
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        out = model(dummy)
    print(f"✓ Forward pass OK — output: {out.shape}")

    # Export ONNX
    print(f"\nExporting ONNX (opset {OPSET}) ...")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        opset_version        = OPSET,
        input_names          = ["input"],
        output_names         = ["embedding"],
        dynamic_axes         = {
            "input"    : {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding  = True,
        verbose              = False,
        export_params        = True,
    )
    print(f"✓ ONNX saved: {ONNX_PATH}")

    # Verify
    try:
        import onnx, onnxruntime as rt

        onnx.checker.check_model(onnx.load(ONNX_PATH))
        print(f"✓ ONNX model valid (checker passed)")

        sess     = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        out_onnx = sess.run(["embedding"], {"input": dummy.numpy()})[0]
        max_diff = float(np.abs(out_onnx - out.numpy()).max())
        cos_sim  = float(np.dot(out_onnx[0], out.numpy()[0]) /
                        (np.linalg.norm(out_onnx[0]) * np.linalg.norm(out.numpy()[0]) + 1e-8))
        print(f"✓ PyTorch vs ONNX max diff  : {max_diff:.6f}  {'✓ OK' if max_diff < 1e-3 else '⚠ WARNING'}")
        print(f"✓ PyTorch vs ONNX cosine sim: {cos_sim:.6f}  {'✓ OK' if cos_sim > 0.999 else '⚠ EMBEDDING BERBEDA!'}")

        size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
        print(f"✓ File size: {size_mb:.2f} MB")

        # Test batch size > 1
        dummy2   = torch.randn(4, 3, 112, 112)
        out_b4   = sess.run(["embedding"], {"input": dummy2.numpy()})[0]
        print(f"✓ Batch test (4×): output shape {out_b4.shape}")

        print(f"\n{'='*55}")
        print(f"EXPORT SUKSES")
        print(f"  Input    : (batch, 3, 112, 112) FLOAT32")
        print(f"  Output   : (batch, 1024) FLOAT32 — embedding")
        print(f"  Opset    : {OPSET}")
        print(f"  Size     : {size_mb:.2f} MB")
        print(f"  Dynamic  : batch size (bisa 1, 4, 8, ...)")
        print(f"  Path     : {ONNX_PATH}")
        print(f"{'='*55}")

    except ImportError as e:
        print(f"Verification skip — install: pip install onnx onnxruntime ({e})")
    except Exception as e:
        print(f"Verification error: {e}")


if __name__ == "__main__":
    export()