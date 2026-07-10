# verify_onnx.py
# Bandingkan output PyTorch vs ONNX dengan foto wajah sungguhan

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys

CKPT_PATH = r"C:\PythonProj\img_bnn\checkpoints_sw357_conv10_imgsign\SW357_conv10_imgsign\best_model_epoch39_plateau.pth"
ONNX_PATH = r"C:\PythonProj\img_bnn\imgnet_conv10_epoch39.onnx"
IMG_PATH  = sys.argv[1] if len(sys.argv) > 1 else None

# ── PyTorch model ──────────────────────────────────────────
class SWBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(240,64),nn.ReLU(True),nn.Linear(64,32))
    def forward(self, x):
        B,C,H,W=x.shape; diffs=[]
        for ws in [3,5,7]:
            p=ws//2; xp=F.pad(x,[p,p,p,p],mode='reflect')
            patches=xp.unfold(2,ws,1).unfold(3,ws,1)
            diff=x.unsqueeze(-1).unsqueeze(-1)-patches
            mid=ws//2
            mask=torch.ones(ws,ws,dtype=torch.bool); mask[mid,mid]=False
            diffs.append(diff[:,:,:,:,mask])
        d=torch.cat(diffs,-1); B,C,H,W,N=d.shape
        o=self.fc(d.permute(0,2,3,1,4).reshape(B*H*W,C*N))
        return o.reshape(B,H,W,-1).permute(0,3,1,2)

class IMGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sw1=SWBlock(); self.bn1=nn.BatchNorm2d(32)
        self.conv2 =nn.Conv2d(32, 64,3,stride=1,padding=1,bias=False); self.bn2 =nn.BatchNorm2d(64)
        self.conv3 =nn.Conv2d(64, 64,3,stride=2,padding=1,bias=False); self.bn3 =nn.BatchNorm2d(64)
        self.conv4 =nn.Conv2d(64,128,3,stride=1,padding=1,bias=False); self.bn4 =nn.BatchNorm2d(128)
        self.conv5 =nn.Conv2d(128,128,3,stride=1,padding=1,bias=False); self.bn5 =nn.BatchNorm2d(128)
        self.conv6 =nn.Conv2d(128,128,3,stride=2,padding=1,bias=False); self.bn6 =nn.BatchNorm2d(128)
        self.conv7 =nn.Conv2d(128,256,3,stride=1,padding=1,bias=False); self.bn7 =nn.BatchNorm2d(256)
        self.conv8 =nn.Conv2d(256,256,3,stride=1,padding=1,bias=False); self.bn8 =nn.BatchNorm2d(256)
        self.conv9 =nn.Conv2d(256,256,3,stride=2,padding=1,bias=False); self.bn9 =nn.BatchNorm2d(256)
        self.conv10=nn.Conv2d(256,256,3,stride=1,padding=1,bias=False); self.bn10=nn.BatchNorm2d(256)
        self.gap=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(256,1024); self.bn=nn.BatchNorm1d(1024)
    def forward(self,x):
        x=F.relu(self.bn1(self.sw1(x)))
        for i in range(2,11):
            x=F.relu(getattr(self,f'bn{i}')(getattr(self,f'conv{i}')(x)))
        return self.bn(self.fc(self.gap(x).view(x.size(0),-1)))

# Load
model = IMGNet()
st = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
if isinstance(st, dict) and "model" in st: st = st["model"]
model.load_state_dict(st); model.eval()

# Load ONNX
import onnxruntime as rt
sess = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

# Test dengan foto wajah atau random
if IMG_PATH:
    arr = np.array(Image.open(IMG_PATH).convert("RGB").resize((112,112), Image.BILINEAR))
    print(f"Testing dengan foto: {IMG_PATH}")
else:
    # Pakai foto sintetis yang mirip wajah (bukan pure random)
    arr = np.random.randint(50, 200, (112,112,3), dtype=np.uint8)
    print("Testing dengan gambar random...")

# Preprocess
t = torch.from_numpy(arr.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
t_np = t.numpy()

# Run keduanya
with torch.no_grad():
    e_torch = model(t).squeeze(0).numpy()
e_onnx = sess.run(None, {inp_name: t_np})[0][0]

# Compare
max_diff = np.abs(e_torch - e_onnx).max()
cos_sim  = np.dot(e_torch, e_onnx) / (np.linalg.norm(e_torch) * np.linalg.norm(e_onnx) + 1e-8)

print(f"\nPyTorch output[:5] : {e_torch[:5]}")
print(f"ONNX    output[:5] : {e_onnx[:5]}")
print(f"\nMax diff  : {max_diff:.8f}")
print(f"Cosine sim: {cos_sim:.8f}")
print(f"\nStatus: {'✓ IDENTIK' if cos_sim > 0.9999 else '✗ BERBEDA — masalah di ONNX!'}")