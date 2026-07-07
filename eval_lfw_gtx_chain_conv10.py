# eval_lfw_gtx_chain_conv10.py — eval LFW untuk checkpoint SW357+Conv7 di GTX lokal
# Output: AMP IMG Score + Chain Score (metric saja, bukan loss)
# Jalankan di GTX, arahkan ke checkpoint yang didownload dari A100/Drive

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── PATH CONFIG (GTX lokal) ───────────────────────
CKPT_PATH  = r"checkpoints_sw357_conv10\SW357_conv10\best_model_epoch1.pth"
LFW_DIR    = "lfw"
PAIRS_FILE = "pairs.txt"

# ── CONFIG (identik training) ─────────────────────
WINDOW_SIZE   = 11
THRESHOLD     = 8
EMB_DIM       = 1024

# ── Chain Score params (hasil kalibrasi) ──────────
NEUTRAL_LEN  = 29
REWARD_RATE  = 0.3
PUNISH_RATE  = 1.0


# ============================================================
# MODEL (identik dengan training)
# ============================================================
class SWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, window_sizes=[3, 5, 7]):
        super().__init__()
        self.window_sizes = window_sizes
        n_diff  = sum(w * w - 1 for w in window_sizes)
        n_input = n_diff * in_ch
        self.fc = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        diffs = []
        for ws in self.window_sizes:
            pad     = ws // 2
            x_pad   = F.pad(x, [pad, pad, pad, pad], mode='reflect')
            patches = x_pad.unfold(2, ws, 1).unfold(3, ws, 1)
            center  = x.unsqueeze(-1).unsqueeze(-1)
            diff    = center - patches
            mid     = ws // 2
            mask    = torch.ones(ws, ws, dtype=torch.bool, device=x.device)
            mask[mid, mid] = False
            diff    = diff[:, :, :, :, mask]
            diffs.append(diff)
        diffs = torch.cat(diffs, dim=-1)
        B, C, H, W, N = diffs.shape
        diffs = diffs.permute(0, 2, 3, 1, 4).reshape(B * H * W, C * N)
        out = self.fc(diffs)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class IMGNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, window_sizes=[3, 5, 7]):
        super().__init__()
        self.sw1    = SWBlock(3, 32, window_sizes)
        self.bn1    = nn.BatchNorm2d(32)
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
        self.fc     = nn.Linear(256, emb_dim)
        self.bn     = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.sw1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.gap(x).view(x.size(0), -1)
        return self.bn(self.fc(x))


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def amp_img_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    total = 0.0
    for i in range(n):
        w1, w2 = e1[i:i+WINDOW_SIZE], e2[i:i+WINDOW_SIZE]
        s1 = np.where(w1 >= 0, 1, -1).astype(np.int8)
        s2 = np.where(w2 >= 0, 1, -1).astype(np.int8)
        if int(np.sum(s1 == s2)) >= THRESHOLD:
            a1, a2 = np.mean(np.abs(w1)), np.mean(np.abs(w2))
            total += max(0.0, 1 - abs(a1 - a2) / max(a1, a2, 1e-6))
    return total / n


def chain_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    if n <= 0:
        return 0.0, 0, 0.0, 0.0

    match_flags = []
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        match_flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)

    total_match = sum(match_flags)
    img_sign    = total_match / n

    n_chains = 0
    in_chain = False
    for a in match_flags:
        if a and not in_chain:
            n_chains += 1; in_chain = True
        elif not a:
            in_chain = False

    if n_chains == 0 or total_match == 0:
        return 0.0, 0, 0.0, img_sign

    avg_chain = total_match / n_chains
    diff      = avg_chain - NEUTRAL_LEN
    base      = img_sign * 100
    score     = base + (REWARD_RATE * diff if diff >= 0 else PUNISH_RATE * diff)
    return float(np.clip(score, 0, 100)), n_chains, avg_chain, img_sign


# ============================================================
# UTILS
# ============================================================
def parse_pairs(pairs_file, lfw_dir):
    pairs = []
    with open(pairs_file) as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            name, i1, i2 = parts
            pairs.append((os.path.join(lfw_dir, name, f"{name}_{int(i1):04d}.jpg"),
                          os.path.join(lfw_dir, name, f"{name}_{int(i2):04d}.jpg"), 1))
        elif len(parts) == 4:
            n1, i1, n2, i2 = parts
            pairs.append((os.path.join(lfw_dir, n1, f"{n1}_{int(i1):04d}.jpg"),
                          os.path.join(lfw_dir, n2, f"{n2}_{int(i2):04d}.jpg"), 0))
    return pairs


_MTCNN = None
def get_mtcnn(device):
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN
        _MTCNN = MTCNN(image_size=112, keep_all=False, device=device, post_process=False)
    return _MTCNN


def get_emb(model, path, device):
    img  = Image.open(path).convert('RGB')
    face = get_mtcnn(device)(img)
    if face is not None:
        arr = face.cpu().numpy().astype(np.float32) / 255.0
        t   = torch.from_numpy(arr).unsqueeze(0).to(device)
    else:
        img = img.resize((112, 112), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).squeeze(0).cpu().numpy()


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {CKPT_PATH}")

    # Cek path
    for label, path, check in [
        ("CKPT_PATH",  CKPT_PATH,  os.path.isfile),
        ("LFW_DIR",    LFW_DIR,    os.path.isdir),
        ("PAIRS_FILE", PAIRS_FILE, os.path.isfile),
    ]:
        status = "✓" if check(path) else "✗"
        print(f"{status} {label:<12} : {path}")
    if not all([os.path.isfile(CKPT_PATH), os.path.isdir(LFW_DIR), os.path.isfile(PAIRS_FILE)]):
        print("\n[ERROR] Ada path yang tidak ditemukan."); return

    # Load model
    model = IMGNet(emb_dim=EMB_DIM, window_sizes=[3, 5, 7]).to(device)
    state = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,} (~{total*4/1024/1024:.2f} MB)\n")

    # Eval
    pairs = parse_pairs(PAIRS_FILE, LFW_DIR)
    amp_scores, chain_scores, labels = [], [], []
    chain_n_list, chain_avg_list = [], []
    skipped = 0

    for idx, (p1, p2, label) in enumerate(pairs):
        try:
            if not os.path.exists(p1) or not os.path.exists(p2):
                skipped += 1; continue
            e1 = get_emb(model, p1, device)
            e2 = get_emb(model, p2, device)
            amp_scores.append(amp_img_score_np(e1, e2))
            cs, nc, ac, _ = chain_score_np(e1, e2)
            chain_scores.append(cs)
            chain_n_list.append(nc)
            chain_avg_list.append(ac)
            labels.append(label)
        except:
            skipped += 1
        if (idx + 1) % 1000 == 0:
            print(f"  [{idx+1}/{len(pairs)}] skipped={skipped}")

    amp_scores    = np.array(amp_scores)
    chain_scores  = np.array(chain_scores)
    labels        = np.array(labels)
    chain_n_arr   = np.array(chain_n_list)
    chain_avg_arr = np.array(chain_avg_list)
    same = labels == 1; diff = labels == 0

    # Sweep AMP threshold
    best_amp_acc, best_amp_thr = 0, 0.5
    for thr in np.arange(0.0, 1.01, 0.01):
        acc = np.mean((amp_scores >= thr).astype(int) == labels)
        if acc > best_amp_acc:
            best_amp_acc, best_amp_thr = acc, thr

    # Sweep Chain threshold
    best_chain_acc, best_chain_thr = 0, 50.0
    for thr in np.arange(0, 101, 1):
        acc = np.mean((chain_scores >= thr).astype(int) == labels)
        if acc > best_chain_acc:
            best_chain_acc, best_chain_thr = acc, thr

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc_amp   = roc_auc_score(labels, amp_scores)
        auc_chain = roc_auc_score(labels, chain_scores)
        fpr, tpr, _ = roc_curve(labels, amp_scores)
        idx_far = np.where(fpr <= 0.001)[0]
        tar = tpr[idx_far[-1]] if len(idx_far) > 0 else 0.0
    except:
        auc_amp = auc_chain = tar = -1

    print(f"\n{'='*55}")
    print(f"LFW EVAL — SW357+Conv10")
    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Pairs      : {len(labels)} (skipped={skipped})")
    print(f"{'='*55}")
    print(f"\n── AMP IMG Score ─────────────────────────────")
    print(f"  Accuracy  : {best_amp_acc*100:.2f}%  (thr={best_amp_thr:.2f})")
    print(f"  AUC       : {auc_amp:.4f}")
    print(f"  TAR@0.1%  : {tar*100:.2f}%")
    print(f"  Same mean : {amp_scores[same].mean():.4f}")
    print(f"  Diff mean : {amp_scores[diff].mean():.4f}")
    print(f"  Gap       : {amp_scores[same].mean()-amp_scores[diff].mean():.4f}")
    print(f"\n── Chain Score (metric, bukan loss) ──────────")
    print(f"  Accuracy  : {best_chain_acc*100:.2f}%  (thr={best_chain_thr:.0f})")
    print(f"  AUC       : {auc_chain:.4f}")
    print(f"  Same mean : {chain_scores[same].mean():.2f}  "
          f"(avg_chain={chain_avg_arr[same].mean():.1f}, n_chains={chain_n_arr[same].mean():.1f})")
    print(f"  Diff mean : {chain_scores[diff].mean():.2f}  "
          f"(avg_chain={chain_avg_arr[diff].mean():.1f}, n_chains={chain_n_arr[diff].mean():.1f})")
    print(f"  Gap       : {chain_scores[same].mean()-chain_scores[diff].mean():.2f}")
    print(f"  Chain gap : {chain_avg_arr[same].mean()-chain_avg_arr[diff].mean():.2f}")
    print(f"{'='*55}")

    # Simpan hasil
    out = CKPT_PATH.replace('.pth', '_lfw_result.txt')
    with open(out, 'w') as f:
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"AMP Accuracy: {best_amp_acc*100:.2f}% (thr={best_amp_thr:.2f})\n")
        f.write(f"AMP AUC: {auc_amp:.4f}\n")
        f.write(f"TAR@0.1%: {tar*100:.2f}%\n")
        f.write(f"Chain Accuracy: {best_chain_acc*100:.2f}% (thr={best_chain_thr:.0f})\n")
        f.write(f"Chain AUC: {auc_chain:.4f}\n")
        f.write(f"Same avg_chain: {chain_avg_arr[same].mean():.1f}\n")
        f.write(f"Diff avg_chain: {chain_avg_arr[diff].mean():.1f}\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
