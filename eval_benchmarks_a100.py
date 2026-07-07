# eval_benchmarks_a100.py
# Eval IMGNet Conv10 di LFW, AgeDB-30, CALFW, CPLFW
# Format: dataset_test.zip dari Drive (format ann.txt atau pairs.txt)

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Install mtcnn kalau belum ada
try:
    from mtcnn import MTCNN as _MTCNN
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "mtcnn", "-q"], check=True)
    from mtcnn import MTCNN as _MTCNN

_mtcnn_instance = None
def get_mtcnn():
    global _mtcnn_instance
    if _mtcnn_instance is None:
        _mtcnn_instance = _MTCNN()
    return _mtcnn_instance

# ── PATH CONFIG ────────────────────────────────────
CKPT_PATH   = "/content/best_model_epoch39_plateau.pth"
EXTRACT_DIR = "/content/dataset_test"  # sudah di-extract sebelum run script ini

# ── CONFIG ────────────────────────────────────────
WINDOW_SIZE  = 11
THRESHOLD    = 8
EMB_DIM      = 1024
NEUTRAL_LEN  = 29
REWARD_RATE  = 0.3
PUNISH_RATE  = 1.0


# ============================================================
# MODEL
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
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.sw1    = SWBlock(3, 32, window_sizes=[3, 5, 7])
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
def img_sign_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    mc = 0
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        if int(np.sum(s1 == s2)) >= THRESHOLD: mc += 1
    return mc / n


def chain_score_np(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    if n <= 0: return 0.0, 0, 0.0
    match_flags = []
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        match_flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)
    total = sum(match_flags); img_sign = total / n
    n_chains = 0; in_chain = False
    for a in match_flags:
        if a and not in_chain: n_chains += 1; in_chain = True
        elif not a: in_chain = False
    if n_chains == 0 or total == 0: return 0.0, 0, 0.0
    avg_chain = total / n_chains
    diff      = avg_chain - NEUTRAL_LEN
    # Tidak x100 — range 0-1 sama dengan IMG Sign
    score     = img_sign + (REWARD_RATE * diff if diff >= 0 else PUNISH_RATE * diff) / 100.0
    return float(np.clip(score, 0, 1)), n_chains, avg_chain


def cosine_score_np(e1, e2):
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))


# ============================================================
# LOAD IMAGE (pre-aligned, no MTCNN needed)
# ============================================================
def get_emb(model, path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img.resize((112, 112), Image.BILINEAR), dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(t).squeeze(0).cpu().numpy()


def get_emb_batch(model, paths, device, batch_size=128):
    """Batch embedding — jauh lebih cepat dari satu-satu"""
    embeddings = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i+batch_size]:
            try:
                img = Image.open(p).convert('RGB')
                arr = np.array(img.resize((112, 112), Image.BILINEAR), dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(arr).permute(2, 0, 1))
            except:
                batch.append(torch.zeros(3, 112, 112))
        t = torch.stack(batch).to(device)
        with torch.no_grad():
            embs = model(t).cpu().numpy()
        embeddings.extend(embs)
    return np.array(embeddings)


def img_sign_score_batch(embs1, embs2):
    """Vectorized IMG Sign Score untuk semua pairs sekaligus"""
    n_pairs = len(embs1)
    n_win   = embs1.shape[1] - WINDOW_SIZE + 1
    scores  = np.zeros(n_pairs)
    for i in range(n_win):
        w1 = embs1[:, i:i+WINDOW_SIZE]
        w2 = embs2[:, i:i+WINDOW_SIZE]
        s1 = np.sign(w1).astype(np.int8)
        s2 = np.sign(w2).astype(np.int8)
        match = (s1 == s2).sum(axis=1) >= THRESHOLD
        scores += match.astype(float)
    return scores / n_win


# ============================================================
# PARSE PAIRS — support ann.txt dan pairs.txt
# ============================================================
def parse_ann(ann_file, img_dir):
    """Format: label img1_path img2_path (label di kolom pertama)"""
    pairs = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label = int(parts[0])
                p1    = os.path.join(img_dir, parts[1])
                p2    = os.path.join(img_dir, parts[2])
                pairs.append((p1, p2, label))
    return pairs


def parse_lfw_pairs(pairs_file, img_dir):
    """Format LFW standar (tab-separated)"""
    pairs = []
    with open(pairs_file) as f:
        lines = f.read().strip().split('\n')
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            name, i1, i2 = parts
            pairs.append((
                os.path.join(img_dir, name, f"{name}_{int(i1):04d}.jpg"),
                os.path.join(img_dir, name, f"{name}_{int(i2):04d}.jpg"), 1))
        elif len(parts) == 4:
            n1, i1, n2, i2 = parts
            pairs.append((
                os.path.join(img_dir, n1, f"{n1}_{int(i1):04d}.jpg"),
                os.path.join(img_dir, n2, f"{n2}_{int(i2):04d}.jpg"), 0))
    return pairs


def find_pairs(dataset_dir, name):
    """Cari pairs file di folder dataset"""
    # Coba berbagai format
    for ann in ['ann.txt', 'pairs.txt', f'{name}_ann.txt', f'{name}_pairs.txt']:
        path = os.path.join(dataset_dir, ann)
        if os.path.exists(path):
            return path, 'ann' if 'ann' in ann else 'lfw'
    return None, None


# ============================================================
# EVAL SATU DATASET
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


def eval_dataset(model, device, pairs, name):
    p1_list = [p1 for p1, p2, _ in pairs]
    p2_list = [p2 for p1, p2, _ in pairs]
    labels  = np.array([l for _, _, l in pairs])

    print(f"  Computing embeddings (batch=128)...")
    embs1 = get_emb_batch(model, p1_list, device)
    embs2 = get_emb_batch(model, p2_list, device)

    print(f"  Computing metrics...")
    sign_scores  = img_sign_score_batch(embs1, embs2)
    amp_scores   = np.array([amp_img_score_np(e1, e2) for e1, e2 in zip(embs1, embs2)])
    chain_scores = np.array([chain_score_np(e1, e2)[0] for e1, e2 in zip(embs1, embs2)])
    cos_scores   = np.array([cosine_score_np(e1, e2) for e1, e2 in zip(embs1, embs2)])

    def best_acc(scores, mn, mx, steps=200):
        best, thr = 0, mn
        for t in np.linspace(mn, mx, steps):
            acc = np.mean((scores >= t).astype(int) == labels)
            if acc > best: best, thr = acc, t
        return best, thr

    # Sweep threshold HANYA dari IMG Sign
    sign_acc, sign_thr = best_acc(sign_scores, 0.0, 1.0)

    # AMP dan Chain pakai threshold yang SAMA dari IMG Sign
    amp_acc   = np.mean((amp_scores   >= sign_thr).astype(int) == labels)
    chain_acc = np.mean((chain_scores >= sign_thr).astype(int) == labels)
    cos_acc, cos_thr = best_acc(cos_scores, -1.0, 1.0)

    # Voting: threshold dari IMG Sign dipakai untuk ketiganya
    votes = ((sign_scores  >= sign_thr).astype(int) +
             (amp_scores   >= sign_thr).astype(int) +
             (chain_scores >= sign_thr).astype(int))
    voting_acc_1 = np.mean((votes >= 1).astype(int) == labels)
    voting_acc_2 = np.mean((votes >= 2).astype(int) == labels)

    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) < 2:
            auc_sign = auc_chain = auc_cos = auc_amp = -1
        else:
            auc_sign  = roc_auc_score(labels, sign_scores)
            auc_chain = roc_auc_score(labels, chain_scores)
            auc_cos   = roc_auc_score(labels, cos_scores)
            auc_amp   = roc_auc_score(labels, amp_scores)
    except: auc_sign = auc_chain = auc_cos = auc_amp = -1

    return {
        'name'        : name,
        'n_pairs'     : len(labels),
        'sign_acc'    : sign_acc,
        'sign_thr'    : sign_thr,
        'amp_acc'     : amp_acc,
        'chain_acc'   : chain_acc,
        'cos_acc'     : cos_acc,
        'voting_acc_1': voting_acc_1,
        'voting_acc_2': voting_acc_2,
        'auc_sign'    : auc_sign,
        'auc_chain'   : auc_chain,
        'auc_amp'     : auc_amp,
        'auc_cos'     : auc_cos,
    }




# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Cek folder sudah ada
    if not os.path.exists(EXTRACT_DIR):
        print(f"[ERROR] {EXTRACT_DIR} tidak ditemukan!")
        print("Jalankan dulu di cell terpisah:")
        print('  import subprocess')
        print('  subprocess.run(["unzip", "-q", "/content/drive/MyDrive/dataset/dataset_test.zip", "-d", "/content/dataset_test"])')
        return

    # List isi folder
    print(f"\nIsi {EXTRACT_DIR}:")
    for item in sorted(os.listdir(EXTRACT_DIR)):
        print(f"  {item}")

    # Load model
    print(f"\nLoading IMGNet Conv10...")
    model = IMGNet(emb_dim=EMB_DIM).to(device)
    state = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,} (~{total*4/1024/1024:.2f} MB)")

    # Deteksi dataset yang ada
    datasets = []
    extract_contents = os.listdir(EXTRACT_DIR)

    # Cek apakah ada subfolder val/
    val_dir = os.path.join(EXTRACT_DIR, 'val')
    search_dir = val_dir if os.path.isdir(val_dir) else EXTRACT_DIR

    for item in sorted(os.listdir(search_dir)):
        item_path = os.path.join(search_dir, item)
        if item.endswith('.txt'):
            datasets.append((item.replace('.txt',''), search_dir,
                           os.path.join(search_dir, item), 'ann'))

    if not datasets:
        # Coba parse langsung semua .txt di search_dir
        print("\nTidak ada subfolder dataset — cek struktur folder:")
        for root, dirs, files in os.walk(EXTRACT_DIR):
            for f in files[:5]:
                print(f"  {os.path.join(root, f)}")
            if len(files) > 5:
                print(f"  ... dan {len(files)-5} file lainnya")
            break
        return

    # Validasi semua dataset dulu sebelum eval apapun
    print(f"\n{'='*60}")
    print("VALIDASI PATH SEMUA DATASET")
    print(f"{'='*60}")
    all_valid = True
    for ds_name, img_dir, ann_path, fmt in datasets:
        if fmt == 'ann':
            pairs_check = parse_ann(ann_path, img_dir)
        else:
            pairs_check = parse_lfw_pairs(ann_path, img_dir)

        if not pairs_check:
            print(f"✗ {ds_name:<20} : pairs kosong")
            all_valid = False
            continue

        p1, p2, _ = pairs_check[0]
        p1_ok = os.path.exists(p1)
        p2_ok = os.path.exists(p2)
        status = "✓" if (p1_ok and p2_ok) else "✗"
        print(f"{status} {ds_name:<20} : {len(pairs_check)} pairs")
        if not p1_ok:
            print(f"  TIDAK ADA: {p1}")
            all_valid = False
        if not p2_ok:
            print(f"  TIDAK ADA: {p2}")
            all_valid = False

    if not all_valid:
        print(f"\n[ERROR] Ada dataset yang path-nya tidak valid — eval dibatalkan")
        print("Periksa struktur folder dan nama file di dataset_test/val/")
        return

    print(f"\n✓ Semua dataset valid — mulai eval")
    print(f"{'='*60}")
    results = []
    for ds_name, img_dir, ann_path, fmt in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {ds_name}")
        print(f"Ann file  : {ann_path}")
        print(f"Format    : {fmt}")

        if fmt == 'ann':
            pairs = parse_ann(ann_path, img_dir)
        else:
            pairs = parse_lfw_pairs(ann_path, img_dir)

        print(f"Pairs     : {len(pairs)}")
        result = eval_dataset(model, device, pairs, ds_name)
        if result:
            results.append(result)

    # Ringkasan
    if results:
        print(f"\n{'='*85}")
        print(f"RINGKASAN — IMGNet Conv10 IMG Sign (epoch 39 plateau)")
        print(f"{'='*85}")
        print(f"{'Dataset':<14} {'IMG Sign':>10} {'AMP':>10} {'Chain':>10} {'Vote 1/3':>10} {'Vote 2/3':>10} {'Cosine':>10}")
        print(f"{'─'*75}")
        for r in results:
            print(f"{r['name']:<14} "
                  f"{r['sign_acc']*100:>9.2f}% "
                  f"{r['amp_acc']*100:>9.2f}% "
                  f"{r['chain_acc']*100:>9.2f}% "
                  f"{r['voting_acc_1']*100:>9.2f}% "
                  f"{r['voting_acc_2']*100:>9.2f}% "
                  f"{r['cos_acc']*100:>9.2f}%")
        print(f"{'='*85}")

        out = "/content/drive/MyDrive/dataset/benchmark_results_epoch39_plateau.txt"
        with open(out, 'w') as f:
            f.write("IMGNet Conv10 IMG Sign — Benchmark Results (epoch 39 plateau)\n")
            f.write(f"Checkpoint: {CKPT_PATH}\n\n")
            f.write(f"{'Dataset':<14} {'IMG Sign':>10} {'AMP':>10} {'Chain':>10} {'Vote 1/3':>10} {'Vote 2/3':>10} {'Cosine':>10}\n")
            f.write("─"*75 + "\n")
            for r in results:
                f.write(f"{r['name']:<14} "
                        f"{r['sign_acc']*100:>9.2f}% "
                        f"{r['amp_acc']*100:>9.2f}% "
                        f"{r['chain_acc']*100:>9.2f}% "
                        f"{r['voting_acc_1']*100:>9.2f}% "
                        f"{r['voting_acc_2']*100:>9.2f}% "
                        f"{r['cos_acc']*100:>9.2f}%\n")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
