# eval_metric_comparison_a100.py
# Test metric IMG Sign / AMP / Chain di embedding FaceNet dan ArcFace
# TIDAK include IMGNet — hanya sebagai metric pembanding
# buffalo_l = ArcFace buffalo_l dari insightface

import os, numpy as np
from PIL import Image

# Install dependencies
import subprocess
subprocess.run(["pip", "install", "facenet-pytorch", "insightface", "onnxruntime-gpu", "-q"], check=True)

# ── PATH CONFIG ────────────────────────────────────
EXTRACT_DIR = "/content/dataset_test"
RESULT_PATH = "/content/drive/MyDrive/dataset/metric_comparison_facenet_arcface.txt"

# ── CONFIG ────────────────────────────────────────
WINDOW_SIZE = 11
THRESHOLD   = 8
NEUTRAL_LEN = 29
REWARD_RATE = 0.3
PUNISH_RATE = 1.0

# Fixed threshold dari IMGNet epoch 29 — dipakai untuk semua embedding
THR_SIGN  = 0.79
THR_AMP   = 0.79
THR_CHAIN = 82.0


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def img_sign_score_batch(embs1, embs2):
    n_win = embs1.shape[1] - WINDOW_SIZE + 1
    scores = np.zeros(len(embs1))
    for i in range(n_win):
        s1 = np.sign(embs1[:, i:i+WINDOW_SIZE]).astype(np.int8)
        s2 = np.sign(embs2[:, i:i+WINDOW_SIZE]).astype(np.int8)
        scores += ((s1 == s2).sum(axis=1) >= THRESHOLD).astype(float)
    return scores / n_win

def amp_img_score_batch(embs1, embs2):
    n_win = embs1.shape[1] - WINDOW_SIZE + 1
    scores = np.zeros(len(embs1))
    for i in range(n_win):
        w1, w2 = embs1[:, i:i+WINDOW_SIZE], embs2[:, i:i+WINDOW_SIZE]
        s1 = np.sign(w1).astype(np.int8); s2 = np.sign(w2).astype(np.int8)
        match = (s1 == s2).sum(axis=1) >= THRESHOLD
        a1 = np.abs(w1).mean(axis=1); a2 = np.abs(w2).mean(axis=1)
        denom = np.maximum(a1, a2); denom = np.where(denom < 1e-6, 1e-6, denom)
        rel_sim = np.clip(1 - np.abs(a1 - a2) / denom, 0, 1)
        scores += (rel_sim * match.astype(float))
    return scores / n_win

def chain_score_single(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    if n <= 0: return 0.0
    flags = []
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)
    total = sum(flags); img_sign = total / n
    n_chains = 0; in_chain = False
    for a in flags:
        if a and not in_chain: n_chains += 1; in_chain = True
        elif not a: in_chain = False
    if n_chains == 0 or total == 0: return 0.0
    avg_chain = total / n_chains
    diff = avg_chain - NEUTRAL_LEN
    # Tidak dikali 100 — range 0-1 sama dengan IMG Sign
    score = img_sign + (REWARD_RATE * diff if diff >= 0 else PUNISH_RATE * diff) / 100.0
    return float(np.clip(score, 0, 1))

def cosine_batch(embs1, embs2):
    n1 = np.linalg.norm(embs1, axis=1, keepdims=True)
    n2 = np.linalg.norm(embs2, axis=1, keepdims=True)
    return (embs1 * embs2).sum(axis=1) / (n1.squeeze() * n2.squeeze() + 1e-8)


# ============================================================
# EVAL
# ============================================================
def eval_config(embs1, embs2, labels, name):
    labels = np.array(labels)
    print(f"  Computing metrics for {name}...")

    sign_scores  = img_sign_score_batch(embs1, embs2)
    amp_scores   = amp_img_score_batch(embs1, embs2)
    chain_scores = np.array([chain_score_single(e1,e2) for e1,e2 in zip(embs1,embs2)])
    cos_scores   = cosine_batch(embs1, embs2)

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

    # Voting pakai fixed threshold dari IMG Sign
    votes = ((sign_scores  >= sign_thr).astype(int) +
             (amp_scores   >= sign_thr).astype(int) +
             (chain_scores >= sign_thr).astype(int))
    v1_acc = np.mean((votes >= 1).astype(int) == labels)
    v2_acc = np.mean((votes >= 2).astype(int) == labels)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, sign_scores) if len(np.unique(labels)) > 1 else -1
    except: auc = -1

    return {
        'name'      : name,
        'n_pairs'   : len(labels),
        'sign_acc'  : sign_acc,  'sign_thr' : sign_thr,
        'amp_acc'   : amp_acc,
        'chain_acc' : chain_acc,
        'cos_acc'   : cos_acc,   'cos_thr'  : cos_thr,
        'v1_acc'    : v1_acc,
        'v2_acc'    : v2_acc,
        'auc'       : auc,
    }


# ============================================================
# PARSE PAIRS
# ============================================================
def parse_ann(ann_file, img_dir):
    pairs = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                pairs.append((os.path.join(img_dir, parts[1]),
                               os.path.join(img_dir, parts[2]),
                               int(parts[0])))
    return pairs


# ============================================================
# EMBEDDING EXTRACTORS
# ============================================================
def load_img_tensor(path, size=112):
    import torch
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def get_facenet_embs(paths, device, batch_size=128):
    import torch
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    embs = []
    for i in range(0, len(paths), batch_size):
        batch = torch.cat([load_img_tensor(p, size=160) for p in paths[i:i+batch_size]]).to(device)
        with torch.no_grad():
            embs.extend(model(batch).cpu().numpy())
        if (i // batch_size + 1) % 10 == 0:
            print(f"    FaceNet {i+batch_size}/{len(paths)}...")
    del model; torch.cuda.empty_cache()
    return np.array(embs)

def get_arcface_embs(paths, device, batch_size=128):
    import torch
    try:
        import subprocess
        subprocess.run(["pip", "install", "insightface", "onnxruntime-gpu", "-q"], check=True)
        import insightface
        from insightface.model_zoo import get_model
        # Load ArcFace buffalo_l recognition model
        import insightface.app
        app = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(112, 112))

        embs = []
        for idx, p in enumerate(paths):
            img_np = np.array(Image.open(p).convert('RGB'))
            faces = app.get(img_np)
            if faces:
                embs.append(faces[0].normed_embedding)
            else:
                # fallback: zero embedding kalau wajah tidak terdeteksi
                embs.append(np.zeros(512))
            if (idx + 1) % 1000 == 0:
                print(f"    ArcFace {idx+1}/{len(paths)}...")

        del app; torch.cuda.empty_cache()
        return np.array(embs)
    except Exception as e:
        print(f"  ArcFace error: {e}")
        return None


# ============================================================
# MAIN
# ============================================================
def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Gunakan semua dataset
    val_dir  = os.path.join(EXTRACT_DIR, 'val')
    datasets = {
        'lfw'     : 'lfw_ann.txt',
        'agedb_30': 'agedb_30_ann.txt',
        'calfw'   : 'calfw_ann.txt',
        'cplfw'   : 'cplfw_ann.txt',
    }

    # Validasi path
    print("\nValidasi path...")
    for ds_name, ann_file in datasets.items():
        ann_path = os.path.join(val_dir, ann_file)
        pairs = parse_ann(ann_path, val_dir)
        p1, p2, _ = pairs[0]
        ok = "✓" if os.path.exists(p1) and os.path.exists(p2) else "✗"
        print(f"  {ok} {ds_name:<12}: {len(pairs)} pairs")

    all_results = {}

    for emb_name, get_embs_fn in [
        ("FaceNet",  lambda paths: get_facenet_embs(paths, device)),
        ("ArcFace",  lambda paths: get_arcface_embs(paths, device)),
    ]:
        print(f"\n{'='*60}")
        print(f"Embedding: {emb_name}")
        print(f"{'='*60}")
        ds_results = []

        for ds_name, ann_file in datasets.items():
            ann_path = os.path.join(val_dir, ann_file)
            pairs    = parse_ann(ann_path, val_dir)
            p1_list  = [p for p,_,_ in pairs]
            p2_list  = [p for _,p,_ in pairs]
            labels   = [l for _,_,l in pairs]

            print(f"\nDataset: {ds_name} ({len(pairs)} pairs)")
            print(f"  Extracting embeddings...")

            # Extract semua path unik sekaligus (efisien)
            all_paths = list(set(p1_list + p2_list))
            emb_map   = {}
            all_embs  = get_embs_fn(all_paths)
            if all_embs is None:
                print(f"  [SKIP] {emb_name} tidak tersedia")
                break
            for path, emb in zip(all_paths, all_embs):
                emb_map[path] = emb

            embs1 = np.array([emb_map[p] for p in p1_list])
            embs2 = np.array([emb_map[p] for p in p2_list])

            result = eval_config(embs1, embs2, labels, f"{emb_name}_{ds_name}")
            ds_results.append(result)

        all_results[emb_name] = ds_results

    # ── Ringkasan ──────────────────────────────────────
    print(f"\n{'='*90}")
    print("RINGKASAN — FaceNet & ArcFace + IMG Sign / AMP / Chain / Voting")
    print(f"{'='*90}")
    header = f"{'Embedding':<18} {'Dataset':<12} {'Sign':>8} {'AMP':>8} {'Chain':>8} {'Cosine':>8} {'V1/3':>7} {'V2/3':>7}"
    print(header)
    print("─"*82)

    lines = [header + "\n" + "─"*82]
    for emb_name, ds_results in all_results.items():
        for r in ds_results:
            ds = r['name'].replace(f"{emb_name}_", "")
            line = (f"{emb_name:<18} {ds:<12} "
                    f"{r['sign_acc']*100:>7.2f}% "
                    f"{r['amp_acc']*100:>7.2f}% "
                    f"{r['chain_acc']*100:>7.2f}% "
                    f"{r['cos_acc']*100:>7.2f}% "
                    f"{r['v1_acc']*100:>6.2f}% "
                    f"{r['v2_acc']*100:>6.2f}%")
            print(line)
            lines.append(line)
        print()
        lines.append("")

    print(f"{'='*90}")

    # Simpan
    with open(RESULT_PATH, 'w') as f:
        f.write("FaceNet & ArcFace — IMG Sign / AMP / Chain / Cosine / Voting\n\n")
        f.write("\n".join(lines))
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
