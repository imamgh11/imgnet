# train_sw357_conv10_imgsign_a100.py
# SW357 + Conv10 (1SW+10Conv) — IMG Sign Score MSE loss
# TIDAK ada AMP/amplitude — murni sign pattern matching
# Loss: MSE (same→1.0, diff→0.0) via IMG Sign score

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import torchvision.transforms as T

# ── PATH CONFIG ────────────────────────────────────
DATA_ROOT  = "/content/data/casia-webface"
CKPT_ROOT  = "/content/drive/MyDrive/dataset/checkpoints_sw357_conv10_imgsign"

# ── HYPERPARAMS ────────────────────────────────────
BATCH_SIZE    = 16
LR            = 1e-4
MAX_PAIRS     = 300
NUM_WORKERS   = 8
WINDOW_SIZE   = 11
THRESHOLD     = 8
EMB_DIM       = 1024
NUM_EPOCHS    = 50
WARMUP_EPOCHS = 5


# ============================================================
# SW BLOCK
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


# ============================================================
# IMGNET — SW357 + Conv10 (1SW+10Conv, 10.58MB)
# Resolusi: 112→56→56→28→28→28→14→14→7→7
# ============================================================
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

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# IMG SIGN SCORE — murni sign pattern, tanpa amplitude
#
# soft_match = tanh(β × E1 × E2) → soft sign agreement per dim
# gate       = sigmoid(50 × (soft_match_sum - threshold + 0.5))
# img_sign   = mean(gate) over all windows
#
# Tidak ada rel_sim, tidak ada amplitude comparison
# ============================================================
def img_sign_score(E1, E2, beta=10.0):
    kernel     = torch.ones(1, 1, WINDOW_SIZE, device=E1.device)
    agreement  = (torch.tanh(beta * E1 * E2) + 1) / 2
    soft_match = F.conv1d(agreement.unsqueeze(1), kernel, stride=1).squeeze(1)
    gate       = torch.sigmoid(50.0 * (soft_match - THRESHOLD + 0.5))
    return gate.mean(dim=1)   # mean over windows


# ============================================================
# MSE LOSS — same→1.0, diff→0.0
# ============================================================
def contrastive_loss(E1_s, E2_s, E1_d, E2_d):
    device = E1_s.device if E1_s.shape[0] > 0 else E1_d.device
    ls = ld = torch.tensor(0.0, device=device)
    if E1_s.shape[0] > 0:
        ls = ((1.0 - img_sign_score(E1_s, E2_s)) ** 2).mean()
    if E1_d.shape[0] > 0:
        ld = (img_sign_score(E1_d, E2_d) ** 2).mean()
    return ls + ld, ls.item(), ld.item()


# ============================================================
# DATASET (tanpa MTCNN)
# ============================================================
class PairDataset(Dataset):
    def __init__(self, root_dir, img_size=112, max_pairs_per_identity=300, augment=False):
        self.img_size = img_size
        self.augment  = augment
        print(f"Loading dataset from: {root_dir}")
        identities = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
        self.identity_images = {}
        for idx, identity in enumerate(identities):
            path   = os.path.join(root_dir, identity)
            images = [os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if len(images) >= 2:
                self.identity_images[identity] = images
            if (idx + 1) % 1000 == 0:
                print(f"  scanning... {idx+1}/{len(identities)}")
        self.identity_list = list(self.identity_images.keys())
        self.pos_pairs = []
        for identity, images in self.identity_images.items():
            n = min(max_pairs_per_identity, len(images))
            for _ in range(n):
                i, j = random.sample(range(len(images)), 2)
                self.pos_pairs.append((images[i], images[j]))
        self.n_neg = len(self.pos_pairs)
        print(f"Identities : {len(self.identity_list)}")
        print(f"Pos pairs  : {len(self.pos_pairs)}")
        print(f"Total      : {len(self)}")

    def __len__(self):
        return len(self.pos_pairs) + self.n_neg

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t   = torch.from_numpy(arr).permute(2, 0, 1)
        if self.augment:
            aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.RandomRotation(degrees=10),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
            t = aug(t)
        return t

    def _random_negative(self):
        id1, id2 = random.sample(self.identity_list, 2)
        return random.choice(self.identity_images[id1]), random.choice(self.identity_images[id2])

    def __getitem__(self, idx):
        if idx < len(self.pos_pairs):
            p1, p2 = self.pos_pairs[idx]
            return self._load(p1), self._load(p2), torch.tensor(1)
        p1, p2 = self._random_negative()
        return self._load(p1), self._load(p2), torch.tensor(0)


# ============================================================
# TRAINING LOOP
# ============================================================
def train(model, train_loader, val_loader, device, name):
    ckpt_dir    = os.path.join(CKPT_ROOT, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, "last_checkpoint.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (ep + 1) / WARMUP_EPOCHS if ep < WARMUP_EPOCHS else 1.0)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    start_epoch = 0
    best_val    = float('inf')
    if os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            best_val    = ckpt.get('best_val', float('inf'))
            print(f"  [{name}] Resumed from epoch {start_epoch}")
        except RuntimeError:
            print(f"  [{name}] Checkpoint tidak kompatibel, training dari awal")
    else:
        print(f"  [{name}] Training dari awal...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        t_loss = t_s = t_d = 0.0; n = 0

        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            img1=img1.to(device); img2=img2.to(device); labels=labels.to(device)
            optimizer.zero_grad()
            E1, E2 = model(img1), model(img2)
            sm, dm = labels == 1, labels == 0
            loss, ls, ld = contrastive_loss(E1[sm], E2[sm], E1[dm], E2[dm])
            if loss.item() > 0:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            t_loss += loss.item(); t_s += ls; t_d += ld; n += 1

            if batch_idx == 0:
                print(f"  [{name}] Epoch {epoch+1} dimulai...")
            if (batch_idx + 1) % 100 == 0:
                with torch.no_grad():
                    s_mean = img_sign_score(E1[sm], E2[sm]).mean().item() if sm.sum() > 0 else 0.0
                    d_mean = img_sign_score(E1[dm], E2[dm]).mean().item() if dm.sum() > 0 else 0.0
                print(f"  [{name}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f} | sign same={s_mean:.3f} diff={d_mean:.3f}")

        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step(); current_lr = warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step(); current_lr = cosine_scheduler.get_last_lr()[0]

        model.eval()
        v_loss = 0.0; nv = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1=img1.to(device); img2=img2.to(device); labels=labels.to(device)
                E1,E2=model(img1),model(img2); sm=labels==1; dm=labels==0
                loss,_,_=contrastive_loss(E1[sm],E2[sm],E1[dm],E2[dm])
                v_loss+=loss.item(); nv+=1
        avg_v = v_loss / max(nv, 1)

        print(f"  [{name}] Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train {t_loss/n:.4f} (same={t_s/n:.4f} diff={t_d/n:.4f}) | "
              f"Val {avg_v:.4f} | LR {current_lr:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            best_path = os.path.join(ckpt_dir, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  [{name}]   -> best saved: best_model_epoch{epoch+1}.pth (val={best_val:.4f})")

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': avg_v, 'best_val': best_val,
        }, resume_path)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_model.pth"))
    print(f"  [{name}] Training selesai!")


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(CKPT_ROOT, exist_ok=True)

    dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nLoading dataset...")
    train_dataset = PairDataset(DATA_ROOT, max_pairs_per_identity=MAX_PAIRS, augment=True)
    val_dataset   = PairDataset(DATA_ROOT, max_pairs_per_identity=MAX_PAIRS, augment=False)
    total   = len(train_dataset)
    indices = list(range(total))
    random.seed(42); random.shuffle(indices)
    val_size = int(total * 0.1)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    pin = (device.type == "cuda")
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(Subset(val_dataset,   val_idx),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Batch/epoch: {len(train_idx)//BATCH_SIZE}")

    name  = "SW357_conv10_imgsign"
    model = IMGNet(emb_dim=EMB_DIM).to(device)
    print(f"Parameters : {model.n_params():,} (~{model.n_params()*4/1024/1024:.2f} MB)")
    print(f"Loss       : IMG Sign MSE (same→1.0, diff→0.0) — tanpa amplitude")
    print(f"Checkpoint : {CKPT_ROOT}")

    train(model, train_loader, val_loader, device, name)


if __name__ == "__main__":
    main()
