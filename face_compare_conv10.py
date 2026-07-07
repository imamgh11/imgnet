# face_compare_conv10.py - IMGNet Conv10 + AMP IMG Score + Chain Score
import tkinter as tk
from tkinter import filedialog, Label, Frame, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN

# ── Config ────────────────────────────────────────
CKPT_PATH   = r"C:\yourpath.pth"
WINDOW_SIZE = 11
THRESHOLD   = 8
EMB_DIM     = 1024
DEBUG       = True

# Chain Score params
NEUTRAL_LEN  = 29
REWARD_RATE  = 0.3
PUNISH_RATE  = 1.0


# ── SW Block ──────────────────────────────────────
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


# ── IMGNet Conv10 (1SW+10Conv) ────────────────────
# Resolusi: 112→56→56→28→28→28→14→14→7→7
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


# ── AMP IMG Score ─────────────────────────────────
def compute_amp_score(emb1, emb2, debug=False):
    n = len(emb1) - WINDOW_SIZE + 1
    total_sim = 0.0
    match_win = 0
    for i in range(n):
        w1, w2 = emb1[i:i+WINDOW_SIZE], emb2[i:i+WINDOW_SIZE]
        s1 = np.where(w1 >= 0, 1, -1).astype(np.int8)
        s2 = np.where(w2 >= 0, 1, -1).astype(np.int8)
        mc = int(np.sum(s1 == s2))
        if mc >= THRESHOLD:
            match_win += 1
            amp1    = np.mean(np.abs(w1))
            amp2    = np.mean(np.abs(w2))
            denom   = max(amp1, amp2, 1e-6)
            rel_sim = max(0.0, 1 - abs(amp1 - amp2) / denom)
            total_sim += rel_sim
            if debug:
                print(f"  Window {i:4d}: sign={mc}/11  amp1={amp1:.4f} amp2={amp2:.4f}  rel_sim={rel_sim:.4f}")
    score = total_sim / n
    if debug:
        print(f"\n  Total windows : {n}")
        print(f"  Match windows : {match_win}/{n} ({match_win/n*100:.1f}%)")
        print(f"  AMP Score     : {score:.4f}")
    return score, match_win, n


# ── IMG Sign Score ────────────────────────────────
def compute_sign_score(emb1, emb2):
    s1 = np.where(emb1 >= 0, 1, -1).astype(np.int8)
    s2 = np.where(emb2 >= 0, 1, -1).astype(np.int8)
    n  = len(s1) - WINDOW_SIZE + 1
    mc = 0
    for i in range(n):
        if int(np.sum(s1[i:i+WINDOW_SIZE] == s2[i:i+WINDOW_SIZE])) >= THRESHOLD:
            mc += 1
    return mc / n


# ── Chain Score ───────────────────────────────────
def compute_chain_score(emb1, emb2, debug=False):
    n = len(emb1) - WINDOW_SIZE + 1
    if n <= 0:
        return 0.0, 0, 0.0, 0.0

    match_flags = []
    for i in range(n):
        s1 = np.where(emb1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        s2 = np.where(emb2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
        match_flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)

    total_match = sum(match_flags)
    img_sign    = total_match / n

    # Hitung chain
    n_chains = 0
    in_chain = False
    chain_lens = []
    cur_len = 0
    for a in match_flags:
        if a:
            if not in_chain:
                n_chains += 1; in_chain = True
            cur_len += 1
        else:
            if in_chain:
                chain_lens.append(cur_len)
                cur_len = 0
            in_chain = False
    if in_chain and cur_len > 0:
        chain_lens.append(cur_len)

    if n_chains == 0 or total_match == 0:
        return 0.0, 0, 0.0, img_sign

    avg_chain = total_match / n_chains
    diff      = avg_chain - NEUTRAL_LEN
    base      = img_sign * 100
    score     = base + (REWARD_RATE * diff if diff >= 0 else PUNISH_RATE * diff)
    score     = float(np.clip(score, 0, 100))

    if debug:
        print(f"\n  Chain Score Analysis:")
        print(f"  IMG Sign (match ratio) : {img_sign:.4f} ({total_match}/{n})")
        print(f"  N chains               : {n_chains}")
        print(f"  Avg chain length       : {avg_chain:.2f}  (neutral={NEUTRAL_LEN})")
        print(f"  Chain lengths          : {chain_lens[:10]}{'...' if len(chain_lens)>10 else ''}")
        print(f"  Chain Score            : {score:.2f}/100")

    return score, n_chains, avg_chain, img_sign


# ── App ───────────────────────────────────────────
class FaceCompareIMGNet:
    def __init__(self, master):
        self.master = master
        master.title("IMGNet Conv10 — AMP IMG Score + Chain Score")
        master.geometry("1080x880")
        master.configure(bg="#0a0e1a")

        self.img1_path = None
        self.img2_path = None
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading IMGNet Conv10 from {CKPT_PATH}...")
        self.model = IMGNet(emb_dim=EMB_DIM).to(self.device)
        state = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        self.model.load_state_dict(state)
        self.model.eval()
        print("IMGNet Conv10 loaded.")

        self.mtcnn = MTCNN(image_size=112, keep_all=False,
                           device=self.device, post_process=False)
        self._build_ui()

    def _build_ui(self):
        BG     = "#0a0e1a"
        CARD   = "#111827"
        BORDER = "#1e293b"
        BLUE   = "#6366f1"
        GREEN  = "#10b981"
        ORANGE = "#f59e0b"
        PURPLE = "#a855f7"
        TEAL   = "#14b8a6"
        SUB    = "#64748b"

        main = Frame(self.master, bg=BG)
        main.pack(expand=True, fill="both", padx=24, pady=18)

        Label(main, text="IMGNet Conv10  ·  1SW+10Conv  ·  AMP IMG + Chain Score",
              font=("Courier", 14, "bold"), bg=BG, fg="#e2e8f0").pack(pady=(0, 2))
        Label(main, text=f"SW windows: 3·5·7  ·  emb {EMB_DIM}D  ·  w={WINDOW_SIZE}  t={THRESHOLD}/11  ·  10.58MB",
              font=("Courier", 9), bg=BG, fg=SUB).pack(pady=(0, 14))

        img_row = Frame(main, bg=BG)
        img_row.pack(fill="both", expand=True)
        self.img1_lbl = self._image_panel(img_row, "IMAGE 1", BLUE,  self.load_img1, "left")
        self.img2_lbl = self._image_panel(img_row, "IMAGE 2", GREEN, self.load_img2, "right")

        tk.Button(main, text="▶  COMPARE", command=self.compare,
                  bg=ORANGE, fg="#0a0e1a", font=("Courier", 13, "bold"),
                  padx=40, pady=10, relief="flat", cursor="hand2").pack(pady=10)

        res = Frame(main, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        res.pack(fill="both", expand=True, padx=4, pady=(0, 8))

        Label(res, text="COMPARISON RESULT",
              font=("Courier", 10, "bold"), bg=CARD, fg=SUB).pack(pady=(12, 6))

        row = Frame(res, bg=CARD)
        row.pack(fill="x", padx=20, pady=(0, 8))
        self.lbl_amp    = self._metric_box(row, "AMP IMG",    GREEN)
        self.lbl_chain  = self._metric_box(row, "CHAIN SCORE", TEAL)
        self.lbl_sign   = self._metric_box(row, "IMG SIGN",   PURPLE)
        self.lbl_match  = self._metric_box(row, "MATCH WIN",  BLUE)
        self.lbl_cosine = self._metric_box(row, "COSINE SIM", ORANGE)

        self.verdict_label = Label(res,
            text="pilih dua gambar dan klik compare",
            font=("Courier", 11), bg=CARD, fg=SUB)
        self.verdict_label.pack(pady=(0, 6))

        txt_frame = Frame(res, bg=CARD)
        txt_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = tk.Scrollbar(txt_frame)
        sb.pack(side="right", fill="y")
        self.detail_text = tk.Text(txt_frame, height=8, bg="#0a0e1a", fg="#c9d1d9",
                                    font=("Courier", 9), relief="flat",
                                    wrap="none", bd=0, yscrollcommand=sb.set,
                                    highlightthickness=1, highlightbackground=BORDER)
        self.detail_text.pack(side="left", fill="both", expand=True)
        sb.config(command=self.detail_text.yview)

        self.status_label = Label(main,
            text=f"IMGNet Conv10 loaded  ·  device={self.device}  ·  ready",
            font=("Courier", 9), bg=BG, fg=SUB)
        self.status_label.pack(side="bottom", pady=6)

    def _image_panel(self, parent, title, color, cmd, side):
        card = Frame(parent, bg="#111827", highlightthickness=1, highlightbackground="#1e293b")
        card.pack(side=side, padx=8, pady=4, fill="both", expand=True)
        Label(card, text=title, font=("Courier", 11, "bold"),
              bg="#111827", fg=color).pack(pady=(12, 6))
        lbl = Label(card, bg="#111827", text="no image", fg="#64748b", font=("Courier", 8))
        lbl.pack(pady=(0, 10))
        tk.Button(card, text=f"select {title.lower()}", command=cmd,
                  bg=color, fg="#0a0e1a", font=("Courier", 9, "bold"),
                  padx=14, pady=5, relief="flat", cursor="hand2").pack(pady=(0, 14))
        return lbl

    def _metric_box(self, parent, label, color):
        box = Frame(parent, bg="#0a0e1a", highlightthickness=1, highlightbackground="#1e293b")
        box.pack(side="left", expand=True, fill="both", padx=6, pady=4)
        Label(box, text=label, font=("Courier", 8, "bold"),
              bg="#0a0e1a", fg=color).pack(pady=(10, 2))
        val = Label(box, text="—", font=("Courier", 18, "bold"),
                    bg="#0a0e1a", fg=color)
        val.pack(pady=(0, 10))
        return val

    def get_embedding(self, img_path):
        img  = Image.open(img_path).convert('RGB')
        boxes, _ = self.mtcnn.detect(img)
        face = self.mtcnn(img)
        if face is not None and boxes is not None:
            print(f"  MTCNN: wajah terdeteksi ✓ ({img_path.split('/')[-1].split(chr(92))[-1]})")
            arr = face.cpu().numpy().astype(np.float32) / 255.0
            t   = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            # simpan boxes untuk display
            self._last_boxes = boxes
        else:
            print(f"  MTCNN: wajah TIDAK terdeteksi, pakai resize biasa ✗ ({img_path.split('/')[-1].split(chr(92))[-1]})")
            img = img.resize((112, 112), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self._last_boxes = None
        with torch.no_grad():
            return self.model(t).squeeze(0).cpu().numpy()

    def _display(self, path, lbl):
        img = Image.open(path).convert('RGB')
        # Deteksi wajah untuk bounding box
        try:
            boxes, probs = self.mtcnn.detect(img)
            if boxes is not None:
                import PIL.ImageDraw as ImageDraw
                draw = ImageDraw.Draw(img)
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    draw.rectangle([x1, y1, x2, y2], outline="#10b981", width=3)
        except:
            pass
        img.thumbnail((200, 160), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl.config(image=tk_img, text="")
        lbl.image = tk_img

    def load_img1(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img1_path = path
            self._display(path, self.img1_lbl)
            self.status_label.config(text="image 1 loaded")

    def load_img2(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img2_path = path
            self._display(path, self.img2_lbl)
            self.status_label.config(text="image 2 loaded")

    def compare(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Warning", "Pilih dua gambar dulu!")
            return
        try:
            self.status_label.config(text="computing embedding...")
            self.master.update()

            emb1 = self.get_embedding(self.img1_path)
            emb2 = self.get_embedding(self.img2_path)

            print("\n" + "="*60)
            print("DEBUG: IMG Sign + AMP + Chain Score Analysis")
            print("="*60)

            amp_score, match_win, n_win = compute_amp_score(emb1, emb2, debug=DEBUG)
            chain_score, n_chains, avg_chain, img_sign = compute_chain_score(emb1, emb2, debug=DEBUG)
            sign_score = compute_sign_score(emb1, emb2)
            cosine     = float(np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

            # Log blok yang match (chain analysis)
            n = len(emb1) - WINDOW_SIZE + 1
            match_flags = []
            for i in range(n):
                s1 = np.where(emb1[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
                s2 = np.where(emb2[i:i+WINDOW_SIZE] >= 0, 1, -1).astype(np.int8)
                match_flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)

            # Ekstrak chain blocks
            chains = []
            in_chain = False; start = 0; cur_len = 0
            for i, a in enumerate(match_flags):
                if a:
                    if not in_chain: start = i; in_chain = True
                    cur_len += 1
                else:
                    if in_chain:
                        chains.append((start, start + cur_len - 1, cur_len))
                        cur_len = 0
                    in_chain = False
            if in_chain: chains.append((start, start + cur_len - 1, cur_len))

            self.lbl_amp.config(text=f"{amp_score:.4f}")
            self.lbl_chain.config(text=f"{chain_score:.1f}")
            self.lbl_sign.config(text=f"{sign_score:.4f}")
            self.lbl_match.config(text=f"{match_win}/{n_win}")
            self.lbl_cosine.config(text=f"{cosine:.4f}")

            # Threshold masing-masing metric
            THR_SIGN  = 0.79
            THR_AMP   = 0.79
            THR_CHAIN = 82.0   # dari eval epoch 29

            sign_pass  = sign_score  >= THR_SIGN
            amp_pass   = amp_score   >= THR_AMP
            chain_pass = chain_score >= THR_CHAIN

            n_pass = sum([sign_pass, amp_pass, chain_pass])

            if n_pass >= 2:
                verdict = "✅ MATCH"
                color   = "#10b981"
            elif n_pass == 1:
                verdict = "⚠️  RAGU-RAGU"
                color   = "#f59e0b"
            else:
                verdict = "❌ DIFFERENT"
                color   = "#ef4444"

            self.verdict_label.config(
                text=f"{verdict}   ·   Sign: {sign_score:.4f} {'✓' if sign_pass else '✗'}   "
                     f"AMP: {amp_score:.4f} {'✓' if amp_pass else '✗'}   "
                     f"Chain: {chain_score:.1f} {'✓' if chain_pass else '✗'}",
                fg=color, font=("Courier", 11, "bold"))

            self.detail_text.delete(1.0, tk.END)
            d  = f"Backbone      : IMGNet Conv10 (1SW+10Conv, 10.58MB)\n"
            d += f"Embedding     : {EMB_DIM}D  ·  Device: {self.device}\n\n"
            d += f"── Verdict: {verdict} ({n_pass}/3 metric lolos) ──────\n"
            d += f"IMG Sign : {sign_score:.4f}  thr={THR_SIGN}  {'✓ LOLOS' if sign_pass else '✗ TIDAK'}\n"
            d += f"AMP IMG  : {amp_score:.4f}  thr={THR_AMP}  {'✓ LOLOS' if amp_pass else '✗ TIDAK'}\n"
            d += f"Chain    : {chain_score:.1f}/100  thr={THR_CHAIN}  {'✓ LOLOS' if chain_pass else '✗ TIDAK'}\n"
            d += f"Cosine   : {cosine:.4f}\n\n"
            d += f"── Chain Detail ───────────────────────────\n"
            d += f"N chains      : {n_chains}\n"
            d += f"Avg chain len : {avg_chain:.2f}  (neutral={NEUTRAL_LEN})\n"
            d += f"Match windows : {match_win}/{n_win} ({match_win/n_win*100:.1f}%)\n\n"
            d += f"── Match Blocks ───────────────────────────\n"
            if chains:
                for idx, (s, e, l) in enumerate(chains[:15]):
                    d += f"  Chain {idx+1:2d}: window {s:4d}-{e:4d}  len={l}\n"
                if len(chains) > 15:
                    d += f"  ... dan {len(chains)-15} chain lainnya\n"
            else:
                d += "  Tidak ada chain yang terdeteksi\n"
            self.detail_text.insert(tk.END, d)

            print(f"\n{'='*60}")
            print(f"Verdict     : {verdict} ({n_pass}/3 lolos)")
            print(f"IMG Sign    : {sign_score:.4f}  {'✓' if sign_pass else '✗'} (thr={THR_SIGN})")
            print(f"AMP Score   : {amp_score:.4f}  {'✓' if amp_pass else '✗'} (thr={THR_AMP})")
            print(f"Chain Score : {chain_score:.1f}  {'✓' if chain_pass else '✗'} (thr={THR_CHAIN})")
            print(f"Cosine      : {cosine:.4f}")
            print(f"\nMatch Blocks ({len(chains)} chains):")
            for idx, (s, e, l) in enumerate(chains[:10]):
                print(f"  Chain {idx+1:2d}: window {s:4d}-{e:4d}  len={l}")
            if len(chains) > 10:
                print(f"  ... dan {len(chains)-10} chain lainnya")
            print(f"{'='*60}\n")

            self.status_label.config(text="comparison complete")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=f"error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    root = tk.Tk()
    app  = FaceCompareIMGNet(root)
    root.mainloop()
