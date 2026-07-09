# imgnet_visualizer.py
# IMGNet Interactive Visualizer — tkinter
# Panel Kiri  : Upload 2 foto → grid 112×112 + animasi SW Block scan
# Panel Tengah: Sliding window embedding analysis (Training vs Metric mode)
# Panel Kanan : Conv2-10 feature maps + Score results

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import math
import threading
import time

# ── Try import torch (optional — fallback ke dummy) ──
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Try import MTCNN ──────────────────────────────
try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(image_size=112, keep_all=False, post_process=False,
                   device="cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu")
    MTCNN_OK = True
except Exception:
    _mtcnn = None
    MTCNN_OK = False

# ── IMGNet Conv10 Model ───────────────────────────
CKPT_PATH = r"C:\PythonProj\img_bnn\checkpoints_sw357_conv10_imgsign\SW357_conv10_imgsign\best_model_epoch39_plateau.pth"
EMB_DIM   = 1024  # Conv10 embedding dim

_imgnet_model  = None
_imgnet_device = "cpu"

if TORCH_OK:
    class _SWBlock(nn.Module):
        def __init__(self):
            super().__init__()
            n_diff = (8 + 24 + 48) * 3  # 240
            self.fc = nn.Sequential(nn.Linear(240, 64), nn.ReLU(inplace=True), nn.Linear(64, 32))
        def forward(self, x):
            B, C, H, W = x.shape
            diffs = []
            for ws in [3, 5, 7]:
                pad   = ws // 2
                x_pad = F.pad(x, [pad,pad,pad,pad], mode='reflect')
                patches = x_pad.unfold(2,ws,1).unfold(3,ws,1)
                diff = x.unsqueeze(-1).unsqueeze(-1) - patches
                mid  = ws // 2
                mask = torch.ones(ws, ws, dtype=torch.bool, device=x.device)
                mask[mid,mid] = False
                diffs.append(diff[:,:,:,:,mask])
            diffs = torch.cat(diffs, -1)
            B,C,H,W,N = diffs.shape
            out = self.fc(diffs.permute(0,2,3,1,4).reshape(B*H*W, C*N))
            return out.reshape(B,H,W,-1).permute(0,3,1,2)

    class _IMGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.sw1    = _SWBlock(); self.bn1 = nn.BatchNorm2d(32)
            self.conv2  = nn.Conv2d(32,  64,  3,stride=1,padding=1,bias=False); self.bn2  = nn.BatchNorm2d(64)
            self.conv3  = nn.Conv2d(64,  64,  3,stride=2,padding=1,bias=False); self.bn3  = nn.BatchNorm2d(64)
            self.conv4  = nn.Conv2d(64,  128, 3,stride=1,padding=1,bias=False); self.bn4  = nn.BatchNorm2d(128)
            self.conv5  = nn.Conv2d(128, 128, 3,stride=1,padding=1,bias=False); self.bn5  = nn.BatchNorm2d(128)
            self.conv6  = nn.Conv2d(128, 128, 3,stride=2,padding=1,bias=False); self.bn6  = nn.BatchNorm2d(128)
            self.conv7  = nn.Conv2d(128, 256, 3,stride=1,padding=1,bias=False); self.bn7  = nn.BatchNorm2d(256)
            self.conv8  = nn.Conv2d(256, 256, 3,stride=1,padding=1,bias=False); self.bn8  = nn.BatchNorm2d(256)
            self.conv9  = nn.Conv2d(256, 256, 3,stride=2,padding=1,bias=False); self.bn9  = nn.BatchNorm2d(256)
            self.conv10 = nn.Conv2d(256, 256, 3,stride=1,padding=1,bias=False); self.bn10 = nn.BatchNorm2d(256)
            self.gap    = nn.AdaptiveAvgPool2d(1)
            self.fc     = nn.Linear(256, 1024)
            self.bn     = nn.BatchNorm1d(1024)
        def forward(self, x):
            x = F.relu(self.bn1(self.sw1(x)))
            x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x))); x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x))); x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x))); x = F.relu(self.bn9(self.conv9(x)))
            x = F.relu(self.bn10(self.conv10(x)))
            x = self.gap(x).view(x.size(0), -1)
            return self.bn(self.fc(x))

    import os
    if os.path.exists(CKPT_PATH):
        try:
            _imgnet_device = "cuda" if torch.cuda.is_available() else "cpu"
            _imgnet_model  = _IMGNet().to(_imgnet_device)
            state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            _imgnet_model.load_state_dict(state)
            _imgnet_model.eval()

            # Register forward hooks untuk capture feature maps
            _feature_maps = {}
            def _make_hook(name):
                def hook(module, inp, out):
                    _feature_maps[name] = out.detach().cpu()
                return hook

            _imgnet_model.sw1.register_forward_hook(_make_hook("sw1"))
            for i in range(2, 11):
                getattr(_imgnet_model, f"conv{i}").register_forward_hook(_make_hook(f"conv{i}"))
            _imgnet_model.gap.register_forward_hook(_make_hook("gap"))

            print(f"✓ IMGNet loaded from {CKPT_PATH}")
        except Exception as e:
            print(f"✗ IMGNet load failed: {e}")
            _imgnet_model = None
    else:
        print(f"✗ Checkpoint tidak ditemukan: {CKPT_PATH}")
else:
    _feature_maps = {}

# ── COLORS ────────────────────────────────────────
BG        = "#0a0e1a"
CARD      = "#111827"
BORDER    = "#1e293b"
BLUE      = "#6366f1"
GREEN     = "#10b981"
ORANGE    = "#f59e0b"
PURPLE    = "#a855f7"
TEAL      = "#14b8a6"
RED       = "#ef4444"
SUB       = "#64748b"
TEXT      = "#e2e8f0"
WHITE     = "#ffffff"
YELLOW    = "#fbbf24"

# ── CONFIG ────────────────────────────────────────
WINDOW_SIZE = 11
THRESHOLD   = 8
EMB_DIM     = 64   # reduced for speed in demo
IMG_SIZE    = 112
BETA        = 10.0


# ============================================================
# DUMMY MODEL (kalau torch tidak ada)
# ============================================================
def dummy_embed(img_array):
    """Generate pseudo-embedding (fallback kalau model tidak ada)"""
    flat = img_array.flatten().astype(np.float32) / 255.0
    np.random.seed(int(flat.sum() * 1000) % 2**31)
    emb = np.random.randn(EMB_DIM).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)

def get_embedding(img_array):
    """Get real IMGNet embedding, fallback ke dummy"""
    if _imgnet_model is not None and TORCH_OK:
        try:
            arr = img_array.astype(np.float32) / 255.0
            t   = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(_imgnet_device)
            with torch.no_grad():
                emb = _imgnet_model(t).squeeze(0).cpu().numpy()
            return emb
        except Exception as e:
            print(f"Embed error: {e}")
    return dummy_embed(img_array)


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def tanh_agreement(e1, e2, beta=BETA):
    return (np.tanh(beta * e1 * e2) + 1) / 2

def img_sign_score(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    scores = []
    for i in range(n):
        w1, w2 = e1[i:i+WINDOW_SIZE], e2[i:i+WINDOW_SIZE]
        s1 = np.where(w1 >= 0, 1, -1)
        s2 = np.where(w2 >= 0, 1, -1)
        mc = int(np.sum(s1 == s2))
        scores.append(mc / WINDOW_SIZE)
    return np.array(scores)

def chain_score(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    flags = []
    for i in range(n):
        s1 = np.where(e1[i:i+WINDOW_SIZE] >= 0, 1, -1)
        s2 = np.where(e2[i:i+WINDOW_SIZE] >= 0, 1, -1)
        flags.append(int(np.sum(s1 == s2)) >= THRESHOLD)
    total = sum(flags); img_s = total / max(n, 1)
    chains = 0; in_c = False
    for f in flags:
        if f and not in_c: chains += 1; in_c = True
        elif not f: in_c = False
    avg_c = total / max(chains, 1)
    diff = avg_c - 29
    score = img_s + (0.3 * diff if diff >= 0 else 1.0 * diff) / 100
    return float(np.clip(score, 0, 1)), chains, avg_c


# ============================================================
# SW BLOCK VISUALIZATION — compute scan result per window
# ============================================================
def sw_scan_result(img_array, window_size=3):
    """
    Scan 112×112 image with SW Block window
    Returns: heat map (H×W) of relational activity
    """
    img = img_array.astype(np.float32) / 255.0
    if len(img.shape) == 3:
        gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    else:
        gray = img
    h, w = gray.shape
    pad = window_size // 2
    padded = np.pad(gray, pad, mode='reflect')
    result = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            patch = padded[r:r+window_size, c:c+window_size]
            center = gray[r, c]
            diffs = patch.flatten()
            mid = len(diffs) // 2
            diffs = np.delete(diffs, mid)
            result[r, c] = np.mean(np.abs(diffs - center))
    return result


# ============================================================
# MAIN APP
# ============================================================
class IMGNetVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("IMGNet Interactive Visualizer")
        root.geometry("1400x900")
        root.configure(bg=BG)
        root.resizable(True, True)

        # State
        self.img1_array  = None
        self.img2_array  = None
        self.emb1        = None
        self.emb2        = None
        self.sw_window   = 3
        self.conv_layer  = 2
        self.win_pos     = 0
        self.mode        = tk.StringVar(value="metric")  # training / metric
        self.animating   = False
        self.sw_animating = False

        self._build_ui()

    # ────────────────────────────────────────────────────────
    # UI BUILD
    # ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=BG, height=50)
        top.pack(fill="x", padx=16, pady=(12,0))

        tk.Label(top,
            text="IMGNet  ·  Multi-Scale Sliding Window Face Verification  ·  Interactive Visualizer",
            font=("Courier", 13, "bold"), bg=BG, fg=TEXT).pack(side="left")

        model_status = "✓ epoch39" if _imgnet_model is not None else "✗ dummy"
        tk.Label(top,
            text=f"EMB {EMB_DIM}D  ·  SW {{3,5,7}}  ·  w={WINDOW_SIZE}  t={THRESHOLD}/11  ·  MTCNN={'✓' if MTCNN_OK else '✗'}  ·  IMGNet={model_status}",
            font=("Courier", 9), bg=BG, fg=SUB).pack(side="right")

        # Main 3-panel layout
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)
        main.grid_columnconfigure(0, weight=2)
        main.grid_columnconfigure(1, weight=3)
        main.grid_columnconfigure(2, weight=2)
        main.grid_rowconfigure(0, weight=1)

        self._build_left(main)
        self._build_center(main)
        self._build_right(main)

    # ── LEFT PANEL ───────────────────────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))

        tk.Label(left, text="INPUT IMAGES  ·  SW BLOCK SCAN",
                 font=("Courier", 10, "bold"), bg=CARD, fg=BLUE).pack(pady=(10,4))

        # Two image upload areas
        imgs = tk.Frame(left, bg=CARD)
        imgs.pack(fill="x", padx=8)
        imgs.grid_columnconfigure(0, weight=1)
        imgs.grid_columnconfigure(1, weight=1)

        self.img1_canvas = self._image_panel(imgs, "IMAGE 1", BLUE, self.load_img1, 0)
        self.img2_canvas = self._image_panel(imgs, "IMAGE 2", GREEN, self.load_img2, 1)

        # SW Block controls
        sw_ctrl = tk.Frame(left, bg=CARD)
        sw_ctrl.pack(fill="x", padx=8, pady=4)

        tk.Label(sw_ctrl, text="SW Window:", font=("Courier", 9), bg=CARD, fg=SUB).pack(side="left")
        for ws in [3, 5, 7]:
            tk.Button(sw_ctrl, text=f"{ws}×{ws}",
                     command=lambda w=ws: self._set_sw_window(w),
                     bg=CARD, fg=ORANGE, font=("Courier", 9, "bold"),
                     relief="flat", padx=6, pady=2,
                     cursor="hand2").pack(side="left", padx=2)

        tk.Button(sw_ctrl, text="▶ ANIMATE SW",
                  command=self.animate_sw,
                  bg=PURPLE, fg=WHITE, font=("Courier", 9, "bold"),
                  relief="flat", padx=10, pady=3,
                  cursor="hand2").pack(side="right", padx=4)

        # SW scan canvas (shows img1 with scanning window overlay)
        tk.Label(left, text="SW Block Scan — Image 1",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack()
        self.sw_canvas = tk.Canvas(left, width=224, height=224, bg="#050810",
                                    highlightthickness=1, highlightbackground=BORDER)
        self.sw_canvas.pack(pady=4)

        # SW heatmap
        tk.Label(left, text="Relational Activity Heatmap",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack()
        self.heat_canvas = tk.Canvas(left, width=224, height=112, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.heat_canvas.pack(pady=4)

        # Conv layer selector
        conv_ctrl = tk.Frame(left, bg=CARD)
        conv_ctrl.pack(fill="x", padx=8, pady=4)
        tk.Label(conv_ctrl, text="Conv Layer:", font=("Courier", 9), bg=CARD, fg=SUB).pack(side="left")
        self.conv_var = tk.IntVar(value=2)
        for i in range(2, 11):
            tk.Radiobutton(conv_ctrl, text=str(i), variable=self.conv_var, value=i,
                          bg=CARD, fg=TEAL, selectcolor=CARD,
                          font=("Courier", 8), command=self._update_conv).pack(side="left")

        # Conv feature map
        tk.Label(left, text="Conv Feature Map (simulated)",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack()
        self.conv_canvas = tk.Canvas(left, width=224, height=56, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.conv_canvas.pack(pady=(4, 4))

        # Ablation study button
        tk.Button(left, text="🔬 ABLATION STUDY",
                  command=self.open_ablation_window,
                  bg="#7c3aed", fg=WHITE, font=("Courier", 10, "bold"),
                  relief="flat", padx=16, pady=6, cursor="hand2").pack(pady=(4,8))

    def _image_panel(self, parent, title, color, cmd, col):
        f = tk.Frame(parent, bg=CARD)
        f.grid(row=0, column=col, padx=4, pady=4)
        tk.Label(f, text=title, font=("Courier", 9, "bold"), bg=CARD, fg=color).pack()
        canvas = tk.Canvas(f, width=104, height=104, bg="#050810",
                           highlightthickness=1, highlightbackground=BORDER)
        canvas.pack()
        tk.Button(f, text="Upload", command=cmd,
                  bg=color, fg=BG, font=("Courier", 8, "bold"),
                  relief="flat", padx=6, pady=2, cursor="hand2").pack(pady=3)
        return canvas

    # ── CENTER PANEL ─────────────────────────────────────────
    def _build_center(self, parent):
        center = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        center.grid(row=0, column=1, sticky="nsew", padx=6)

        tk.Label(center, text="SLIDING WINDOW EMBEDDING ANALYSIS",
                 font=("Courier", 10, "bold"), bg=CARD, fg=PURPLE).pack(pady=(10,4))

        # Mode selector
        mode_f = tk.Frame(center, bg=CARD)
        mode_f.pack()
        for val, label, col in [("metric","METRIC MODE",GREEN),("training","TRAINING MODE",ORANGE)]:
            tk.Radiobutton(mode_f, text=label, variable=self.mode, value=val,
                          bg=CARD, fg=col, selectcolor=CARD,
                          font=("Courier", 9, "bold"),
                          command=self._update_center).pack(side="left", padx=12)

        # Window position info
        self.win_info = tk.Label(center,
            text="Window: —  |  Position: —/—",
            font=("Courier", 9), bg=CARD, fg=SUB)
        self.win_info.pack()

        # Main embedding visualization canvas
        self.emb_canvas = tk.Canvas(center, width=560, height=180, bg="#050810",
                                     highlightthickness=1, highlightbackground=BORDER)
        self.emb_canvas.pack(padx=8, pady=4)

        # Window detail canvas (shows values in current window)
        self.win_canvas = tk.Canvas(center, width=560, height=140, bg="#050810",
                                     highlightthickness=1, highlightbackground=BORDER)
        self.win_canvas.pack(padx=8, pady=4)

        # tanh curve canvas (training mode)
        self.tanh_frame = tk.Frame(center, bg=CARD)
        self.tanh_frame.pack(fill="x", padx=8)
        tk.Label(self.tanh_frame, text="tanh(β·E1·E2) Agreement Curve  (β=10)",
                 font=("Courier", 8), bg=CARD, fg=ORANGE).pack()
        self.tanh_canvas = tk.Canvas(self.tanh_frame, width=560, height=120, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.tanh_canvas.pack()

        # Navigation controls
        nav = tk.Frame(center, bg=CARD)
        nav.pack(pady=6)

        tk.Button(nav, text="◀◀ FIRST", command=self._win_first,
                  bg=CARD, fg=SUB, font=("Courier", 9), relief="flat",
                  padx=8, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(nav, text="◀ PREV", command=self._win_prev,
                  bg=CARD, fg=BLUE, font=("Courier", 9, "bold"), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(nav, text="▶ NEXT", command=self._win_next,
                  bg=BLUE, fg=WHITE, font=("Courier", 9, "bold"), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(nav, text="▶▶ AUTO", command=self._win_auto,
                  bg=PURPLE, fg=WHITE, font=("Courier", 9, "bold"), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(nav, text="■ STOP", command=self._win_stop,
                  bg=RED, fg=WHITE, font=("Courier", 9, "bold"), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)

        # Score summary (live)
        score_f = tk.Frame(center, bg=CARD)
        score_f.pack(fill="x", padx=8, pady=4)
        self.lbl_sign  = self._score_box(score_f, "IMG SIGN",  GREEN)
        self.lbl_amp   = self._score_box(score_f, "AMP IMG",   ORANGE)
        self.lbl_chain = self._score_box(score_f, "CHAIN",     TEAL)
        self.lbl_cos   = self._score_box(score_f, "COSINE",    PURPLE)

        # Verdict — besar dan tegas
        self.verdict_lbl = tk.Label(center,
            text="Upload dua gambar untuk memulai analisis",
            font=("Courier", 22, "bold"), bg=CARD, fg=SUB,
            pady=12, padx=20,
            highlightthickness=2, highlightbackground=BORDER)
        self.verdict_lbl.pack(pady=8, fill="x", padx=16)

    def _score_box(self, parent, label, color):
        f = tk.Frame(parent, bg="#0a0e1a", highlightthickness=1, highlightbackground=BORDER)
        f.pack(side="left", expand=True, fill="both", padx=4, pady=2)
        tk.Label(f, text=label, font=("Courier", 7, "bold"), bg="#0a0e1a", fg=color).pack(pady=(6,1))
        lbl = tk.Label(f, text="—", font=("Courier", 16, "bold"), bg="#0a0e1a", fg=color)
        lbl.pack(pady=(0,6))
        return lbl

    # ── RIGHT PANEL ──────────────────────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        right.grid(row=0, column=2, sticky="nsew", padx=(6,0))

        tk.Label(right, text="CONV PROCESSING  ·  FEATURE MAPS",
                 font=("Courier", 10, "bold"), bg=CARD, fg=TEAL).pack(pady=(10,4))

        # Resolution path
        res_path = tk.Frame(right, bg=CARD)
        res_path.pack(fill="x", padx=8, pady=2)
        steps = [
            ("SW1", "112→56"), ("Conv2", "56→56"), ("Conv3", "56→28"),
            ("Conv4", "28→28"), ("Conv5", "28→28"), ("Conv6", "28→14"),
            ("Conv7", "14→14"), ("Conv8", "14→14"), ("Conv9", "14→7"),
            ("Conv10","7→7"),   ("GAP","→EMB"),
        ]
        for i, (name, res) in enumerate(steps):
            col = BLUE if name.startswith("SW") else (TEAL if "GAP" in name else GREEN)
            f = tk.Frame(res_path, bg=CARD)
            f.grid(row=i//4, column=i%4, padx=2, pady=1)
            tk.Label(f, text=name, font=("Courier", 7, "bold"), bg=CARD, fg=col).pack()
            tk.Label(f, text=res, font=("Courier", 6), bg=CARD, fg=SUB).pack()

        tk.Label(right, text="Simulated Feature Maps per Layer",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack(pady=(6,2))

        # Feature map display (3 canvases for selected layer)
        self.feat_canvases = []
        feat_f = tk.Frame(right, bg=CARD)
        feat_f.pack(padx=8)
        for i in range(3):
            c = tk.Canvas(feat_f, width=90, height=90, bg="#050810",
                         highlightthickness=1, highlightbackground=BORDER)
            c.grid(row=0, column=i, padx=2)
            self.feat_canvases.append(c)

        # Embedding vector display
        tk.Label(right, text="Final Embedding (1024D → visualized)",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack(pady=(8,2))

        self.emb1_bar = tk.Canvas(right, width=280, height=40, bg="#050810",
                                   highlightthickness=1, highlightbackground=BORDER)
        self.emb1_bar.pack(padx=8)
        tk.Label(right, text="Embedding 1", font=("Courier", 7), bg=CARD, fg=BLUE).pack()

        self.emb2_bar = tk.Canvas(right, width=280, height=40, bg="#050810",
                                   highlightthickness=1, highlightbackground=BORDER)
        self.emb2_bar.pack(padx=8, pady=2)
        tk.Label(right, text="Embedding 2", font=("Courier", 7), bg=CARD, fg=GREEN).pack()

        # Sign pattern display
        tk.Label(right, text="Sign Pattern Match (per window)",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack(pady=(6,2))
        self.sign_canvas = tk.Canvas(right, width=280, height=60, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.sign_canvas.pack(padx=8)

        # Chain visualization
        tk.Label(right, text="Chain Pattern (continuous matches)",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack(pady=(6,2))
        self.chain_canvas = tk.Canvas(right, width=280, height=40, bg="#050810",
                                       highlightthickness=1, highlightbackground=BORDER)
        self.chain_canvas.pack(padx=8, pady=(0,8))


    # ────────────────────────────────────────────────────────
    # IMAGE LOADING
    # ────────────────────────────────────────────────────────
    def load_img1(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img1_array = self._load_img(path)
            self._display_img(self.img1_array, self.img1_canvas, 104)
            self._compute_and_update()

    def load_img2(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.img2_array = self._load_img(path)
            self._display_img(self.img2_array, self.img2_canvas, 104)
            self._compute_and_update()

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        # Coba crop wajah dengan MTCNN
        if MTCNN_OK and _mtcnn is not None:
            try:
                face = _mtcnn(img)
                if face is not None:
                    # face tensor (3, 112, 112) float
                    arr = face.permute(1, 2, 0).numpy()
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                    return arr
            except Exception:
                pass
        # Fallback: resize biasa
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img)

    def _display_img(self, arr, canvas, size):
        img = Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img


    # ────────────────────────────────────────────────────────
    # COMPUTE EMBEDDINGS AND UPDATE ALL PANELS
    # ────────────────────────────────────────────────────────
    def _compute_and_update(self):
        if self.img1_array is None or self.img2_array is None:
            return
        # Compute embeddings
        self.emb1 = get_embedding(self.img1_array)
        self.emb2 = get_embedding(self.img2_array)
        self.win_pos = 0
        self._update_sw_scan()
        self._update_center()
        self._update_right()
        self._update_scores()

    def _update_scores(self):
        if self.emb1 is None or self.emb2 is None: return
        e1, e2 = self.emb1, self.emb2
        n = len(e1) - WINDOW_SIZE + 1

        # IMG Sign Score
        total_match = sum(
            1 for i in range(n)
            if sum(1 for j in range(WINDOW_SIZE)
                   if (e1[i+j]>=0) == (e2[i+j]>=0)) >= THRESHOLD
        )
        sign = total_match / max(n, 1)

        # AMP Score
        amp_total = 0.0
        for i in range(n):
            w1, w2 = e1[i:i+WINDOW_SIZE], e2[i:i+WINDOW_SIZE]
            s1 = np.where(w1 >= 0, 1, -1)
            s2 = np.where(w2 >= 0, 1, -1)
            if int(np.sum(s1 == s2)) >= THRESHOLD:
                a1, a2 = np.mean(np.abs(w1)), np.mean(np.abs(w2))
                amp_total += max(0.0, 1 - abs(a1-a2) / max(a1,a2,1e-6))
        amp = amp_total / max(n, 1)

        # Chain
        cs, n_chains, avg_chain = chain_score(e1, e2)

        # Cosine
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))

        self.lbl_sign.config(text=f"{sign:.3f}")
        self.lbl_amp.config(text=f"{amp:.3f}")
        self.lbl_chain.config(text=f"{cs:.3f}")
        self.lbl_cos.config(text=f"{cos:.3f}")

        # Verdict
        thr = 0.79
        n_pass = sum([sign >= thr, amp >= thr, cs >= thr])
        if n_pass >= 2:
            self.verdict_lbl.config(
                text="✅  MATCH",
                fg=WHITE, bg="#064e3b",
                highlightbackground=GREEN,
                font=("Courier", 26, "bold"))
        elif n_pass == 1:
            self.verdict_lbl.config(
                text="⚠️   UNCERTAIN",
                fg=WHITE, bg="#78350f",
                highlightbackground=ORANGE,
                font=("Courier", 26, "bold"))
        else:
            self.verdict_lbl.config(
                text="❌  DIFFERENT",
                fg=WHITE, bg="#450a0a",
                highlightbackground=RED,
                font=("Courier", 26, "bold"))


    # ────────────────────────────────────────────────────────
    # SW BLOCK SCAN VISUALIZATION
    # ────────────────────────────────────────────────────────
    def _set_sw_window(self, ws):
        self.sw_window = ws
        self._update_sw_scan()

    def _update_sw_scan(self):
        if self.img1_array is None: return
        # Show image with SW window overlay
        self._draw_sw_overlay(0, 0)
        # Draw heatmap
        self._draw_heatmap()
        # Update conv feature map
        self._update_conv()

    def _draw_sw_overlay(self, scan_r, scan_c):
        if self.img1_array is None: return
        canvas_size = 224
        scale = canvas_size / IMG_SIZE

        img = Image.fromarray(self.img1_array.astype(np.uint8))
        img = img.resize((canvas_size, canvas_size), Image.NEAREST)

        # Draw scanning window
        draw = ImageDraw.Draw(img, "RGBA")
        ws = self.sw_window
        x0 = int(scan_c * scale)
        y0 = int(scan_r * scale)
        x1 = int((scan_c + ws) * scale)
        y1 = int((scan_r + ws) * scale)
        # Window highlight
        draw.rectangle([x0, y0, x1, y1], fill=(255, 165, 0, 60), outline=(255, 165, 0, 200), width=2)
        # Center pixel
        cx = int((scan_c + ws//2) * scale)
        cy = int((scan_r + ws//2) * scale)
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(255, 100, 100, 200))

        # Draw arrows to neighbors (just cardinal)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = scan_r + ws//2 + dr, scan_c + ws//2 + dc
            if 0 <= nr < IMG_SIZE and 0 <= nc < IMG_SIZE:
                nx = int(nc * scale)
                ny = int(nr * scale)
                draw.line([cx, cy, nx, ny], fill=(100, 255, 200, 150), width=1)

        tk_img = ImageTk.PhotoImage(img)
        self.sw_canvas.delete("all")
        self.sw_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.sw_canvas.image = tk_img

        # Label current window
        self.sw_canvas.create_text(4, 4, anchor="nw",
            text=f"SW {ws}×{ws}  pos=({scan_r},{scan_c})",
            font=("Courier", 8), fill=ORANGE)

    def _draw_heatmap(self):
        if self.img1_array is None: return
        hmap = sw_scan_result(self.img1_array, self.sw_window)
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

        # Colormap: dark blue → cyan → yellow
        h_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        h_img[:,:,0] = (hmap_norm * 255).astype(np.uint8)
        h_img[:,:,1] = ((1 - hmap_norm) * 200).astype(np.uint8)
        h_img[:,:,2] = ((1 - hmap_norm) * 255).astype(np.uint8)

        pil_img = Image.fromarray(h_img).resize((224, 112), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.heat_canvas.delete("all")
        self.heat_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.heat_canvas.image = tk_img
        self.heat_canvas.create_text(4, 4, anchor="nw",
            text=f"Relational diff (SW {self.sw_window}×{self.sw_window})",
            font=("Courier", 7), fill=ORANGE)

    def animate_sw(self):
        if self.img1_array is None: return
        self.sw_animating = not self.sw_animating
        if self.sw_animating:
            self._sw_animate_loop(0, 0)

    def _sw_animate_loop(self, r, c):
        if not self.sw_animating: return
        ws = self.sw_window
        stride = max(1, ws // 2)
        self._draw_sw_overlay(r, c)
        # Next position
        nc = c + stride
        nr = r
        if nc + ws > IMG_SIZE:
            nc = 0
            nr = r + stride
        if nr + ws > IMG_SIZE:
            nr = 0
            nc = 0
        self.root.after(80, self._sw_animate_loop, nr, nc)

    def _update_conv(self):
        if self.img1_array is None: return
        layer = self.conv_var.get()
        layer_name = f"conv{layer}"

        # Ukuran resolusi per layer
        sizes = {2:56, 3:28, 4:28, 5:28, 6:14, 7:14, 8:14, 9:7, 10:7}
        sz = sizes.get(layer, 14)

        # Coba pakai feature map asli dari hooks
        fmap = _feature_maps.get(layer_name, None)

        if fmap is not None:
            # fmap: (1, C, H, W) — ambil 3 channel pertama
            fmap_np = fmap[0].numpy()  # (C, H, W)
            n_ch = fmap_np.shape[0]
            for i, canvas in enumerate(self.feat_canvases):
                ch_idx = int(i * n_ch / 3)
                ch = fmap_np[ch_idx]
                # Normalize
                vmin, vmax = ch.min(), ch.max()
                ch_norm = (ch - vmin) / (vmax - vmin + 1e-8)
                # Colorize
                rgb = np.zeros((ch.shape[0], ch.shape[1], 3), dtype=np.uint8)
                if i == 0:
                    rgb[:,:,0] = (ch_norm * 255).astype(np.uint8)
                    rgb[:,:,2] = ((1-ch_norm) * 150).astype(np.uint8)
                elif i == 1:
                    rgb[:,:,1] = (ch_norm * 255).astype(np.uint8)
                    rgb[:,:,2] = ((1-ch_norm) * 100).astype(np.uint8)
                else:
                    rgb[:,:,0] = (ch_norm * 150).astype(np.uint8)
                    rgb[:,:,1] = (ch_norm * 200).astype(np.uint8)
                pil = Image.fromarray(rgb).resize((90, 90), Image.NEAREST)
                tk_img = ImageTk.PhotoImage(pil)
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=tk_img)
                canvas.image = tk_img
                canvas.create_text(4, 4, anchor="nw",
                    text=f"Ch{ch_idx+1}/{n_ch} {layer_name} {ch.shape[0]}²",
                    font=("Courier", 6), fill=WHITE)
        else:
            # Fallback simulasi kalau belum ada embedding
            from PIL import ImageFilter
            img = Image.fromarray(self.img1_array.astype(np.uint8)).convert("L")
            img_small = img.resize((sz, sz), Image.BILINEAR)
            for i, canvas in enumerate(self.feat_canvases):
                filtered = img_small.filter(ImageFilter.GaussianBlur(radius=i+1))
                arr = np.array(filtered, dtype=np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                rgb = np.zeros((sz, sz, 3), dtype=np.uint8)
                rgb[:,:,i % 3] = (arr * 200).astype(np.uint8)
                pil = Image.fromarray(rgb).resize((90, 90), Image.NEAREST)
                tk_img = ImageTk.PhotoImage(pil)
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=tk_img)
                canvas.image = tk_img
                canvas.create_text(4, 4, anchor="nw",
                    text=f"simulated {layer_name} {sz}²",
                    font=("Courier", 6), fill=SUB)

        # Conv activation bar
        self.conv_canvas.delete("all")
        if fmap is not None:
            # Rata-rata semua channel → 1D bar
            avg = fmap[0].mean(dim=0).numpy()  # (H, W)
            avg_flat = avg.flatten()
            avg_norm = (avg_flat - avg_flat.min()) / (avg_flat.max() - avg_flat.min() + 1e-8)
            cw = 224
            for x in range(cw):
                idx = int(x / cw * len(avg_norm))
                v = int(avg_norm[idx] * 255)
                col = f"#{v:02x}{min(255,v+60):02x}{max(0,255-v):02x}"
                self.conv_canvas.create_line(x, 0, x, 56, fill=col)
            self.conv_canvas.create_text(4, 4, anchor="nw",
                text=f"Conv{layer} mean activation  ({sz}×{sz}, {fmap.shape[1]}ch)  — REAL",
                font=("Courier", 7), fill=TEAL)
        else:
            self.conv_canvas.create_text(4, 28, anchor="w",
                text=f"Upload gambar untuk lihat feature map Conv{layer}",
                font=("Courier", 7), fill=SUB)


    # ────────────────────────────────────────────────────────
    # CENTER PANEL UPDATE
    # ────────────────────────────────────────────────────────
    def _update_center(self):
        if self.emb1 is None or self.emb2 is None: return
        self._draw_embedding_bars()
        self._draw_window_detail()
        if self.mode.get() == "training":
            self._draw_tanh_curve()
        else:
            self._draw_tanh_curve()  # always show

    def _draw_embedding_bars(self):
        """Draw full embedding as bar chart with current window highlighted"""
        if self.emb1 is None: return
        canvas = self.emb_canvas
        canvas.delete("all")
        W, H = 560, 180
        n = len(self.emb1)
        bar_w = W / n
        mid = H // 2

        # Draw grid lines
        canvas.create_line(0, mid, W, mid, fill=BORDER, width=1)
        canvas.create_text(4, 4, anchor="nw",
            text=f"Embedding vectors ({n}D)  — Biru=E1  Hijau=E2",
            font=("Courier", 8), fill=SUB)

        n_win = n - WINDOW_SIZE + 1

        for i in range(n):
            # Highlight current window
            in_window = self.win_pos <= i < self.win_pos + WINDOW_SIZE
            x0 = i * bar_w
            x1 = x0 + bar_w - 0.5

            # E1
            v1 = float(self.emb1[i])
            h1 = abs(v1) * (mid - 10)
            col1 = BLUE if not in_window else "#a5b4fc"
            if v1 >= 0:
                canvas.create_rectangle(x0, mid-h1, x1, mid, fill=col1, outline="")
            else:
                canvas.create_rectangle(x0, mid, x1, mid+h1, fill=col1, outline="")

            # E2
            v2 = float(self.emb2[i])
            h2 = abs(v2) * (mid - 10) * 0.6
            col2 = GREEN if not in_window else "#6ee7b7"
            if v2 >= 0:
                canvas.create_rectangle(x0, mid-h2, x1, mid, fill=col2, outline="", stipple="gray25")
            else:
                canvas.create_rectangle(x0, mid, x1, mid+h2, fill=col2, outline="", stipple="gray25")

        # Draw window highlight box
        wx0 = self.win_pos * bar_w
        wx1 = (self.win_pos + WINDOW_SIZE) * bar_w
        canvas.create_rectangle(wx0, 2, wx1, H-2, outline=ORANGE, width=2)
        canvas.create_text(wx0+2, H-14, anchor="sw",
            text=f"w={self.win_pos}", font=("Courier", 7), fill=ORANGE)

        # Update info
        n_match = sum(1 for j in range(WINDOW_SIZE)
                      if (self.emb1[self.win_pos+j] >= 0) == (self.emb2[self.win_pos+j] >= 0))
        self.win_info.config(
            text=f"Window: {self.win_pos}  |  Position: {self.win_pos}/{n_win-1}  |  Match: {n_match}/{WINDOW_SIZE}  ({'✓ PASS' if n_match>=THRESHOLD else '✗ FAIL'})",
            fg=GREEN if n_match >= THRESHOLD else RED
        )

    def _draw_window_detail(self):
        """Draw detailed view of current window"""
        canvas = self.win_canvas
        canvas.delete("all")
        W, H = 560, 140

        if self.emb1 is None: return
        mode = self.mode.get()

        pos = self.win_pos
        w1 = self.emb1[pos:pos+WINDOW_SIZE]
        w2 = self.emb2[pos:pos+WINDOW_SIZE]

        bar_w = W / WINDOW_SIZE
        mid = H // 2 - 10

        canvas.create_text(4, 4, anchor="nw",
            text=f"Window [{pos}:{pos+WINDOW_SIZE}]  —  {'Training: tanh agreement' if mode=='training' else 'Metric: sign matching'}",
            font=("Courier", 8), fill=ORANGE if mode == "training" else PURPLE)

        for i in range(WINDOW_SIZE):
            x0 = i * bar_w + 2
            x1 = x0 + bar_w - 4
            xc = (x0 + x1) / 2

            v1 = float(w1[i])
            v2 = float(w2[i])
            same_sign = (v1 >= 0) == (v2 >= 0)

            if mode == "training":
                # Show tanh agreement value
                agree = float(tanh_agreement(v1, v2))
                col = self._lerp_color(RED, GREEN, agree)
                h = agree * (mid - 5)
                canvas.create_rectangle(x0, mid-h, x1, mid, fill=col, outline="")
                canvas.create_text(xc, H-20, anchor="center",
                    text=f"{agree:.2f}", font=("Courier", 6), fill=col)
                # Show gradient arrow
                if agree > 0.5:
                    canvas.create_text(xc, mid-h-10, anchor="center",
                        text="▲", font=("Courier", 8), fill=GREEN)
                else:
                    canvas.create_text(xc, mid+8, anchor="center",
                        text="▼", font=("Courier", 8), fill=RED)
            else:
                # Metric mode: show sign match
                s1 = "+" if v1 >= 0 else "−"
                s2 = "+" if v2 >= 0 else "−"
                col = GREEN if same_sign else RED
                canvas.create_rectangle(x0, 20, x1, mid, fill=col, outline="")
                canvas.create_text(xc, 30, anchor="center",
                    text=s1, font=("Courier", 12, "bold"), fill=WHITE)
                canvas.create_text(xc, 50, anchor="center",
                    text=s2, font=("Courier", 12, "bold"), fill=WHITE)
                canvas.create_text(xc, mid+8, anchor="center",
                    text="✓" if same_sign else "✗",
                    font=("Courier", 10), fill=col)

            # E1 and E2 values
            canvas.create_text(xc, H-8, anchor="center",
                text=f"{v1:.1f}", font=("Courier", 5), fill=BLUE)

        # Match count bar
        n_match = sum(1 for j in range(WINDOW_SIZE)
                      if (w1[j] >= 0) == (w2[j] >= 0))
        match_w = (n_match / WINDOW_SIZE) * (W - 20)
        canvas.create_rectangle(10, H-4, 10+match_w, H-1,
                                  fill=GREEN if n_match >= THRESHOLD else RED, outline="")
        canvas.create_text(W//2, H-3, anchor="center",
            text=f"Match: {n_match}/{WINDOW_SIZE}  (thr={THRESHOLD})  {'PASS ✓' if n_match>=THRESHOLD else 'FAIL ✗'}",
            font=("Courier", 7), fill=GREEN if n_match >= THRESHOLD else RED)

    def _draw_tanh_curve(self):
        """Draw tanh curve for current window"""
        canvas = self.tanh_canvas
        canvas.delete("all")
        W, H = 560, 120

        if self.emb1 is None: return

        pos = self.win_pos
        w1 = self.emb1[pos:pos+WINDOW_SIZE]
        w2 = self.emb2[pos:pos+WINDOW_SIZE]

        # Draw axes
        mid_y = H // 2
        canvas.create_line(0, mid_y, W, mid_y, fill=BORDER, width=1, dash=(4,2))
        canvas.create_line(W//2, 0, W//2, H, fill=BORDER, width=1, dash=(4,2))

        # Draw tanh curve (general)
        xs = np.linspace(-3, 3, W)
        ys_tanh = (np.tanh(xs) + 1) / 2  # agreement curve

        pts_curve = []
        for px in range(W):
            x_val = xs[px]
            y_val = ys_tanh[px]
            py = int(mid_y - y_val * (mid_y - 10))
            pts_curve.append((px, py))

        for i in range(len(pts_curve)-1):
            canvas.create_line(pts_curve[i][0], pts_curve[i][1],
                               pts_curve[i+1][0], pts_curve[i+1][1],
                               fill=ORANGE, width=2)

        # Plot actual window values as dots
        for j in range(WINDOW_SIZE):
            v1, v2 = float(w1[j]), float(w2[j])
            prod = v1 * v2 * BETA
            agree = (math.tanh(prod) + 1) / 2
            # Map prod to x
            px = int((prod + 3) / 6 * W)
            px = max(0, min(W-1, px))
            py = int(mid_y - agree * (mid_y - 10))
            same = (v1 >= 0) == (v2 >= 0)
            col = GREEN if same else RED
            canvas.create_oval(px-4, py-4, px+4, py+4, fill=col, outline=WHITE)

        # Labels
        canvas.create_text(4, 4, anchor="nw",
            text=f"tanh(β·E1·E2) — β={BETA}  |  Hijau=sign cocok  Merah=berbeda  |  {'Training: gradient dorong ke 1.0' if self.mode.get()=='training' else 'Metric: ambang batas sign'}",
            font=("Courier", 7), fill=SUB)
        canvas.create_text(4, H-4, anchor="sw",
            text="prod<0 (berbeda tanda)", font=("Courier", 7), fill=RED)
        canvas.create_text(W-4, H-4, anchor="se",
            text="prod>0 (sama tanda)", font=("Courier", 7), fill=GREEN)

        # Training mode: show gradient arrows
        if self.mode.get() == "training":
            canvas.create_text(W//2, 10, anchor="center",
                text="▲ Loss = (1-score)² → dorong agreement ke 1.0 untuk same-pair",
                font=("Courier", 7), fill=YELLOW)


    # ────────────────────────────────────────────────────────
    # RIGHT PANEL UPDATE
    # ────────────────────────────────────────────────────────
    def _update_right(self):
        if self.emb1 is None: return
        self._draw_emb_bar(self.emb1_bar, self.emb1, BLUE)
        self._draw_emb_bar(self.emb2_bar, self.emb2, GREEN)
        self._draw_sign_pattern()
        self._draw_chain_pattern()

    def _draw_emb_bar(self, canvas, emb, color):
        canvas.delete("all")
        W, H = 280, 40
        n = len(emb)
        bw = W / n
        mid = H // 2
        for i, v in enumerate(emb):
            x0 = i * bw
            h = abs(float(v)) * (mid - 2)
            col = color if float(v) >= 0 else RED
            if float(v) >= 0:
                canvas.create_rectangle(x0, mid-h, x0+bw-0.5, mid, fill=col, outline="")
            else:
                canvas.create_rectangle(x0, mid, x0+bw-0.5, mid+h, fill=col, outline="")

    def _draw_sign_pattern(self):
        canvas = self.sign_canvas
        canvas.delete("all")
        if self.emb1 is None: return
        W, H = 280, 60
        n = len(self.emb1) - WINDOW_SIZE + 1
        bw = W / n
        scores = img_sign_score(self.emb1, self.emb2)
        for i, s in enumerate(scores):
            x0 = i * bw
            col = GREEN if s >= THRESHOLD/WINDOW_SIZE else RED
            h = s * (H - 4)
            canvas.create_rectangle(x0, H-h, x0+bw-0.3, H, fill=col, outline="")
        canvas.create_text(4, 4, anchor="nw",
            text=f"Sign match score per window (thr={THRESHOLD}/{WINDOW_SIZE})",
            font=("Courier", 6), fill=SUB)

    def _draw_chain_pattern(self):
        canvas = self.chain_canvas
        canvas.delete("all")
        if self.emb1 is None: return
        W, H = 280, 40
        e1, e2 = self.emb1, self.emb2
        n = len(e1) - WINDOW_SIZE + 1
        bw = W / n
        in_chain = False
        for i in range(n):
            s1 = np.where(e1[i:i+WINDOW_SIZE]>=0, 1, -1)
            s2 = np.where(e2[i:i+WINDOW_SIZE]>=0, 1, -1)
            match = int(np.sum(s1==s2)) >= THRESHOLD
            x0 = i * bw
            if match:
                canvas.create_rectangle(x0, 8, x0+bw-0.3, H-8, fill=TEAL, outline="")
                if not in_chain:
                    canvas.create_line(x0, 4, x0, H-4, fill=WHITE, width=1)
                in_chain = True
            else:
                in_chain = False
        canvas.create_text(4, 4, anchor="nw",
            text="Chain pattern (hijau=match run, garis=chain start)",
            font=("Courier", 6), fill=SUB)


    # ────────────────────────────────────────────────────────
    # WINDOW NAVIGATION
    # ────────────────────────────────────────────────────────
    def _win_first(self):
        self.win_pos = 0
        self._update_center()

    def _win_next(self):
        if self.emb1 is None: return
        n = len(self.emb1) - WINDOW_SIZE + 1
        self.win_pos = min(self.win_pos + 1, n - 1)
        self._update_center()

    def _win_prev(self):
        self.win_pos = max(self.win_pos - 1, 0)
        self._update_center()

    def _win_stop(self):
        self.animating = False

    def _win_auto(self):
        self.animating = True
        self._auto_loop()

    def _auto_loop(self):
        if not self.animating: return
        if self.emb1 is None: return
        n = len(self.emb1) - WINDOW_SIZE + 1
        self.win_pos = (self.win_pos + 1) % n
        self._update_center()
        self.root.after(120, self._auto_loop)


    # ────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────
    def _lerp_color(self, c1, c2, t):
        r1,g1,b1 = int(c1[1:3],16), int(c1[3:5],16), int(c1[5:7],16)
        r2,g2,b2 = int(c2[1:3],16), int(c2[3:5],16), int(c2[5:7],16)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def open_ablation_window(self):
        """Buka window ablation study terpisah"""
        if self.img1_array is None:
            tk.messagebox.showwarning("Warning", "Upload dulu Image 1!")
            return
        AblationWindow(self.root, self.img1_array, self.emb1)


# ============================================================
# ABLATION STUDY WINDOW
# Hapus region wajah → lihat delta embedding per dimensi
# ============================================================
class AblationWindow(tk.Toplevel):
    REGIONS = {
        "Mata Kiri"   : (25, 20, 50, 55),   # r1,c1,r2,c2
        "Mata Kanan"  : (25, 57, 50, 90),
        "Hidung"      : (50, 35, 75, 77),
        "Mulut"       : (75, 28, 95, 84),
        "Dahi"        : (5,  20, 28, 92),
        "Rahang Kiri" : (75, 5,  112, 42),
        "Rahang Kanan": (75, 70, 112, 107),
        "Semua Mata"  : (20, 15, 55, 97),
        "Bagian Atas" : (0,  0,  56, 112),
        "Bagian Bawah": (56, 0,  112, 112),
    }
    MASK_COLOR = 128  # abu-abu untuk okluasi

    def __init__(self, parent, img_array, emb_original):
        super().__init__(parent)
        self.title("IMGNet — Ablation Study: Occlusion Sensitivity")
        self.geometry("1200x780")
        self.configure(bg=BG)

        self.img_original  = img_array.copy()
        self.emb_original  = emb_original.copy() if emb_original is not None else None
        self.selected_regs = {}   # name → tk.BooleanVar
        self.delta_cache   = {}   # name → delta array

        self._build_ui()
        self._precompute_all()

    def _build_ui(self):
        # Title
        tk.Label(self, text="Ablation Study  ·  Occlusion Sensitivity Analysis",
                 font=("Courier", 13, "bold"), bg=BG, fg=PURPLE).pack(pady=(10,2))
        tk.Label(self,
            text="Hapus region wajah → bandingkan embedding → lihat dimensi mana yang paling sensitif",
            font=("Courier", 9), bg=BG, fg=SUB).pack(pady=(0,8))

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=4)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=3)
        main.grid_rowconfigure(0, weight=1)

        # ── LEFT: region selector + preview ─────────────────
        left = tk.Frame(main, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))

        tk.Label(left, text="PILIH REGION OKLUASI",
                 font=("Courier", 10, "bold"), bg=CARD, fg=ORANGE).pack(pady=(10,4))

        # Checkboxes per region
        for name in self.REGIONS:
            var = tk.BooleanVar(value=False)
            self.selected_regs[name] = var
            cb = tk.Checkbutton(left, text=name, variable=var,
                               bg=CARD, fg=TEXT, selectcolor=CARD,
                               font=("Courier", 9),
                               command=self._update_preview)
            cb.pack(anchor="w", padx=16)

        tk.Button(left, text="□ Clear All",
                  command=self._clear_all,
                  bg=CARD, fg=RED, font=("Courier", 8),
                  relief="flat", pady=2, cursor="hand2").pack(pady=4)

        tk.Button(left, text="■ Select All",
                  command=self._select_all,
                  bg=CARD, fg=GREEN, font=("Courier", 8),
                  relief="flat", pady=2, cursor="hand2").pack()

        # Preview foto asli + masked
        tk.Label(left, text="Original", font=("Courier", 8), bg=CARD, fg=SUB).pack(pady=(12,0))
        self.orig_canvas = tk.Canvas(left, width=140, height=140, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.orig_canvas.pack(padx=8)
        self._show_img(self.img_original, self.orig_canvas, 140)

        tk.Label(left, text="With Occlusion", font=("Courier", 8), bg=CARD, fg=ORANGE).pack(pady=(6,0))
        self.mask_canvas = tk.Canvas(left, width=140, height=140, bg="#050810",
                                      highlightthickness=1, highlightbackground=BORDER)
        self.mask_canvas.pack(padx=8, pady=(0,8))

        # Masked embedding delta score
        self.delta_score_lbl = tk.Label(left, text="Δ score: —",
                                         font=("Courier", 11, "bold"), bg=CARD, fg=YELLOW)
        self.delta_score_lbl.pack(pady=4)

        # ── RIGHT: delta visualization ───────────────────────
        right = tk.Frame(main, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="DELTA EMBEDDING — |E_original - E_occluded| per dimensi",
                 font=("Courier", 10, "bold"), bg=CARD, fg=TEAL).pack(pady=(10,2))
        tk.Label(right,
            text="Dimensi dengan delta TINGGI = sensitif terhadap region yang dihapus",
            font=("Courier", 8), bg=CARD, fg=SUB).pack()

        # Delta bar chart
        self.delta_canvas = tk.Canvas(right, width=820, height=200, bg="#050810",
                                       highlightthickness=1, highlightbackground=BORDER)
        self.delta_canvas.pack(padx=8, pady=4, fill="x")

        # Smoothed delta (running average)
        tk.Label(right, text="Smoothed Delta (window=20) — identifikasi cluster region",
                 font=("Courier", 8), bg=CARD, fg=SUB).pack()
        self.smooth_canvas = tk.Canvas(right, width=820, height=120, bg="#050810",
                                        highlightthickness=1, highlightbackground=BORDER)
        self.smooth_canvas.pack(padx=8, pady=2, fill="x")

        # Multi-region overlay
        tk.Label(right, text="Perbandingan Semua Region (overlay)",
                 font=("Courier", 9, "bold"), bg=CARD, fg=PURPLE).pack(pady=(8,2))
        self.overlay_canvas = tk.Canvas(right, width=820, height=160, bg="#050810",
                                         highlightthickness=1, highlightbackground=BORDER)
        self.overlay_canvas.pack(padx=8, pady=2, fill="x")

        # Top sensitive dimensions
        tk.Label(right, text="Top 10 Dimensi Paling Sensitif",
                 font=("Courier", 9, "bold"), bg=CARD, fg=YELLOW).pack(pady=(8,2))
        self.top_dims_lbl = tk.Label(right, text="—",
                                      font=("Courier", 9), bg=CARD, fg=TEXT,
                                      justify="left", wraplength=800)
        self.top_dims_lbl.pack(padx=16, pady=(0,8))

        # Update button
        tk.Button(right, text="🔄 ANALYZE",
                  command=self._update_all,
                  bg=PURPLE, fg=WHITE, font=("Courier", 11, "bold"),
                  relief="flat", padx=20, pady=6, cursor="hand2").pack(pady=4)

    def _show_img(self, arr, canvas, size):
        img = Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img

    def _apply_mask(self, regions):
        """Terapkan okluasi abu-abu ke region yang dipilih"""
        masked = self.img_original.copy()
        for name in regions:
            r1, c1, r2, c2 = self.REGIONS[name]
            masked[r1:r2, c1:c2] = self.MASK_COLOR
        return masked

    def _precompute_all(self):
        """Precompute delta untuk semua region"""
        if _imgnet_model is None or self.emb_original is None: return

        def worker():
            for name, (r1,c1,r2,c2) in self.REGIONS.items():
                masked = self.img_original.copy()
                masked[r1:r2, c1:c2] = self.MASK_COLOR
                emb_masked = get_embedding(masked)
                self.delta_cache[name] = np.abs(self.emb_original - emb_masked)
            self.root.after(0, lambda: self._draw_overlay())

        threading.Thread(target=worker, daemon=True).start()

    def _clear_all(self):
        for v in self.selected_regs.values(): v.set(False)
        self._update_preview()

    def _select_all(self):
        for v in self.selected_regs.values(): v.set(True)
        self._update_preview()

    def _update_preview(self):
        selected = [n for n, v in self.selected_regs.items() if v.get()]
        masked = self._apply_mask(selected)
        # Draw mask outline on preview
        img = Image.fromarray(masked.astype(np.uint8)).resize((140, 140), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        scale = 140 / 112
        for name in selected:
            r1,c1,r2,c2 = self.REGIONS[name]
            draw.rectangle([c1*scale, r1*scale, c2*scale, r2*scale],
                          outline="#f59e0b", width=2)
            draw.text((c1*scale+2, r1*scale+2), name[:4], fill="#f59e0b")
        tk_img = ImageTk.PhotoImage(img)
        self.mask_canvas.delete("all")
        self.mask_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.mask_canvas.image = tk_img

    def _update_all(self):
        selected = [n for n, v in self.selected_regs.items() if v.get()]
        if not selected:
            return
        self._update_preview()

        # Compute combined delta
        if _imgnet_model is not None:
            masked = self._apply_mask(selected)
            emb_masked = get_embedding(masked)
            delta = np.abs(self.emb_original - emb_masked)
            self._draw_delta(delta, f"Delta: {', '.join(selected)}")
            self._draw_smoothed(delta)

            # Score drop
            n = len(self.emb_original) - WINDOW_SIZE + 1
            orig_sign = sum(
                1 for i in range(n)
                if sum(1 for j in range(WINDOW_SIZE)
                       if (self.emb_original[i+j]>=0)==(emb_masked[i+j]>=0)) >= THRESHOLD
            ) / max(n, 1)
            self.delta_score_lbl.config(
                text=f"IMG Sign drop: {1-orig_sign:.3f}",
                fg=RED if (1-orig_sign) > 0.1 else YELLOW)

            # Top 10 sensitive dims
            top10 = np.argsort(delta)[-10:][::-1]
            self.top_dims_lbl.config(
                text=f"Dimensi: {list(top10)}  |  Delta: {[f'{delta[i]:.3f}' for i in top10]}")

    def _draw_delta(self, delta, title="Delta"):
        canvas = self.delta_canvas
        canvas.delete("all")
        W = canvas.winfo_width() or 820
        H = 200
        n = len(delta)
        bw = W / n
        d_max = delta.max() + 1e-8

        for i, d in enumerate(delta):
            x0 = i * bw
            h = (d / d_max) * (H - 20)
            # Color: low=blue, high=red
            t = d / d_max
            r = int(255 * t)
            b = int(255 * (1-t))
            col = f"#{r:02x}00{b:02x}"
            canvas.create_rectangle(x0, H-h, x0+bw-0.3, H, fill=col, outline="")

        # Mark top peaks
        top5 = np.argsort(delta)[-5:]
        for idx in top5:
            x = idx * bw + bw/2
            h = (delta[idx] / d_max) * (H - 20)
            canvas.create_oval(x-3, H-h-3, x+3, H-h+3, fill=YELLOW, outline="")
            canvas.create_text(x, H-h-10, text=str(idx),
                               font=("Courier", 6), fill=YELLOW)

        canvas.create_text(4, 4, anchor="nw", text=title,
                          font=("Courier", 8), fill=TEAL)
        canvas.create_text(W-4, 4, anchor="ne",
                          text=f"max_delta={delta.max():.4f}  mean={delta.mean():.4f}",
                          font=("Courier", 7), fill=SUB)

    def _draw_smoothed(self, delta, window=20):
        canvas = self.smooth_canvas
        canvas.delete("all")
        W = canvas.winfo_width() or 820
        H = 120

        # Running average
        smoothed = np.convolve(delta, np.ones(window)/window, mode='same')
        s_max = smoothed.max() + 1e-8
        n = len(smoothed)
        bw = W / n

        # Draw as filled area
        pts = [(0, H)]
        for i, s in enumerate(smoothed):
            x = i * bw
            y = H - (s / s_max) * (H - 10)
            pts.append((x, y))
        pts.append((W, H))

        if len(pts) > 2:
            canvas.create_polygon(pts, fill="#1e3a5f", outline="")
            # Draw line on top
            for i in range(len(pts)-2):
                canvas.create_line(pts[i+1][0], pts[i+1][1],
                                   pts[i+2][0], pts[i+2][1],
                                   fill=BLUE, width=1)

        # Find peaks in smoothed
        peaks = []
        for i in range(1, n-1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > s_max * 0.5:
                    peaks.append(i)

        for pk in peaks[:5]:
            x = pk * bw
            y = H - (smoothed[pk] / s_max) * (H - 10)
            canvas.create_oval(x-4, y-4, x+4, y+4, fill=ORANGE, outline="")
            canvas.create_text(x, y-12, text=f"dim{pk}",
                               font=("Courier", 6), fill=ORANGE)

        canvas.create_text(4, 4, anchor="nw",
                          text=f"Smoothed delta (window={window}) — cluster = kemungkinan region spasial di embedding",
                          font=("Courier", 7), fill=SUB)

    def _draw_overlay(self):
        """Overlay semua region yang sudah diprecompute"""
        canvas = self.overlay_canvas
        canvas.delete("all")
        if not self.delta_cache: return

        W = canvas.winfo_width() or 820
        H = 160

        REGION_COLORS = [
            BLUE, GREEN, ORANGE, RED, PURPLE,
            TEAL, YELLOW, "#f472b6", "#34d399", "#60a5fa"
        ]

        names = list(self.delta_cache.keys())
        for idx, name in enumerate(names):
            delta = self.delta_cache[name]
            d_max = max(d.max() for d in self.delta_cache.values()) + 1e-8
            n = len(delta)
            bw = W / n
            col = REGION_COLORS[idx % len(REGION_COLORS)]

            smoothed = np.convolve(delta, np.ones(15)/15, mode='same')
            pts = []
            for i, s in enumerate(smoothed):
                x = i * bw
                y = H - 10 - (s / d_max) * (H - 20)
                pts.append((x, y))

            for i in range(len(pts)-1):
                canvas.create_line(pts[i][0], pts[i][1],
                                   pts[i+1][0], pts[i+1][1],
                                   fill=col, width=1)

        # Legend
        for idx, name in enumerate(names):
            col = REGION_COLORS[idx % len(REGION_COLORS)]
            x = 8 + (idx % 5) * 155
            y = 8 + (idx // 5) * 14
            canvas.create_rectangle(x, y, x+8, y+8, fill=col, outline="")
            canvas.create_text(x+10, y, anchor="nw",
                               text=name, font=("Courier", 6), fill=col)

        canvas.create_text(W//2, H-4, anchor="s",
            text="Tiap warna = region berbeda  ·  Puncak = cluster dimensi sensitif",
            font=("Courier", 7), fill=SUB)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = IMGNetVisualizer(root)
    root.mainloop()
