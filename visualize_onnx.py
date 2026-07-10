# imgnet_visualizer_onnx.py
# IMGNet Interactive Visualizer — pakai ONNX Runtime (tidak butuh PyTorch)
# Panel Kiri  : Upload 2 foto + MTCNN crop
# Panel Tengah: Sliding window embedding analysis (Training vs Metric mode)
# Panel Kanan : Embedding bars + Sign + Chain pattern

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import math
from collections import Counter

# ── CONFIG ─────────────────────────────────────────────────
ONNX_PATH   = r"C:\PythonProj\img_bnn\imgnet_conv10_epoch39.onnx"
WINDOW_SIZE = 11
THRESHOLD   = 8
EMB_DIM     = 1024
IMG_SIZE    = 112
BETA        = 10.0
NEUTRAL_LEN = 29
REWARD_RATE = 0.3
PUNISH_RATE = 1.0

# ── COLORS ─────────────────────────────────────────────────
BG=    "#0a0e1a"; CARD=  "#111827"; BORDER="#1e293b"
BLUE=  "#6366f1"; GREEN= "#10b981"; ORANGE="#f59e0b"
PURPLE="#a855f7"; TEAL=  "#14b8a6"; RED=   "#ef4444"
YELLOW="#fbbf24"; WHITE= "#ffffff"; SUB=   "#64748b"; TEXT="#e2e8f0"

# ── ONNX RUNTIME LOAD ──────────────────────────────────────
try:
    import onnxruntime as rt
    import os
    if os.path.exists(ONNX_PATH):
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
        _sess = rt.InferenceSession(ONNX_PATH, providers=providers)
        _inp_name = _sess.get_inputs()[0].name
        ONNX_OK = True
        # Detect actual device
        used = _sess.get_providers()[0]
        DEVICE_STR = "CUDA" if "CUDA" in used else "CPU"
        print(f"✓ ONNX loaded — provider: {used}")
    else:
        _sess = None; ONNX_OK = False; DEVICE_STR = "—"
        print(f"✗ ONNX file tidak ditemukan: {ONNX_PATH}")
except ImportError:
    _sess = None; ONNX_OK = False; DEVICE_STR = "—"
    print("✗ onnxruntime tidak terinstall — pip install onnxruntime")

# ── MTCNN ──────────────────────────────────────────────────
try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(image_size=112, keep_all=False, post_process=False, device="cpu")
    MTCNN_OK = True
except: _mtcnn = None; MTCNN_OK = False


# ── INFERENCE ──────────────────────────────────────────────
def get_emb(arr):
    """arr: np.uint8 (112,112,3) → embedding (1024,)"""
    if _sess is not None:
        t = arr.astype(np.float32) / 255.0
        t = t.transpose(2, 0, 1)[np.newaxis]  # (1,3,112,112)
        return _sess.run(None, {_inp_name: t})[0][0]
    # Fallback dummy
    np.random.seed(int(arr.sum()) % 2**31)
    e = np.random.randn(EMB_DIM).astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-8)

def load_face(path):
    img = Image.open(path).convert("RGB")
    if MTCNN_OK and _mtcnn:
        try:
            face = _mtcnn(img)
            if face is not None:
                return np.clip(face.permute(1,2,0).numpy(), 0, 255).astype(np.uint8)
        except: pass
    return np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


# ── METRICS ────────────────────────────────────────────────
def img_sign_score(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    return sum(1 for i in range(n)
               if sum(1 for j in range(WINDOW_SIZE)
                      if (e1[i+j]>=0)==(e2[i+j]>=0)) >= THRESHOLD) / max(n,1)

def amp_score(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1; tot = 0.0
    for i in range(n):
        w1, w2 = e1[i:i+WINDOW_SIZE], e2[i:i+WINDOW_SIZE]
        s1 = np.where(w1>=0,1,-1).astype(np.int8)
        s2 = np.where(w2>=0,1,-1).astype(np.int8)
        if int(np.sum(s1==s2)) >= THRESHOLD:
            a1, a2 = np.mean(np.abs(w1)), np.mean(np.abs(w2))
            tot += max(0.0, 1 - abs(a1-a2)/max(a1,a2,1e-6))
    return tot / max(n,1)

def chain_score(e1, e2):
    n = len(e1) - WINDOW_SIZE + 1
    flags = [int(np.sum(
                np.where(e1[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8) ==
                np.where(e2[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8)
             )) >= THRESHOLD for i in range(n)]
    total = sum(flags); sg = total / max(n,1)
    nc = 0; ic = False
    for f in flags:
        if f and not ic: nc+=1; ic=True
        elif not f: ic=False
    if nc==0 or total==0: return 0.0, 0, 0.0
    ac = total/nc; diff = ac - NEUTRAL_LEN
    s = sg + (REWARD_RATE*diff if diff>=0 else PUNISH_RATE*diff)/100
    return float(np.clip(s,0,1)), nc, ac

def cosine(e1, e2):
    return float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-8))

def tanh_agreement(e1, e2):
    return (np.tanh(BETA * e1 * e2) + 1) / 2


# ============================================================
# APP
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMGNet Visualizer — ONNX Runtime")
        self.geometry("1500x880")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.arr1 = None; self.arr2 = None
        self.e1   = None; self.e2   = None
        self.win_pos  = 0
        self.animating = False
        self.mode = tk.StringVar(value="metric")

        self._build()

    def _build(self):
        # Top bar
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=12, pady=(8,4))
        tk.Label(top, text="IMGNet  ·  Interactive Visualizer  ·  ONNX Runtime",
                 font=("Courier",13,"bold"), bg=BG, fg=TEXT).pack(side="left")
        st = f"ONNX={'✓' if ONNX_OK else '✗'}  Device={DEVICE_STR}  MTCNN={'✓' if MTCNN_OK else '✗'}  w={WINDOW_SIZE} t={THRESHOLD}/11"
        tk.Label(top, text=st, font=("Courier",9), bg=BG, fg=SUB).pack(side="right")

        main = tk.Frame(self, bg=BG); main.pack(fill="both", expand=True, padx=8, pady=4)
        main.grid_columnconfigure(0, weight=0, minsize=240)
        main.grid_columnconfigure(1, weight=1)
        main.grid_columnconfigure(2, weight=0, minsize=260)
        main.grid_rowconfigure(0, weight=1)

        self._build_left(main)
        self._build_center(main)
        self._build_right(main)

    # ── LEFT ─────────────────────────────────────────────
    def _build_left(self, parent):
        lf = tk.Frame(parent, bg=CARD, highlightthickness=1,
                      highlightbackground=BORDER, width=240)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        lf.grid_propagate(False)

        tk.Label(lf, text="INPUT IMAGES", font=("Courier",10,"bold"),
                 bg=CARD, fg=BLUE).pack(pady=(10,4))

        # Upload buttons
        bf = tk.Frame(lf, bg=CARD); bf.pack(fill="x", padx=8)
        tk.Button(bf, text="Upload Foto 1", command=self.upload1,
                  bg=BLUE, fg=WHITE, font=("Courier",9,"bold"),
                  relief="flat", pady=5, cursor="hand2").pack(side="left", expand=True, fill="x", padx=(0,2))
        tk.Button(bf, text="Upload Foto 2", command=self.upload2,
                  bg=GREEN, fg=WHITE, font=("Courier",9,"bold"),
                  relief="flat", pady=5, cursor="hand2").pack(side="left", expand=True, fill="x", padx=(2,0))

        # Preview
        pf = tk.Frame(lf, bg=CARD); pf.pack(fill="x", padx=8, pady=6)
        pf.grid_columnconfigure(0, weight=1); pf.grid_columnconfigure(1, weight=1)
        for col, label, color, attr in [(0,"Foto 1",BLUE,"c1"),(1,"Foto 2",GREEN,"c2")]:
            f = tk.Frame(pf, bg=CARD); f.grid(row=0, column=col, padx=2)
            tk.Label(f, text=label, font=("Courier",8,"bold"), bg=CARD, fg=color).pack()
            c = tk.Canvas(f, width=100, height=100, bg="#050810",
                          highlightthickness=1, highlightbackground=BORDER)
            c.pack(); setattr(self, attr, c)

        # Score boxes
        tk.Label(lf, text="METRICS", font=("Courier",9,"bold"),
                 bg=CARD, fg=SUB).pack(pady=(8,2))
        sf = tk.Frame(lf, bg=CARD); sf.pack(fill="x", padx=6)
        self.m_sign  = self._mbox(sf, "IMG SIGN",  GREEN)
        self.m_amp   = self._mbox(sf, "AMP IMG",   ORANGE)
        self.m_chain = self._mbox(sf, "CHAIN",     TEAL)
        self.m_cos   = self._mbox(sf, "COSINE",    PURPLE)

        # Verdict
        self.verdict = tk.Label(lf, text="—",
                                font=("Courier",20,"bold"), bg=CARD, fg=SUB,
                                pady=8, highlightthickness=2, highlightbackground=BORDER)
        self.verdict.pack(fill="x", padx=8, pady=6)

        # Chain detail
        self.chain_lbl = tk.Label(lf, text="", font=("Courier",8),
                                   bg=CARD, fg=SUB, justify="left", wraplength=200)
        self.chain_lbl.pack(padx=10)

        # Mode
        tk.Label(lf, text="MODE", font=("Courier",8,"bold"),
                 bg=CARD, fg=SUB).pack(pady=(10,2))
        mf = tk.Frame(lf, bg=CARD); mf.pack()
        for val, label, col in [("metric","METRIC",GREEN),("training","TRAINING",ORANGE)]:
            tk.Radiobutton(mf, text=label, variable=self.mode, value=val,
                          bg=CARD, fg=col, selectcolor=CARD,
                          font=("Courier",8,"bold"),
                          command=self._update_center).pack(side="left", padx=6)

        # Nav buttons
        tk.Label(lf, text="WINDOW NAV", font=("Courier",8,"bold"),
                 bg=CARD, fg=SUB).pack(pady=(10,2))
        nf = tk.Frame(lf, bg=CARD); nf.pack(fill="x", padx=6)
        for text, cmd, col in [
            ("◀◀",self._win_first,SUB), ("◀",self._win_prev,BLUE),
            ("▶",self._win_next,BLUE),  ("AUTO",self._win_auto,PURPLE),
            ("■",self._win_stop,RED)
        ]:
            tk.Button(nf, text=text, command=cmd, bg=CARD, fg=col,
                     font=("Courier",9,"bold"), relief="flat",
                     padx=4, pady=4, cursor="hand2").pack(side="left", expand=True)

        # Win info
        self.win_info = tk.Label(lf, text="Window: —", font=("Courier",8),
                                  bg=CARD, fg=SUB, wraplength=200)
        self.win_info.pack(padx=8, pady=4)

    def _mbox(self, parent, label, color):
        f = tk.Frame(parent, bg="#0a0e1a", highlightthickness=1, highlightbackground=BORDER)
        f.pack(side="left", expand=True, fill="both", padx=2, pady=2)
        tk.Label(f, text=label, font=("Courier",6,"bold"), bg="#0a0e1a", fg=color).pack(pady=(3,0))
        lbl = tk.Label(f, text="—", font=("Courier",12,"bold"), bg="#0a0e1a", fg=color)
        lbl.pack(pady=(0,3)); return lbl

    # ── CENTER ───────────────────────────────────────────
    def _build_center(self, parent):
        cf = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        cf.grid(row=0, column=1, sticky="nsew", padx=6)

        tk.Label(cf, text="SLIDING WINDOW EMBEDDING ANALYSIS",
                 font=("Courier",10,"bold"), bg=CARD, fg=PURPLE).pack(pady=(8,2))

        # Embedding overlay bar
        tk.Label(cf, text="Embedding: Biru=E1  Hijau=E2  — orange box = window aktif",
                 font=("Courier",8), bg=CARD, fg=SUB).pack()
        self.c_emb = tk.Canvas(cf, bg="#050810", height=180,
                                highlightthickness=1, highlightbackground=BORDER)
        self.c_emb.pack(fill="x", padx=8, pady=2)

        # Window detail
        self.win_title = tk.Label(cf, text="Window detail",
                                   font=("Courier",8), bg=CARD, fg=SUB)
        self.win_title.pack()
        self.c_win = tk.Canvas(cf, bg="#050810", height=150,
                                highlightthickness=1, highlightbackground=BORDER)
        self.c_win.pack(fill="x", padx=8, pady=2)

        # tanh curve
        tk.Label(cf, text="tanh(β·E1·E2) Agreement Curve  (β=10)  — titik = dimensi di window aktif",
                 font=("Courier",8), bg=CARD, fg=ORANGE).pack()
        self.c_tanh = tk.Canvas(cf, bg="#050810", height=130,
                                 highlightthickness=1, highlightbackground=BORDER)
        self.c_tanh.pack(fill="x", padx=8, pady=2)

        # Sign pattern all windows
        tk.Label(cf, text="Sign match ratio per window  (hijau ≥ threshold, merah = tidak)",
                 font=("Courier",8), bg=CARD, fg=GREEN).pack(pady=(4,0))
        self.c_sign = tk.Canvas(cf, bg="#050810", height=70,
                                 highlightthickness=1, highlightbackground=BORDER)
        self.c_sign.pack(fill="x", padx=8, pady=2)

        # Chain pattern
        tk.Label(cf, text="Chain pattern  (rantai match kontinu)",
                 font=("Courier",8), bg=CARD, fg=TEAL).pack()
        self.c_chain = tk.Canvas(cf, bg="#050810", height=55,
                                  highlightthickness=1, highlightbackground=BORDER)
        self.c_chain.pack(fill="x", padx=8, pady=(2,6))

    # ── RIGHT ────────────────────────────────────────────
    def _build_right(self, parent):
        rf = tk.Frame(parent, bg=CARD, highlightthickness=1,
                      highlightbackground=BORDER, width=260)
        rf.grid(row=0, column=2, sticky="nsew", padx=(6,0))
        rf.grid_propagate(False)

        tk.Label(rf, text="EMBEDDING BARS",
                 font=("Courier",10,"bold"), bg=CARD, fg=TEAL).pack(pady=(8,2))

        tk.Label(rf, text="E1 — Foto 1 (1024D)", font=("Courier",8,"bold"),
                 bg=CARD, fg=BLUE).pack()
        self.c_e1 = tk.Canvas(rf, bg="#050810", height=55,
                               highlightthickness=1, highlightbackground=BORDER)
        self.c_e1.pack(fill="x", padx=8, pady=2)

        tk.Label(rf, text="E2 — Foto 2 (1024D)", font=("Courier",8,"bold"),
                 bg=CARD, fg=GREEN).pack()
        self.c_e2 = tk.Canvas(rf, bg="#050810", height=55,
                               highlightthickness=1, highlightbackground=BORDER)
        self.c_e2.pack(fill="x", padx=8, pady=2)

        tk.Label(rf, text="ARCHITECTURE", font=("Courier",9,"bold"),
                 bg=CARD, fg=SUB).pack(pady=(12,4))
        steps = [
            ("SW1",   "112→56", BLUE),
            ("Conv2", "56→56",  GREEN), ("Conv3", "56→28",  GREEN),
            ("Conv4", "28→28",  GREEN), ("Conv5", "28→28",  GREEN),
            ("Conv6", "28→14",  GREEN), ("Conv7", "14→14",  GREEN),
            ("Conv8", "14→14",  GREEN), ("Conv9", "14→7",   GREEN),
            ("Conv10","7→7",    GREEN), ("GAP",   "→256",   TEAL),
            ("FC",    "→1024",  PURPLE),
        ]
        gf = tk.Frame(rf, bg=CARD); gf.pack(padx=8, fill="x")
        for i, (name, res, col) in enumerate(steps):
            f = tk.Frame(gf, bg=CARD); f.grid(row=i//3, column=i%3, padx=2, pady=1, sticky="w")
            tk.Label(f, text=name, font=("Courier",7,"bold"), bg=CARD, fg=col).pack()
            tk.Label(f, text=res,  font=("Courier",6),        bg=CARD, fg=SUB).pack()

        tk.Label(rf, text="STATS", font=("Courier",9,"bold"),
                 bg=CARD, fg=SUB).pack(pady=(12,4))
        self.stats_lbl = tk.Label(rf, text="—", font=("Courier",8),
                                   bg=CARD, fg=TEXT, justify="left", wraplength=240)
        self.stats_lbl.pack(padx=10)

    # ── UPLOAD ───────────────────────────────────────────
    def upload1(self):
        path = filedialog.askopenfilename(filetypes=[("Image","*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.arr1 = load_face(path)
        self.e1   = get_emb(self.arr1)
        self._show(self.arr1, self.c1, 100)
        self._refresh()

    def upload2(self):
        path = filedialog.askopenfilename(filetypes=[("Image","*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.arr2 = load_face(path)
        self.e2   = get_emb(self.arr2)
        self._show(self.arr2, self.c2, 100)
        self._refresh()

    def _show(self, arr, canvas, size):
        img = Image.fromarray(arr.astype(np.uint8)).resize((size,size), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img

    def _refresh(self):
        if self.e1 is None or self.e2 is None: return
        self._update_scores()
        self._draw_emb_bars()
        self._update_center()
        self._draw_sign_pattern()
        self._draw_chain_pattern()
        self._update_stats()

    # ── SCORES ───────────────────────────────────────────
    def _update_scores(self):
        e1, e2 = self.e1, self.e2
        sg = img_sign_score(e1, e2)
        ap = amp_score(e1, e2)
        cs, nc, ac = chain_score(e1, e2)
        co = cosine(e1, e2)
        self.m_sign.config(text=f"{sg:.3f}")
        self.m_amp.config(text=f"{ap:.3f}")
        self.m_chain.config(text=f"{cs:.3f}")
        self.m_cos.config(text=f"{co:.3f}")
        self.chain_lbl.config(text=f"Chains: {nc}  AvgLen: {ac:.1f}\n(neutral={NEUTRAL_LEN})")

        thr = 0.79; npass = sum([sg>=thr, ap>=thr, cs>=thr])
        if npass >= 2:
            self.verdict.config(text="✅  MATCH",    fg=WHITE, bg="#064e3b", highlightbackground=GREEN)
        elif npass == 1:
            self.verdict.config(text="⚠️  UNCERTAIN", fg=WHITE, bg="#78350f", highlightbackground=ORANGE)
        else:
            self.verdict.config(text="❌  DIFFERENT", fg=WHITE, bg="#450a0a", highlightbackground=RED)

    def _update_stats(self):
        e1, e2 = self.e1, self.e2
        n = len(e1) - WINDOW_SIZE + 1
        n_match = sum(1 for i in range(n)
                      if sum(1 for j in range(WINDOW_SIZE)
                             if (e1[i+j]>=0)==(e2[i+j]>=0)) >= THRESHOLD)
        self.stats_lbl.config(
            text=f"EMB dim    : {len(e1)}\n"
                 f"N windows  : {n}\n"
                 f"Pass thr   : {n_match}/{n}\n"
                 f"E1 norm    : {np.linalg.norm(e1):.3f}\n"
                 f"E2 norm    : {np.linalg.norm(e2):.3f}\n"
                 f"E1 pos dim : {(e1>=0).sum()}/{len(e1)}\n"
                 f"E2 pos dim : {(e2>=0).sum()}/{len(e2)}\n"
                 f"Sign agree : {((e1>=0)==(e2>=0)).sum()}/{len(e1)}")

    # ── EMBEDDING BARS ────────────────────────────────────
    def _draw_emb_bars(self):
        c = self.c_emb; c.delete("all")
        W = c.winfo_width() or 900; H = 180
        n = len(self.e1); bw = W/n; mid = H//2
        c.create_line(0, mid, W, mid, fill=BORDER, width=1, dash=(3,2))
        for i in range(n):
            x0 = i*bw; x1 = x0+bw-0.3
            v1 = float(self.e1[i]); h1 = abs(v1)*(mid-4)
            in_win = self.win_pos <= i < self.win_pos + WINDOW_SIZE
            col1 = "#a5b4fc" if in_win else BLUE
            if v1>=0: c.create_rectangle(x0, mid-h1, x1, mid, fill=col1, outline="")
            else:     c.create_rectangle(x0, mid,    x1, mid+h1, fill=col1, outline="")
            v2 = float(self.e2[i]); h2 = abs(v2)*(mid-4)*0.7
            col2 = "#6ee7b7" if in_win else GREEN
            if v2>=0: c.create_rectangle(x0, mid-h2, x1, mid, fill=col2, outline="", stipple="gray25")
            else:     c.create_rectangle(x0, mid,    x1, mid+h2, fill=col2, outline="", stipple="gray25")

        # Window highlight box
        wx0 = self.win_pos * bw; wx1 = (self.win_pos + WINDOW_SIZE) * bw
        c.create_rectangle(wx0, 2, wx1, H-2, outline=ORANGE, width=2)

        # Emb bars kanan
        for cv, emb, col in [(self.c_e1, self.e1, BLUE), (self.c_e2, self.e2, GREEN)]:
            cv.delete("all")
            W2 = cv.winfo_width() or 240; H2 = 55
            n2 = len(emb); bw2 = W2/n2; mid2 = H2//2
            for i, v in enumerate(emb):
                x0 = i*bw2; h = abs(float(v))*(mid2-2)
                fc = col if float(v)>=0 else RED
                if float(v)>=0: cv.create_rectangle(x0, mid2-h, x0+bw2-0.3, mid2, fill=fc, outline="")
                else:           cv.create_rectangle(x0, mid2,   x0+bw2-0.3, mid2+h, fill=fc, outline="")

    # ── CENTER DRAWS ──────────────────────────────────────
    def _update_center(self):
        if self.e1 is None: return
        self._draw_emb_bars()
        self._draw_window_detail()
        self._draw_tanh_curve()

        # Update win info
        e1, e2 = self.e1, self.e2
        n = len(e1) - WINDOW_SIZE + 1
        n_match = sum(1 for j in range(WINDOW_SIZE)
                      if (e1[self.win_pos+j]>=0)==(e2[self.win_pos+j]>=0))
        self.win_info.config(
            text=f"Win {self.win_pos}/{n-1}  match {n_match}/{WINDOW_SIZE}  "
                 f"{'✓ PASS' if n_match>=THRESHOLD else '✗ FAIL'}",
            fg=GREEN if n_match>=THRESHOLD else RED)
        self.win_title.config(
            text=f"Window [{self.win_pos}:{self.win_pos+WINDOW_SIZE}]  — "
                 f"{'tanh agreement (training)' if self.mode.get()=='training' else 'sign matching (metric)'}")

    def _draw_window_detail(self):
        c = self.c_win; c.delete("all")
        W = c.winfo_width() or 900; H = 150
        pos = self.win_pos
        w1 = self.e1[pos:pos+WINDOW_SIZE]
        w2 = self.e2[pos:pos+WINDOW_SIZE]
        bw = W / WINDOW_SIZE; mid = H//2 - 10
        mode = self.mode.get()

        for i in range(WINDOW_SIZE):
            x0 = i*bw+2; x1 = x0+bw-4; xc = (x0+x1)/2
            v1, v2 = float(w1[i]), float(w2[i])
            same = (v1>=0) == (v2>=0)

            if mode == "training":
                agree = float(tanh_agreement(v1, v2))
                col = self._lerp(RED, GREEN, agree)
                h = agree*(mid-5)
                c.create_rectangle(x0, mid-h, x1, mid, fill=col, outline="")
                c.create_text(xc, H-22, anchor="center",
                    text=f"{agree:.2f}", font=("Courier",6), fill=col)
                arrow = "▲" if agree>0.5 else "▼"
                c.create_text(xc, mid-h-10, anchor="center",
                    text=arrow, font=("Courier",9),
                    fill=GREEN if agree>0.5 else RED)
                # Gradient label
                if agree > 0.5:
                    c.create_text(xc, H-8, anchor="center",
                        text="→1", font=("Courier",5), fill=GREEN)
                else:
                    c.create_text(xc, H-8, anchor="center",
                        text="→0", font=("Courier",5), fill=RED)
            else:
                s1 = "+" if v1>=0 else "−"
                s2 = "+" if v2>=0 else "−"
                col = GREEN if same else RED
                c.create_rectangle(x0, 20, x1, mid, fill=col, outline="")
                c.create_text(xc, 32, anchor="center",
                    text=s1, font=("Courier",12,"bold"), fill=WHITE)
                c.create_text(xc, 54, anchor="center",
                    text=s2, font=("Courier",12,"bold"), fill=WHITE)
                c.create_text(xc, mid+8, anchor="center",
                    text="✓" if same else "✗", font=("Courier",9), fill=col)

        # Match bar
        n_match = sum(1 for j in range(WINDOW_SIZE)
                      if (w1[j]>=0)==(w2[j]>=0))
        mw = (n_match/WINDOW_SIZE)*(W-16)
        c.create_rectangle(8, H-6, 8+mw, H-1,
                           fill=GREEN if n_match>=THRESHOLD else RED, outline="")
        c.create_text(W//2, H-4, anchor="center",
            text=f"Match {n_match}/{WINDOW_SIZE}  thr={THRESHOLD}  {'PASS ✓' if n_match>=THRESHOLD else 'FAIL ✗'}",
            font=("Courier",7), fill=GREEN if n_match>=THRESHOLD else RED)

    def _draw_tanh_curve(self):
        c = self.c_tanh; c.delete("all")
        W = c.winfo_width() or 900; H = 130
        mid_y = H//2
        c.create_line(0, mid_y, W, mid_y, fill=BORDER, width=1, dash=(3,2))
        c.create_line(W//2, 0, W//2, H, fill=BORDER, width=1, dash=(3,2))

        # Curve
        xs = np.linspace(-3, 3, W)
        ys = (np.tanh(xs) + 1) / 2
        pts = [(i, int(mid_y - ys[i]*(mid_y-8))) for i in range(W)]
        for i in range(len(pts)-1):
            c.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                         fill=ORANGE, width=2)

        # Dots per dimensi di window
        pos = self.win_pos
        w1 = self.e1[pos:pos+WINDOW_SIZE]
        w2 = self.e2[pos:pos+WINDOW_SIZE]
        for j in range(WINDOW_SIZE):
            prod = float(w1[j]) * float(w2[j]) * BETA
            agree = (math.tanh(prod) + 1) / 2
            px = int((prod + 3) / 6 * W); px = max(0, min(W-1, px))
            py = int(mid_y - agree*(mid_y-8))
            same = (w1[j]>=0) == (w2[j]>=0)
            c.create_oval(px-4, py-4, px+4, py+4,
                         fill=GREEN if same else RED, outline=WHITE)

        mode = self.mode.get()
        c.create_text(4, 4, anchor="nw",
            text=f"{'Training: gradient dorong ke 1.0 (same) atau 0.0 (diff)' if mode=='training' else 'Metric: posisi dot = agreement di window ini'}",
            font=("Courier",7), fill=YELLOW)
        c.create_text(4, H-4, anchor="sw",
            text="prod<0 (beda tanda)", font=("Courier",6), fill=RED)
        c.create_text(W-4, H-4, anchor="se",
            text="prod>0 (sama tanda)", font=("Courier",6), fill=GREEN)

    def _draw_sign_pattern(self):
        c = self.c_sign; c.delete("all")
        W = c.winfo_width() or 900; H = 70
        e1, e2 = self.e1, self.e2
        n = len(e1) - WINDOW_SIZE + 1; bw = W/n
        for i in range(n):
            mc = sum(1 for j in range(WINDOW_SIZE)
                     if (e1[i+j]>=0)==(e2[i+j]>=0))
            h = (mc/WINDOW_SIZE)*(H-4)
            col = GREEN if mc>=THRESHOLD else RED
            c.create_rectangle(i*bw, H-h, i*bw+bw-0.3, H, fill=col, outline="")
        # Threshold line
        ty = H - (THRESHOLD/WINDOW_SIZE)*(H-4)
        c.create_line(0, ty, W, ty, fill=YELLOW, width=1, dash=(4,2))
        c.create_text(4, 4, anchor="nw",
            text=f"Sign match ratio per window  (garis kuning = thr {THRESHOLD}/{WINDOW_SIZE})",
            font=("Courier",7), fill=SUB)

    def _draw_chain_pattern(self):
        c = self.c_chain; c.delete("all")
        W = c.winfo_width() or 900; H = 55
        e1, e2 = self.e1, self.e2
        n = len(e1) - WINDOW_SIZE + 1; bw = W/n
        in_chain = False; chain_num = 0
        for i in range(n):
            mc = sum(1 for j in range(WINDOW_SIZE)
                     if (e1[i+j]>=0)==(e2[i+j]>=0))
            match = mc >= THRESHOLD; x0 = i*bw
            if match:
                c.create_rectangle(x0, 8, x0+bw-0.3, H-8, fill=TEAL, outline="")
                if not in_chain:
                    in_chain = True; chain_num += 1
                    c.create_line(x0, 4, x0, H-4, fill=WHITE, width=1)
                    c.create_text(x0+1, 5, anchor="nw",
                        text=str(chain_num), font=("Courier",5), fill=WHITE)
            else:
                in_chain = False
        c.create_text(4, 4, anchor="nw",
            text=f"Chain pattern — teal=match, putih=awal chain baru",
            font=("Courier",7), fill=SUB)

    # ── WINDOW NAV ────────────────────────────────────────
    def _win_first(self):
        self.win_pos=0; self._update_center()
    def _win_prev(self):
        self.win_pos=max(0,self.win_pos-1); self._update_center()
    def _win_next(self):
        if self.e1 is None: return
        n=len(self.e1)-WINDOW_SIZE+1
        self.win_pos=min(self.win_pos+1,n-1); self._update_center()
    def _win_stop(self):
        self.animating=False
    def _win_auto(self):
        self.animating=True; self._auto_loop()
    def _auto_loop(self):
        if not self.animating or self.e1 is None: return
        n=len(self.e1)-WINDOW_SIZE+1
        self.win_pos=(self.win_pos+1)%n
        self._update_center()
        self.after(100, self._auto_loop)

    # ── HELPERS ──────────────────────────────────────────
    def _lerp(self, c1, c2, t):
        r1,g1,b1=int(c1[1:3],16),int(c1[3:5],16),int(c1[5:7],16)
        r2,g2,b2=int(c2[1:3],16),int(c2[3:5],16),int(c2[5:7],16)
        r=int(r1+(r2-r1)*t); g=int(g1+(g2-g1)*t); b=int(b1+(b2-b1)*t)
        return f"#{r:02x}{g:02x}{b:02x}"


# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.after(200, app._refresh)
    app.mainloop()