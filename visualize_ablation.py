# imgnet_ablation_viewer.py
# IMGNet Ablation + Comparison Viewer
# Kiri : Upload 2 foto + score metrics (IMG Sign, AMP, Chain, Cosine)
# Kanan: Compare (overlay/sign/chain/embedding) & Ablation (baca-ulang sliding window)
# Ablation: bandingkan HASIL BACA ULANG sliding window (asli vs terdampak-occlusion)
#           → tampilkan blok/window mana saja yang berubah, bukan cuma skor akhir

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import threading
from collections import Counter

# ── CONFIG ─────────────────────────────────────────────────
CKPT_PATH   = r"C:\PythonProj\img_bnn\checkpoints_sw357_conv10_imgsign\SW357_conv10_imgsign\best_model_epoch39_plateau.pth"
WINDOW_SIZE = 11
THRESHOLD   = 8
EMB_DIM     = 1024
IMG_SIZE    = 112
NEUTRAL_LEN = 29
REWARD_RATE = 0.3
PUNISH_RATE = 1.0
MASK_COLOR  = 128
SOFT_BETA          = 5.0   # sharpness tanh agreement per-dimensi (soft sign+magnitude)
SOFT_GATE_SHARPNESS= 15.0  # sharpness sigmoid gate di sekitar THRESHOLD (landai)

BG=    "#0a0e1a"; CARD=  "#111827"; BORDER="#1e293b"
BLUE=  "#6366f1"; GREEN= "#10b981"; ORANGE="#f59e0b"
PURPLE="#a855f7"; TEAL=  "#14b8a6"; RED=   "#ef4444"
YELLOW="#fbbf24"; WHITE= "#ffffff"; SUB=   "#64748b"; TEXT="#e2e8f0"

REGION_COLS = [
    "#60a5fa","#34d399","#f59e0b","#f472b6","#a78bfa",
    "#fb923c","#22d3ee","#e879f9","#facc15","#6ee7b7"
]
REGIONS = {
    "Mata Kiri"   : (25, 20, 50, 55),
    "Mata Kanan"  : (25, 57, 50, 90),
    "Semua Mata"  : (18, 12, 52, 100),
    "Hidung"      : (50, 35, 75, 77),
    "Mulut"       : (72, 25, 95, 87),
    "Dahi"        : (4,  18, 27, 94),
    "Rahang Kiri" : (72, 3,  112, 45),
    "Rahang Kanan": (72, 67, 112, 109),
    "Bagian Atas" : (0,  0,  56,  112),
    "Bagian Bawah": (56, 0,  112, 112),
}
CUSTOM_KEY = "Custom (6 Titik)"

# ── MODEL LOAD ─────────────────────────────────────────────
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    TORCH_OK = True
except: TORCH_OK = False

try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(image_size=112, keep_all=False, post_process=False,
                   device="cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu")
    MTCNN_OK = True
except: _mtcnn = None; MTCNN_OK = False

_model = None; _device = "cpu"

if TORCH_OK:
    class _SW(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(240,64),nn.ReLU(True),nn.Linear(64,32))
        def forward(self, x):
            B,C,H,W=x.shape; diffs=[]
            for ws in [3,5,7]:
                p=ws//2; xp=F.pad(x,[p,p,p,p],mode='reflect')
                pat=xp.unfold(2,ws,1).unfold(3,ws,1)
                d=x.unsqueeze(-1).unsqueeze(-1)-pat
                m=torch.ones(ws,ws,dtype=torch.bool,device=x.device); m[ws//2,ws//2]=False
                diffs.append(d[:,:,:,:,m])
            d=torch.cat(diffs,-1); B,C,H,W,N=d.shape
            o=self.fc(d.permute(0,2,3,1,4).reshape(B*H*W,C*N))
            return o.reshape(B,H,W,-1).permute(0,3,1,2)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sw1=_SW(); self.bn1=nn.BatchNorm2d(32)
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

    import os
    if os.path.exists(CKPT_PATH):
        try:
            _device="cuda" if torch.cuda.is_available() else "cpu"
            _model=_Net().to(_device)
            st=torch.load(CKPT_PATH,map_location="cpu",weights_only=False)
            if isinstance(st,dict) and "model" in st: st=st["model"]
            _model.load_state_dict(st); _model.eval()
            print(f"✓ IMGNet loaded  device={_device}")
        except Exception as e: print(f"✗ {e}"); _model=None


# ── UTILS ──────────────────────────────────────────────────
def get_emb(arr):
    if _model and TORCH_OK:
        try:
            t=torch.from_numpy(arr.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(_device)
            with torch.no_grad(): return _model(t).squeeze(0).cpu().numpy()
        except: pass
    np.random.seed(int(arr.sum())%2**31)
    e=np.random.randn(EMB_DIM).astype(np.float32)
    return e/(np.linalg.norm(e)+1e-8)

def load_face(path):
    img=Image.open(path).convert("RGB")
    if MTCNN_OK and _mtcnn:
        try:
            face=_mtcnn(img)
            if face is not None:
                return np.clip(face.permute(1,2,0).numpy(),0,255).astype(np.uint8)
        except: pass
    return np.array(img.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR))

def occlude(arr, name):
    out=arr.copy(); r1,c1,r2,c2=REGIONS[name]
    out[r1:r2,c1:c2]=MASK_COLOR; return out

def polygon_to_mask(points, size=IMG_SIZE):
    """points: list of (x,y) dalam ruang 112x112 (x=kolom, y=baris).
    Return boolean mask HxW — True = di dalam polygon."""
    img=Image.new('L',(size,size),0)
    ImageDraw.Draw(img).polygon(points,fill=255)
    return np.array(img)>0

def occlude_mask(arr, mask):
    if mask is None: return arr
    out=arr.copy(); out[mask]=MASK_COLOR; return out

def img_sign_score(e1,e2):
    n=len(e1)-WINDOW_SIZE+1
    return sum(1 for i in range(n)
               if sum(1 for j in range(WINDOW_SIZE)
                      if (e1[i+j]>=0)==(e2[i+j]>=0))>=THRESHOLD)/max(n,1)

def amp_score(e1,e2):
    n=len(e1)-WINDOW_SIZE+1; tot=0.0
    for i in range(n):
        w1,w2=e1[i:i+WINDOW_SIZE],e2[i:i+WINDOW_SIZE]
        s1=np.where(w1>=0,1,-1).astype(np.int8); s2=np.where(w2>=0,1,-1).astype(np.int8)
        if int(np.sum(s1==s2))>=THRESHOLD:
            a1,a2=np.mean(np.abs(w1)),np.mean(np.abs(w2))
            tot+=max(0.0,1-abs(a1-a2)/max(a1,a2,1e-6))
    return tot/max(n,1)

def chain_score(e1,e2):
    n=len(e1)-WINDOW_SIZE+1
    flags=[int(np.sum(np.where(e1[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8)==
                      np.where(e2[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8)))>=THRESHOLD
           for i in range(n)]
    total=sum(flags); sg=total/max(n,1)
    nc=0; ic=False
    for f in flags:
        if f and not ic: nc+=1; ic=True
        elif not f: ic=False
    if nc==0 or total==0: return 0.0,0,0.0
    ac=total/nc; diff=ac-NEUTRAL_LEN
    score=sg+(REWARD_RATE*diff if diff>=0 else PUNISH_RATE*diff)/100
    return float(np.clip(score,0,1)),nc,ac

def cosine(e1,e2):
    return float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-8))

def lerp_color(t, c1="#ef4444", c2="#10b981"):
    r1,g1,b1=int(c1[1:3],16),int(c1[3:5],16),int(c1[5:7],16)
    r2,g2,b2=int(c2[1:3],16),int(c2[3:5],16),int(c2[5:7],16)
    r=int(r1+(r2-r1)*t); g=int(g1+(g2-g1)*t); b=int(b1+(b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"

# ── ABLATION: BACA ULANG SLIDING WINDOW ────────────────────
def window_sign_match(e1, e2):
    """Baca ulang tiap sliding window (window_size=11, threshold=8) antara e1 dan e2.
    Return array boolean per window: True = window MASIH cocok (>=THRESHOLD bit
    searah), False = window ini BERUBAH akibat oklusi.
    e1 = embedding asli, e2 = embedding foto yang sudah di-occlude."""
    n=len(e1)-WINDOW_SIZE+1
    out=np.zeros(n,dtype=bool)
    for i in range(n):
        s1=np.where(e1[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8)
        s2=np.where(e2[i:i+WINDOW_SIZE]>=0,1,-1).astype(np.int8)
        out[i]=int(np.sum(s1==s2))>=THRESHOLD
    return out

def soft_window_match(e1, e2, beta=SOFT_BETA, gate_sharpness=SOFT_GATE_SHARPNESS):
    """Versi soft/differentiable dari window_sign_match + amp ratio (numpy port,
    tanpa torch), berdasarkan:
        agreement  = (tanh(beta * E1 * E2) + 1) / 2      # per-dimensi
        soft_match = sliding-window-sum(agreement)        # analog count 0..WINDOW_SIZE
        gate       = sigmoid(gate_sharpness*(soft_match - THRESHOLD + 0.5))
        amp1/amp2  = sliding-window-mean(|E1|), sliding-window-mean(|E2|)
    Tidak ada cliff seperti window_sign_match: 'agreement' juga mempertimbangkan
    magnitude, dan 'gate' transisi mulus di sekitar THRESHOLD alih-alih lompat.
    Return (gate, amp_ratio, combined) — semua array sepanjang n_win, TANPA
    di-gating oleh hasil hard-threshold (beda dari window_amp_sim)."""
    def window_sum(arr, w):
        c=np.cumsum(np.insert(arr,0,0.0))
        return c[w:]-c[:-w]
    prod=e1.astype(np.float64)*e2.astype(np.float64)
    agreement=(np.tanh(beta*prod)+1)/2
    soft_match=window_sum(agreement, WINDOW_SIZE)
    gate=1.0/(1.0+np.exp(-gate_sharpness*(soft_match-THRESHOLD+0.5)))
    amp1=window_sum(np.abs(e1.astype(np.float64)), WINDOW_SIZE)/WINDOW_SIZE
    amp2=window_sum(np.abs(e2.astype(np.float64)), WINDOW_SIZE)/WINDOW_SIZE
    amp_ratio=np.clip(1.0-np.abs(amp1-amp2)/np.maximum(np.maximum(amp1,amp2),1e-6),0.0,1.0)
    combined=gate*amp_ratio
    return gate, amp_ratio, combined

def changed_segments(blocks):
    """Kelompokkan window yang BERUBAH (False) menjadi segmen kontinu.
    Return list of (start_idx, length) — dipakai untuk breakdown, bukan cuma
    angka akhir."""
    segs=[]; i=0; n=len(blocks)
    while i<n:
        if not blocks[i]:
            j=i
            while j<n and not blocks[j]: j+=1
            segs.append((i,j-i)); i=j
        else:
            i+=1
    return segs


# ============================================================
# APP
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMGNet — Ablation + Comparison Viewer")
        self.geometry("1600x900")
        self.configure(bg=BG)
        self.resizable(True,True)

        # State
        self.arr1=None; self.arr2=None
        self.e1=None;   self.e2=None
        self.dc1={};    self.dc2={}   # cache per region: {"emb":..., "blocks":...}
        self.sel=tk.StringVar(value="Semua Mata")
        self.mode=tk.StringVar(value="compare")  # compare / ablation
        self.custom_points={1:[],2:[]}   # titik polygon TERPISAH per foto, max 6 tiap foto
        self.custom_mask={1:None,2:None} # mask hasil polygon per foto
        self.custom_active_photo=1       # foto mana yg lagi digambar di kanvas custom

        self._build()

    # ── BUILD ─────────────────────────────────────────────
    def _build(self):
        top=tk.Frame(self,bg=BG); top.pack(fill="x",padx=12,pady=(8,4))
        tk.Label(top,text="IMGNet  ·  Ablation + Comparison Viewer",
                 font=("Courier",13,"bold"),bg=BG,fg=TEXT).pack(side="left")
        st=f"Model={'✓' if _model else '✗ dummy'}  MTCNN={'✓' if MTCNN_OK else '✗'}"
        tk.Label(top,text=st,font=("Courier",9),bg=BG,fg=SUB).pack(side="right")

        main=tk.Frame(self,bg=BG); main.pack(fill="both",expand=True,padx=8,pady=4)
        main.grid_columnconfigure(0,weight=0,minsize=280)
        main.grid_columnconfigure(1,weight=1)
        main.grid_rowconfigure(0,weight=1)

        self._build_left(main)
        self._build_right(main)

    def _build_left(self,parent):
        lf_container=tk.Frame(parent,bg=CARD,highlightthickness=1,
                    highlightbackground=BORDER,width=280)
        lf_container.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        lf_container.grid_propagate(False)

        # Panel kiri dibungkus Canvas+Scrollbar supaya SEMUA kontrol (termasuk
        # region Custom & checkbox tampilan) selalu bisa dicapai walau window
        # dipersempit / kontennya makin panjang — sebelumnya item di bawah
        # bisa "hilang" (terpotong) tanpa cara buat mengaksesnya.
        lf_canvas=tk.Canvas(lf_container,bg=CARD,highlightthickness=0)
        lf_scroll=tk.Scrollbar(lf_container,orient="vertical",command=lf_canvas.yview)
        lf_canvas.configure(yscrollcommand=lf_scroll.set)
        lf_canvas.pack(side="left",fill="both",expand=True)
        lf_scroll.pack(side="right",fill="y")

        lf=tk.Frame(lf_canvas,bg=CARD)   # frame konten asli — semua widget di bawah nempel di sini
        lf_window=lf_canvas.create_window((0,0),window=lf,anchor="nw")

        def _on_lf_configure(event):
            lf_canvas.configure(scrollregion=lf_canvas.bbox("all"))
        lf.bind("<Configure>",_on_lf_configure)

        def _on_canvas_configure(event):
            lf_canvas.itemconfig(lf_window,width=event.width)
        lf_canvas.bind("<Configure>",_on_canvas_configure)

        def _on_mousewheel(event):
            lf_canvas.yview_scroll(int(-1*(event.delta/120)),"units")
        def _bind_wheel(_): lf_canvas.bind_all("<MouseWheel>",_on_mousewheel)
        def _unbind_wheel(_): lf_canvas.unbind_all("<MouseWheel>")
        lf_canvas.bind("<Enter>",_bind_wheel)
        lf_canvas.bind("<Leave>",_unbind_wheel)

        # Upload buttons
        tk.Label(lf,text="FOTO",font=("Courier",10,"bold"),bg=CARD,fg=BLUE).pack(pady=(8,4))
        btn_f=tk.Frame(lf,bg=CARD); btn_f.pack(fill="x",padx=8)
        tk.Button(btn_f,text="Upload Foto 1",command=self.upload1,
                  bg=BLUE,fg=WHITE,font=("Courier",9,"bold"),
                  relief="flat",pady=4,cursor="hand2").pack(side="left",expand=True,fill="x",padx=2)
        tk.Button(btn_f,text="Upload Foto 2",command=self.upload2,
                  bg=GREEN,fg=WHITE,font=("Courier",9,"bold"),
                  relief="flat",pady=4,cursor="hand2").pack(side="left",expand=True,fill="x",padx=2)

        # Preview row
        prev=tk.Frame(lf,bg=CARD); prev.pack(fill="x",padx=6,pady=4)
        prev.grid_columnconfigure(0,weight=1); prev.grid_columnconfigure(1,weight=1)
        for col,label,color,attr in [(0,"Foto 1",BLUE,"c1"),(1,"Foto 2",GREEN,"c2")]:
            f=tk.Frame(prev,bg=CARD); f.grid(row=0,column=col,padx=2)
            tk.Label(f,text=label,font=("Courier",8,"bold"),bg=CARD,fg=color).pack()
            c=tk.Canvas(f,width=120,height=120,bg="#050810",
                        highlightthickness=1,highlightbackground=BORDER); c.pack()
            setattr(self,attr,c)

        # Mode tabs
        tk.Label(lf,text="MODE",font=("Courier",9,"bold"),bg=CARD,fg=SUB).pack(pady=(6,2))
        mf=tk.Frame(lf,bg=CARD); mf.pack(fill="x",padx=8)
        for val,label,col in [("compare","COMPARE",TEAL),("ablation","ABLATION",PURPLE)]:
            tk.Radiobutton(mf,text=label,variable=self.mode,value=val,
                          bg=CARD,fg=col,selectcolor=CARD,
                          font=("Courier",9,"bold"),
                          command=self._refresh).pack(side="left",padx=6)

        # Indikator region aktif — selalu kelihatan di atas (nggak ikut ke-scroll)
        # biar nggak ambigu region mana yang benar-benar dipilih.
        self.region_active_lbl=tk.Label(lf,text="Region aktif: —",
                                         font=("Courier",8,"bold"),bg="#0a0e1a",fg=YELLOW,
                                         anchor="w",padx=6,pady=3)
        self.region_active_lbl.pack(fill="x",padx=8,pady=(2,4))

        # Metrics (compare mode)
        self.metric_f=tk.Frame(lf,bg=CARD); self.metric_f.pack(fill="x",padx=8,pady=4)
        self.m_sign  =self._mbox(self.metric_f,"IMG SIGN",GREEN)
        self.m_amp   =self._mbox(self.metric_f,"AMP IMG", ORANGE)
        self.m_chain =self._mbox(self.metric_f,"CHAIN",   TEAL)
        self.m_cos   =self._mbox(self.metric_f,"COSINE",  PURPLE)

        # Verdict
        self.verdict_lbl=tk.Label(lf,text="—",
                                   font=("Courier",18,"bold"),bg=CARD,fg=SUB,
                                   pady=6,highlightthickness=2,highlightbackground=BORDER)
        self.verdict_lbl.pack(fill="x",padx=8,pady=4)

        # Chain detail
        self.chain_lbl=tk.Label(lf,text="",font=("Courier",8),bg=CARD,fg=SUB,
                                 justify="left",wraplength=250)
        self.chain_lbl.pack(padx=10)

        # Ablation region selector
        tk.Label(lf,text="REGION OKLUASI",font=("Courier",9,"bold"),
                 bg=CARD,fg=ORANGE).pack(pady=(10,2))
        for i,name in enumerate(REGIONS):
            col=REGION_COLS[i%len(REGION_COLS)]
            tk.Radiobutton(lf,text=name,variable=self.sel,value=name,
                          bg=CARD,fg=TEXT,selectcolor=CARD,
                          font=("Courier",9),
                          command=self._on_region).pack(anchor="w",padx=16,pady=1)
        tk.Radiobutton(lf,text=CUSTOM_KEY,variable=self.sel,value=CUSTOM_KEY,
                      bg=CARD,fg=PURPLE,selectcolor=CARD,
                      font=("Courier",9,"bold"),
                      command=self._on_region).pack(anchor="w",padx=16,pady=(4,1))

        # ── Ablation display toggles ────────────────────────
        tk.Label(lf,text="TAMPILAN BACA-ULANG WINDOW",font=("Courier",9,"bold"),
                 bg=CARD,fg=YELLOW).pack(pady=(10,2))
        self.v_show_asli      = tk.BooleanVar(value=True)   # garis biru = embedding asli
        self.v_show_terdampak = tk.BooleanVar(value=True)   # garis merah = embedding ter-occlude
        self.v_show_blocks    = tk.BooleanVar(value=True)   # bar blok hijau/merah per window
        self.v_show_delta     = tk.BooleanVar(value=True)   # sinyal delta besar (spike)

        tk.Checkbutton(lf,text="Garis Asli (Biru)",variable=self.v_show_asli,
                       bg=CARD,fg=BLUE,selectcolor=CARD,activebackground=CARD,
                       activeforeground=BLUE,font=("Courier",9,"bold"),
                       command=self._on_region).pack(anchor="w",padx=16,pady=1)
        tk.Checkbutton(lf,text="Garis Terdampak (Merah)",variable=self.v_show_terdampak,
                       bg=CARD,fg=RED,selectcolor=CARD,activebackground=CARD,
                       activeforeground=RED,font=("Courier",9,"bold"),
                       command=self._on_region).pack(anchor="w",padx=16,pady=1)
        tk.Checkbutton(lf,text="Bar Blok Berubah (Hijau/Merah)",variable=self.v_show_blocks,
                       bg=CARD,fg=YELLOW,selectcolor=CARD,activebackground=CARD,
                       activeforeground=YELLOW,font=("Courier",9,"bold"),
                       command=self._on_region).pack(anchor="w",padx=16,pady=1)
        tk.Checkbutton(lf,text="Tandai Delta Besar (Spike)",variable=self.v_show_delta,
                       bg=CARD,fg=ORANGE,selectcolor=CARD,activebackground=CARD,
                       activeforeground=ORANGE,font=("Courier",9,"bold"),
                       command=self._on_region).pack(anchor="w",padx=16,pady=1)

        # Ablation breakdown (bukan cuma skor akhir)
        self.abl_lbl=tk.Label(lf,text="",font=("Courier",8),bg=CARD,fg=TEXT,
                               justify="left",wraplength=250)
        self.abl_lbl.pack(padx=10,pady=6)

    def _mbox(self,parent,label,color):
        f=tk.Frame(parent,bg="#0a0e1a",highlightthickness=1,highlightbackground=BORDER)
        f.pack(side="left",expand=True,fill="both",padx=2,pady=2)
        tk.Label(f,text=label,font=("Courier",6,"bold"),bg="#0a0e1a",fg=color).pack(pady=(4,0))
        lbl=tk.Label(f,text="—",font=("Courier",13,"bold"),bg="#0a0e1a",fg=color)
        lbl.pack(pady=(0,4)); return lbl

    def _build_right(self,parent):
        rf=tk.Frame(parent,bg=CARD,highlightthickness=1,highlightbackground=BORDER)
        rf.grid(row=0,column=1,sticky="nsew")

        # ── COMPARE MODE canvases ──────────────────────────
        self.cmp_frame=tk.Frame(rf,bg=CARD); self.cmp_frame.pack(fill="both",expand=True)

        tk.Label(self.cmp_frame,
            text="Embedding: Biru=Foto1  Hijau=Foto2  — window aktif di-highlight",
            font=("Courier",9,"bold"),bg=CARD,fg=TEAL).pack(pady=(6,1))
        self.c_emb=tk.Canvas(self.cmp_frame,bg="#050810",height=180,
                              highlightthickness=1,highlightbackground=BORDER)
        self.c_emb.pack(fill="x",padx=8,pady=2)

        tk.Label(self.cmp_frame,
            text="Sign match per window (hijau=match ≥8/11, merah=tidak)",
            font=("Courier",9,"bold"),bg=CARD,fg=GREEN).pack(pady=(4,1))
        self.c_sign=tk.Canvas(self.cmp_frame,bg="#050810",height=80,
                               highlightthickness=1,highlightbackground=BORDER)
        self.c_sign.pack(fill="x",padx=8,pady=2)

        tk.Label(self.cmp_frame,
            text="Chain pattern (rantai match kontinu — panjang rantai = kualitas kecocokan)",
            font=("Courier",9,"bold"),bg=CARD,fg=TEAL).pack(pady=(4,1))
        self.c_chain=tk.Canvas(self.cmp_frame,bg="#050810",height=70,
                                highlightthickness=1,highlightbackground=BORDER)
        self.c_chain.pack(fill="x",padx=8,pady=2)

        # Embedding bars foto 1 dan 2
        emb_row=tk.Frame(self.cmp_frame,bg=CARD)
        emb_row.pack(fill="x",padx=8,pady=(4,2))
        emb_row.grid_columnconfigure(0,weight=1)
        emb_row.grid_columnconfigure(1,weight=1)

        lf1=tk.Frame(emb_row,bg=CARD); lf1.grid(row=0,column=0,sticky="ew",padx=(0,4))
        tk.Label(lf1,text="Embedding Foto 1 (1024D)",font=("Courier",8,"bold"),
                 bg=CARD,fg=BLUE).pack()
        self.c_emb1=tk.Canvas(lf1,bg="#050810",height=60,
                               highlightthickness=1,highlightbackground=BORDER)
        self.c_emb1.pack(fill="x")

        lf2=tk.Frame(emb_row,bg=CARD); lf2.grid(row=0,column=1,sticky="ew",padx=(4,0))
        tk.Label(lf2,text="Embedding Foto 2 (1024D)",font=("Courier",8,"bold"),
                 bg=CARD,fg=GREEN).pack()
        self.c_emb2=tk.Canvas(lf2,bg="#050810",height=60,
                               highlightthickness=1,highlightbackground=BORDER)
        self.c_emb2.pack(fill="x")

        # ── ABLATION MODE canvases ─────────────────────────
        self.abl_frame=tk.Frame(rf,bg=CARD); # tidak di-pack dulu

        # Kanvas gambar polygon custom (6 titik) — hanya di-pack saat
        # region terpilih == CUSTOM_KEY, disisipkan di atas panel hasil
        self.custom_draw_frame=tk.Frame(self.abl_frame,bg=CARD)
        tk.Label(self.custom_draw_frame,
            text="OKLUSI CUSTOM — klik 6 titik di foto (urut, otomatis nutup jadi polygon).  "
                 "Foto1 & Foto2 digambar TERPISAH — pilih tab di bawah.",
            font=("Courier",9,"bold"),bg=CARD,fg=PURPLE).pack(pady=(6,2))
        toggle_row=tk.Frame(self.custom_draw_frame,bg=CARD); toggle_row.pack(pady=(0,4))
        self.btn_custom_p1=tk.Button(toggle_row,text="✎ Gambar Foto 1",
                  command=lambda:self._switch_custom_photo(1),
                  font=("Courier",9,"bold"),relief="flat",padx=10,cursor="hand2")
        self.btn_custom_p1.pack(side="left",padx=4)
        self.btn_custom_p2=tk.Button(toggle_row,text="✎ Gambar Foto 2",
                  command=lambda:self._switch_custom_photo(2),
                  font=("Courier",9,"bold"),relief="flat",padx=10,cursor="hand2")
        self.btn_custom_p2.pack(side="left",padx=4)
        draw_row=tk.Frame(self.custom_draw_frame,bg=CARD); draw_row.pack()
        self.custom_scale=3
        cs=IMG_SIZE*self.custom_scale
        self.c_custom_draw=tk.Canvas(draw_row,width=cs,height=cs,bg="#050810",
                                      highlightthickness=1,highlightbackground=BORDER,
                                      cursor="crosshair")
        self.c_custom_draw.pack(padx=4,pady=2)
        self.c_custom_draw.bind("<Button-1>", self._on_custom_click)
        btn_row=tk.Frame(self.custom_draw_frame,bg=CARD); btn_row.pack(pady=(2,6))
        tk.Button(btn_row,text="Reset Titik",command=self._reset_custom_points,
                  bg=RED,fg=WHITE,font=("Courier",9,"bold"),relief="flat",
                  padx=8,cursor="hand2").pack(side="left",padx=4)
        self.btn_apply_custom=tk.Button(btn_row,text="Terapkan Oklusi",
                  command=self._apply_custom_occlusion,
                  bg=PURPLE,fg=WHITE,font=("Courier",9,"bold"),relief="flat",
                  padx=8,cursor="hand2",state="disabled")
        self.btn_apply_custom.pack(side="left",padx=4)
        self.custom_status_lbl=tk.Label(btn_row,text="0/6 titik",
                  font=("Courier",9,"bold"),bg=CARD,fg=SUB)
        self.custom_status_lbl.pack(side="left",padx=8)
        self._update_custom_toggle_style()

        self.abl_d1_title=tk.Label(self.abl_frame,
            text="Foto1 — Baca Ulang Sliding Window: Asli vs Terdampak",
            font=("Courier",9,"bold"),bg=CARD,fg=WHITE)
        self.abl_d1_title.pack(pady=(6,1))
        self.c_d1=tk.Canvas(self.abl_frame,bg="#050810",height=90,
                             highlightthickness=1,highlightbackground=BORDER)
        self.c_d1.pack(fill="x",padx=8,pady=2)
        self.c_blk1=tk.Canvas(self.abl_frame,bg="#050810",height=50,
                               highlightthickness=1,highlightbackground=BORDER)
        self.c_blk1.pack(fill="x",padx=8,pady=(0,6))

        self.abl_d2_title=tk.Label(self.abl_frame,
            text="Foto2 — Baca Ulang Sliding Window: Asli vs Terdampak",
            font=("Courier",9,"bold"),bg=CARD,fg=WHITE)
        self.abl_d2_title.pack(pady=(4,1))
        self.c_d2=tk.Canvas(self.abl_frame,bg="#050810",height=90,
                             highlightthickness=1,highlightbackground=BORDER)
        self.c_d2.pack(fill="x",padx=8,pady=2)
        self.c_blk2=tk.Canvas(self.abl_frame,bg="#050810",height=50,
                               highlightthickness=1,highlightbackground=BORDER)
        self.c_blk2.pack(fill="x",padx=8,pady=(0,6))

        tk.Label(self.abl_frame,
            text="Perbandingan region — % window yang berubah kalau region ini di-occlude (Biru=Foto1  Hijau=Foto2)",
            font=("Courier",9,"bold"),bg=CARD,fg=WHITE).pack(pady=(4,1))
        self.c_multi=tk.Canvas(self.abl_frame,bg="#050810",height=150,
                                highlightthickness=1,highlightbackground=BORDER)
        self.c_multi.pack(fill="x",padx=8,pady=(2,6))

        tk.Label(self.abl_frame,
            text="Konsistensi blok berubah: Foto1 vs Foto2 pada region terpilih — blok merah di lokasi SAMA = region ini konsisten di-encode walau beda orang",
            font=("Courier",9,"bold"),bg=CARD,fg=WHITE).pack(pady=(4,1))
        self.c_consist=tk.Canvas(self.abl_frame,bg="#050810",height=110,
                                  highlightthickness=1,highlightbackground=BORDER)
        self.c_consist.pack(fill="x",padx=8,pady=(2,6))

        # Bind resize agar grafik re-draw saat ukuran berubah
        for cv in [self.c_emb,self.c_sign,self.c_chain,
                   self.c_emb1,self.c_emb2,
                   self.c_d1,self.c_d2,self.c_blk1,self.c_blk2,
                   self.c_multi,self.c_consist]:
            cv.bind("<Configure>", lambda e: self.after(50, self._redraw_all))

        # Default: show compare
        self.cmp_frame.pack(fill="both",expand=True)

    # ── UPLOAD ────────────────────────────────────────────
    def _redraw_all(self):
        """Re-draw semua grafik yang aktif setelah window resize"""
        if self.mode.get()=="compare":
            self._draw_compare()
        else:
            self._on_region()
            self._draw_multi()

    def _get_w(self, canvas, fallback=1200):
        """Get actual canvas width after rendering"""
        w = canvas.winfo_width()
        return w if w > 10 else fallback

    def upload1(self):
        path=filedialog.askopenfilename(filetypes=[("Image","*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.arr1=load_face(path); self.e1=get_emb(self.arr1)
        self._show(self.arr1,self.c1,120)
        self.dc1={}
        threading.Thread(target=self._precompute,args=(1,),daemon=True).start()
        self._refresh()

    def upload2(self):
        path=filedialog.askopenfilename(filetypes=[("Image","*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.arr2=load_face(path); self.e2=get_emb(self.arr2)
        self._show(self.arr2,self.c2,120)
        self.dc2={}
        threading.Thread(target=self._precompute,args=(2,),daemon=True).start()
        self._refresh()

    def _show(self,arr,canvas,size,region=None):
        img=Image.fromarray(arr.astype(np.uint8)).resize((size,size),Image.NEAREST)
        if region:
            draw=ImageDraw.Draw(img); r1,c1,r2,c2=REGIONS[region]
            sc=size/IMG_SIZE
            draw.rectangle([c1*sc,r1*sc,c2*sc,r2*sc],outline=ORANGE,width=2)
        tk_img=ImageTk.PhotoImage(img)
        canvas.delete("all"); canvas.create_image(0,0,anchor="nw",image=tk_img)
        canvas.image=tk_img

    def _precompute(self,which):
        """Untuk tiap region: occlude foto, hitung ulang embedding, lalu BACA ULANG
        sliding window (window_sign_match) antara embedding asli vs embedding
        ter-occlude. Hasilnya array boolean per window (blocks), bukan kurva delta."""
        arr=self.arr1 if which==1 else self.arr2
        emb=self.e1   if which==1 else self.e2
        dc =self.dc1  if which==1 else self.dc2
        for name in REGIONS:
            occ=occlude(arr,name); eo=get_emb(occ)
            blocks=window_sign_match(emb,eo)   # True=window tetap, False=window berubah
            dc[name]={"emb":eo,"blocks":blocks}
        if self.custom_mask.get(which) is not None:
            self._compute_custom_for(which)
        self.after(0,self._draw_multi)
        self.after(0,self._on_region)

    # ── REFRESH MODE ─────────────────────────────────────
    def _refresh(self):
        mode=self.mode.get()
        if mode=="compare":
            self.abl_frame.pack_forget()
            self.cmp_frame.pack(fill="both",expand=True)
            self._draw_compare()
        else:
            self.cmp_frame.pack_forget()
            self.abl_frame.pack(fill="both",expand=True)
            self._on_region()

    # ── COMPARE DRAWS ─────────────────────────────────────
    def _draw_compare(self):
        if self.e1 is None or self.e2 is None: return
        e1,e2=self.e1,self.e2

        # Scores
        sg=img_sign_score(e1,e2); ap=amp_score(e1,e2)
        cs,nc,ac=chain_score(e1,e2); co=cosine(e1,e2)
        self.m_sign.config(text=f"{sg:.3f}")
        self.m_amp.config(text=f"{ap:.3f}")
        self.m_chain.config(text=f"{cs:.3f}")
        self.m_cos.config(text=f"{co:.3f}")
        self.chain_lbl.config(
            text=f"Chains: {nc}  AvgLen: {ac:.1f}  (neutral={NEUTRAL_LEN})")

        thr=0.79; npass=sum([sg>=thr,ap>=thr,cs>=thr])
        if npass>=2:
            self.verdict_lbl.config(text="✅  MATCH",fg=WHITE,bg="#064e3b",
                                     highlightbackground=GREEN)
        elif npass==1:
            self.verdict_lbl.config(text="⚠️  UNCERTAIN",fg=WHITE,bg="#78350f",
                                     highlightbackground=ORANGE)
        else:
            self.verdict_lbl.config(text="❌  DIFFERENT",fg=WHITE,bg="#450a0a",
                                     highlightbackground=RED)

        # Embedding bar
        self._draw_emb_bars()
        self._draw_sign_pattern()
        self._draw_chain_pattern()
        self._draw_emb_single(self.c_emb1,self.e1,BLUE,"Foto 1")
        self._draw_emb_single(self.c_emb2,self.e2,GREEN,"Foto 2")

    def _draw_emb_bars(self):
        c=self.c_emb; c.delete("all")
        W=self._get_w(c); H=180
        n=len(self.e1); bw=W/n; mid=H//2
        c.create_line(0,mid,W,mid,fill=BORDER,width=1,dash=(3,2))
        for i in range(n):
            x0=i*bw; x1=x0+bw-0.3
            v1=float(self.e1[i]); h1=abs(v1)*(mid-4)
            if v1>=0: c.create_rectangle(x0,mid-h1,x1,mid,fill=BLUE,outline="")
            else:     c.create_rectangle(x0,mid,x1,mid+h1,fill=BLUE,outline="")
            v2=float(self.e2[i]); h2=abs(v2)*(mid-4)*0.7
            if v2>=0: c.create_rectangle(x0,mid-h2,x1,mid,fill=GREEN,outline="",stipple="gray25")
            else:     c.create_rectangle(x0,mid,x1,mid+h2,fill=GREEN,outline="",stipple="gray25")
        c.create_text(4,4,anchor="nw",
            text="Biru=Foto1  Hijau=Foto2  (stipple)  — dimensi yang searah = sign cocok",
            font=("Courier",7),fill=SUB)

    def _draw_emb_single(self,canvas,emb,color,label):
        """Grafik embedding individual (1024D) untuk satu foto — dipakai di kedua canvas bawah."""
        canvas.delete("all")
        if emb is None:
            W=self._get_w(canvas); H=60
            canvas.create_text(W//2,H//2,text=f"Upload {label}...",
                               font=("Courier",9),fill=SUB)
            return
        W=self._get_w(canvas); H=60
        n=len(emb); bw=W/n; mid=H//2
        canvas.create_line(0,mid,W,mid,fill=BORDER,width=1,dash=(3,2))
        for i in range(n):
            x0=i*bw; x1=x0+bw-0.3
            v=float(emb[i]); h=abs(v)*(mid-3)
            if v>=0: canvas.create_rectangle(x0,mid-h,x1,mid,fill=color,outline="")
            else:    canvas.create_rectangle(x0,mid,x1,mid+h,fill=color,outline="")
        canvas.create_text(4,2,anchor="nw",
            text=f"{label}  min={emb.min():.2f}  max={emb.max():.2f}  mean={emb.mean():.3f}",
            font=("Courier",7),fill=SUB)

    def _draw_sign_pattern(self):
        c=self.c_sign; c.delete("all")
        W=self._get_w(c); H=80
        e1,e2=self.e1,self.e2; n=len(e1)-WINDOW_SIZE+1; bw=W/n
        for i in range(n):
            mc=sum(1 for j in range(WINDOW_SIZE)
                   if (e1[i+j]>=0)==(e2[i+j]>=0))
            ratio=mc/WINDOW_SIZE; h=ratio*(H-4)
            col=GREEN if mc>=THRESHOLD else RED
            x0=i*bw
            c.create_rectangle(x0,H-h,x0+bw-0.3,H,fill=col,outline="")
        c.create_line(0,H-(THRESHOLD/WINDOW_SIZE)*(H-4),
                      W,H-(THRESHOLD/WINDOW_SIZE)*(H-4),
                      fill=YELLOW,width=1,dash=(4,2))
        c.create_text(4,4,anchor="nw",
            text=f"Sign match ratio per window  (thr={THRESHOLD}/{WINDOW_SIZE} = garis kuning)",
            font=("Courier",7),fill=SUB)

    def _draw_chain_pattern(self):
        c=self.c_chain; c.delete("all")
        W=self._get_w(c); H=70
        e1,e2=self.e1,self.e2; n=len(e1)-WINDOW_SIZE+1; bw=W/n
        in_chain=False; chain_start=0; chain_num=0
        for i in range(n):
            mc=sum(1 for j in range(WINDOW_SIZE)
                   if (e1[i+j]>=0)==(e2[i+j]>=0))
            match=mc>=THRESHOLD; x0=i*bw
            if match:
                c.create_rectangle(x0,10,x0+bw-0.3,H-10,fill=TEAL,outline="")
                if not in_chain:
                    chain_start=i; in_chain=True; chain_num+=1
                    c.create_line(x0,4,x0,H-4,fill=WHITE,width=1)
                    c.create_text(x0+1,6,anchor="nw",
                        text=str(chain_num),font=("Courier",6),fill=WHITE)
            else:
                in_chain=False
        c.create_text(4,4,anchor="nw",
            text=f"Chain pattern — teal=match, garis putih=awal chain baru, angka=nomor chain",
            font=("Courier",7),fill=SUB)

    # ── ABLATION: BACA ULANG SLIDING WINDOW ───────────────
    def _switch_custom_photo(self, which):
        self.custom_active_photo=which
        self._update_custom_toggle_style()
        self._draw_custom_canvas_bg()

    def _update_custom_toggle_style(self):
        active=self.custom_active_photo
        p1_done=self.custom_mask.get(1) is not None
        p2_done=self.custom_mask.get(2) is not None
        self.btn_custom_p1.config(
            bg=(BLUE if active==1 else CARD), fg=WHITE if active==1 else BLUE,
            relief=("sunken" if active==1 else "flat"),
            text=("✎ Gambar Foto 1 ✓" if p1_done else "✎ Gambar Foto 1"))
        self.btn_custom_p2.config(
            bg=(GREEN if active==2 else CARD), fg=WHITE if active==2 else GREEN,
            relief=("sunken" if active==2 else "flat"),
            text=("✎ Gambar Foto 2 ✓" if p2_done else "✎ Gambar Foto 2"))

    def _draw_custom_canvas_bg(self):
        """Gambar ulang background (foto yg lagi aktif di-edit) di kanvas gambar polygon."""
        self.c_custom_draw.delete("all")
        cs=IMG_SIZE*self.custom_scale
        which=self.custom_active_photo
        arr=self.arr1 if which==1 else self.arr2
        if arr is not None:
            img=Image.fromarray(arr.astype(np.uint8)).resize((cs,cs),Image.NEAREST)
            tkimg=ImageTk.PhotoImage(img)
            self.c_custom_draw.create_image(0,0,anchor="nw",image=tkimg)
            self.c_custom_draw.image=tkimg
        else:
            self.c_custom_draw.create_text(cs//2,cs//2,text=f"Upload Foto {which} dulu",
                                           font=("Courier",10),fill=SUB)
        self._redraw_custom_points()

    def _redraw_custom_points(self):
        self.c_custom_draw.delete("pt")
        sc=self.custom_scale
        which=self.custom_active_photo
        pts_orig=self.custom_points[which]
        pts=[(x*sc,y*sc) for (x,y) in pts_orig]
        col=BLUE if which==1 else GREEN
        for i,(x,y) in enumerate(pts):
            self.c_custom_draw.create_oval(x-4,y-4,x+4,y+4,fill=col,outline=WHITE,tags="pt")
            self.c_custom_draw.create_text(x+8,y-8,text=str(i+1),fill=WHITE,
                                           font=("Courier",8,"bold"),tags="pt")
        for i in range(len(pts)-1):
            x0,y0=pts[i]; x1,y1=pts[i+1]
            self.c_custom_draw.create_line(x0,y0,x1,y1,fill=col,width=2,tags="pt")
        if len(pts)==6:
            x0,y0=pts[0]; x5,y5=pts[5]
            self.c_custom_draw.create_line(x5,y5,x0,y0,fill=col,width=2,dash=(4,2),tags="pt")
        self.custom_status_lbl.config(
            text=f"Foto{which}: {len(pts_orig)}/6 titik")
        self.btn_apply_custom.config(state=("normal" if len(pts_orig)==6 else "disabled"))

    def _on_custom_click(self, event):
        which=self.custom_active_photo
        if len(self.custom_points[which])>=6: return
        sc=self.custom_scale
        ox=min(max(event.x/sc,0),IMG_SIZE-1)
        oy=min(max(event.y/sc,0),IMG_SIZE-1)
        self.custom_points[which].append((ox,oy))
        self._redraw_custom_points()

    def _reset_custom_points(self):
        which=self.custom_active_photo
        self.custom_points[which]=[]
        self._redraw_custom_points()

    def _compute_custom_for(self, which):
        """Hitung ulang embedding+blocks utk region CUSTOM_KEY satu foto,
        pakai self.custom_mask[which] milik foto itu sendiri (independen dari
        foto satunya). Aman dipanggil dari thread background (cuma nulis ke
        dict, tidak sentuh widget)."""
        mask=self.custom_mask.get(which)
        if mask is None: return
        arr=self.arr1 if which==1 else self.arr2
        emb=self.e1   if which==1 else self.e2
        dc =self.dc1  if which==1 else self.dc2
        if arr is None or emb is None: return
        occ=occlude_mask(arr,mask)
        eo=get_emb(occ)
        dc[CUSTOM_KEY]={"emb":eo,"blocks":window_sign_match(emb,eo)}

    def _apply_custom_occlusion(self):
        which=self.custom_active_photo
        pts=self.custom_points[which]
        if len(pts)!=6: return
        self.custom_mask[which]=polygon_to_mask(pts)
        self._compute_custom_for(which)
        self._update_custom_toggle_style()
        self._draw_multi()
        self._on_region()

    def _on_region(self,*_):
        if self.mode.get()!="ablation": return
        name=self.sel.get()
        self.region_active_lbl.config(text=f"Region aktif: {name}")
        if name==CUSTOM_KEY:
            self.custom_draw_frame.pack(fill="x",padx=8,pady=(2,4),before=self.abl_d1_title)
            self._draw_custom_canvas_bg()
            if self.arr1 is not None:
                self._show(occlude_mask(self.arr1,self.custom_mask.get(1)),self.c1,120)
            if self.arr2 is not None:
                self._show(occlude_mask(self.arr2,self.custom_mask.get(2)),self.c2,120)
        else:
            self.custom_draw_frame.pack_forget()
            if self.arr1 is not None: self._show(occlude(self.arr1,name),self.c1,120,region=name)
            if self.arr2 is not None: self._show(occlude(self.arr2,name),self.c2,120,region=name)
        self._draw_window_reread(name)
        self._draw_consistency(name)

    def _draw_window_reread(self,name):
        """Untuk tiap foto: gambar embedding asli vs terdampak (garis), lalu di
        bawahnya bar blok per sliding window (hijau=window tetap, merah=window
        berubah). Breakdown ditampilkan (jumlah window berubah, jumlah segmen,
        rata-rata panjang segmen) — bukan cuma satu angka skor akhir."""
        show_asli      = self.v_show_asli.get()
        show_terdampak = self.v_show_terdampak.get()
        show_blocks    = self.v_show_blocks.get()
        show_delta     = self.v_show_delta.get()

        pairs = [
            (self.c_d1, self.c_blk1, self.dc1, self.e1, self.abl_d1_title, 1),
            (self.c_d2, self.c_blk2, self.dc2, self.e2, self.abl_d2_title, 2),
        ]

        for line_canvas, blk_canvas, dc, orig_emb, title_lbl, which in pairs:
            line_canvas.delete("all"); blk_canvas.delete("all")
            Wl=self._get_w(line_canvas); Hl=90
            Wb=self._get_w(blk_canvas)

            if name not in dc or orig_emb is None:
                msg=("Gambar 6 titik & klik Terapkan Oklusi dulu"
                     if name==CUSTOM_KEY else f"Menghitung Foto{which}...")
                line_canvas.create_text(Wl//2,Hl//2,text=msg,
                                        font=("Courier",9),fill=SUB)
                continue

            occ_emb = dc[name]["emb"]
            blocks  = dc[name]["blocks"]     # True=tetap, False=berubah
            n_dim   = len(orig_emb)
            n_win   = len(blocks)

            # ── Hitung delta per-dimensi & spike besar ──
            delta = orig_emb.astype(np.float64) - occ_emb.astype(np.float64)
            abs_delta = np.abs(delta)
            mu, sd = float(abs_delta.mean()), float(abs_delta.std())
            spike_thr = mu + 1.5*sd if sd > 1e-9 else abs_delta.max()+1
            spike_idx = np.where(abs_delta >= spike_thr)[0]
            d_max = float(abs_delta.max()) if abs_delta.size else 1e-8

            # ── Baris atas: nilai embedding asli vs terdampak, full 1024D ──
            mid=Hl/2
            g_max=max(float(np.abs(orig_emb).max()),float(np.abs(occ_emb).max()))+1e-8
            bw_d=Wl/n_dim

            # spike bands digambar dulu (di belakang garis) — makin tinggi delta
            # makin pekat oranye, tanda arah: naik keatas jika asli>terdampak,
            # kebawah jika terdampak>asli
            if show_delta:
                for i in spike_idx:
                    x0=i*bw_d
                    t=min(1.0,(abs_delta[i]-spike_thr)/max(d_max-spike_thr,1e-8))
                    band_col = lerp_color(0.15+0.85*t, c1="#3a2a00", c2=YELLOW)
                    line_canvas.create_rectangle(x0,4,x0+bw_d+0.6,Hl-4,
                                                 fill=band_col,outline="")

            line_canvas.create_line(0,mid,Wl,mid,fill=BORDER,width=1,dash=(3,2))
            if show_asli:
                pts=[(i*bw_d, mid-(orig_emb[i]/g_max)*(mid-4)) for i in range(n_dim)]
                for i in range(len(pts)-1):
                    line_canvas.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],
                                            fill=BLUE,width=1)
            if show_terdampak:
                pts=[(i*bw_d, mid-(occ_emb[i]/g_max)*(mid-4)) for i in range(n_dim)]
                for i in range(len(pts)-1):
                    line_canvas.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],
                                            fill=RED,width=1)

            # marker segitiga di puncak tiap spike: ▲ oranye kalau asli>terdampak
            # (delta positif, dimensi "turun" akibat oklusi), ▼ ungu kalau
            # terdampak>asli (delta negatif, dimensi "naik" akibat oklusi)
            if show_delta:
                for i in spike_idx:
                    xc=i*bw_d+bw_d/2
                    yo=mid-(orig_emb[i]/g_max)*(mid-4)
                    yt=mid-(occ_emb[i]/g_max)*(mid-4)
                    y_ext = min(yo,yt)-6   # titik ekstrem paling atas
                    if delta[i] >= 0:
                        # asli lebih tinggi/lebih positif -> segitiga naik, oranye
                        line_canvas.create_polygon(
                            xc,y_ext-6, xc-4,y_ext, xc+4,y_ext,
                            fill=ORANGE, outline="")
                    else:
                        y_ext2 = max(yo,yt)+6
                        line_canvas.create_polygon(
                            xc,y_ext2+6, xc-4,y_ext2, xc+4,y_ext2,
                            fill=PURPLE, outline="")

            n_spike=len(spike_idx)
            line_canvas.create_text(4,2,anchor="nw",
                text=f"Foto{which} [{name}]  Biru=Asli  Merah=Terdampak  "
                     f"Kuning=zona delta besar  ▲/▼=spike ({n_spike} dim, thr≈{spike_thr:.3f})",
                font=("Courier",7),fill=SUB)

            # ── Baris bawah: hasil BACA ULANG tiap sliding window, di-encode
            #    pakai AMPLITUDO (persis rumus amp_score) — tinggi & warna bar
            #    = amp_sim (hijau tinggi=amplitudo mirip, merah pendek=jomplang
            #    atau sign gagal). Titik oranye = window sign-nya MATCH tapi
            #    amplitudonya anomali (di atas ambang adaptif mean+1.5*std,
            #    dihitung hanya dari window yang match). ──
            n_changed=int(np.sum(~blocks))
            segs=changed_segments(blocks)
            avg_len=(n_changed/len(segs)) if segs else 0.0

            gate,amp_ratio,combined=soft_window_match(orig_emb,occ_emb)  # dipakai utk bar visual (soft)
            amp_sim=np.where(blocks,amp_ratio,0.0)  # basis anomali (hard-gated), diturunkan dari amp_ratio
            matched_idx=np.where(blocks)[0]
            if matched_idx.size>0:
                amp_dev=1-amp_sim[matched_idx]
                mu_a,sd_a=float(amp_dev.mean()),float(amp_dev.std())
                amp_thr=mu_a+1.5*sd_a if sd_a>1e-9 else float(amp_dev.max())+1.0
                avg_amp=float(amp_sim[matched_idx].mean())
                anomaly_idx=[i for i in matched_idx if (1-amp_sim[i])>=amp_thr]
            else:
                amp_thr=2.0; avg_amp=0.0; anomaly_idx=[]

            if show_blocks:
                Hb=50; bw_w=Wb/n_win
                for i in range(n_win):
                    x0=i*bw_w; sim=combined[i]
                    h=max(2.0,sim*(Hb-4))
                    col=lerp_color(sim,c1=RED,c2=GREEN)
                    blk_canvas.create_rectangle(x0,Hb-h,x0+bw_w+0.5,Hb,fill=col,outline="")
                for i in anomaly_idx:
                    xc=i*bw_w+bw_w/2; h=max(2.0,combined[i]*(Hb-4))
                    blk_canvas.create_oval(xc-3,Hb-h-9,xc+3,Hb-h-3,fill=ORANGE,outline="")
                blk_canvas.create_text(Wb-4,2,anchor="ne",
                    text="tinggi/warna=soft gate×amp  •=anomali amplitudo (hard)",
                    font=("Courier",6),fill=SUB)

            blk_canvas.create_text(4,2,anchor="nw",
                text=f"Berubah:{n_changed}/{n_win}({n_changed/max(n_win,1)*100:.0f}%) "
                     f"seg:{len(segs)} avglen:{avg_len:.1f} | "
                     f"AMP avg:{avg_amp:.2f} anomali:{len(anomaly_idx)}",
                font=("Courier",7,"bold"),fill=WHITE)

            title_lbl.config(
                text=f"Foto{which} [{name}] — Baca Ulang Sliding Window  "
                     f"(berubah: {n_changed}/{n_win} = {n_changed/max(n_win,1)*100:.1f}%  "
                     f"spike delta: {n_spike}  amp avg: {avg_amp:.2f}  amp anomali: {len(anomaly_idx)})")

    def _draw_multi(self):
        """Bandingkan semua region: berapa persen window yang berubah kalau
        region tsb ditutup. Biru=Foto1, Hijau=Foto2 — makin tinggi bar makin
        sensitif region itu terhadap oklusi."""
        c=self.c_multi; c.delete("all")
        if not self.dc1 and not self.dc2: return
        W=self._get_w(c); H=150
        names=list(REGIONS.keys())
        if CUSTOM_KEY in self.dc1 or CUSTOM_KEY in self.dc2:
            names=names+[CUSTOM_KEY]
        n=len(names)
        slot=W/n; bar_w=slot*0.32; base=H-28

        c.create_line(0,base,W,base,fill=BORDER)
        for idx,name in enumerate(names):
            xc=idx*slot+slot/2
            col=REGION_COLS[idx%len(REGION_COLS)]
            if name in self.dc1:
                r1=np.sum(~self.dc1[name]["blocks"])/len(self.dc1[name]["blocks"])
                h1=r1*base
                c.create_rectangle(xc-bar_w,base-h1,xc,base,fill=BLUE,outline="")
                c.create_text(xc-bar_w/2,base-h1-2,text=f"{r1*100:.0f}",
                              anchor="s",font=("Courier",6),fill=BLUE)
            if name in self.dc2:
                r2=np.sum(~self.dc2[name]["blocks"])/len(self.dc2[name]["blocks"])
                h2=r2*base
                c.create_rectangle(xc,base-h2,xc+bar_w,base,fill=GREEN,outline="")
                c.create_text(xc+bar_w/2,base-h2-2,text=f"{r2*100:.0f}",
                              anchor="s",font=("Courier",6),fill=GREEN)
            c.create_text(xc,base+4,text=name.split()[0],anchor="n",
                          font=("Courier",6),fill=col)
        c.create_text(W-4,4,anchor="ne",
            text="Biru=Foto1  Hijau=Foto2  (angka=% window berubah per region)",
            font=("Courier",7),fill=SUB)

    def _draw_consistency(self,name):
        """Bandingkan blok yang berubah pada region terpilih antara Foto1 dan
        Foto2 (bukan kurva delta yang di-smooth). Konsistensi diukur dengan
        Jaccard overlap dari posisi window yang sama-sama berubah."""
        c=self.c_consist; c.delete("all")
        W=self._get_w(c); H=110
        if name not in self.dc1 or name not in self.dc2:
            c.create_text(W//2,H//2,text="Tunggu precompute...",
                         font=("Courier",9),fill=SUB); return
        b1=self.dc1[name]["blocks"]; b2=self.dc2[name]["blocks"]
        n=min(len(b1),len(b2)); bw=W/n

        row1_top,row1_bot = 8,44
        row2_top,row2_bot = 54,90

        for i in range(n):
            x0=i*bw
            c.create_rectangle(x0,row1_top,x0+bw+0.5,row1_bot,
                               fill=(RED if not b1[i] else "#0f2418"),outline="")
            c.create_rectangle(x0,row2_top,x0+bw+0.5,row2_bot,
                               fill=(ORANGE if not b2[i] else "#0f2418"),outline="")

        changed1=~b1[:n]; changed2=~b2[:n]
        inter=int(np.sum(changed1 & changed2))
        union=int(np.sum(changed1 | changed2))
        jacc=inter/union if union>0 else 0.0
        col_j=GREEN if jacc>0.5 else (YELLOW if jacc>0.2 else RED)

        c.create_text(4,H-2,anchor="sw",
            text=f"Atas=Foto1(merah=berubah)  Bawah=Foto2(oranye=berubah)  overlap={inter}/{union}",
            font=("Courier",7),fill=SUB)
        c.create_text(W-4,4,anchor="ne",
            text=f"Konsistensi (Jaccard): {jacc:.3f}",
            font=("Courier",9,"bold"),fill=col_j)

        sim1=img_sign_score(self.e1,self.dc1[name]["emb"]) if self.e1 is not None else 0
        sim2=img_sign_score(self.e2,self.dc2[name]["emb"]) if self.e2 is not None else 0
        n1=int(np.sum(changed1)); n2=int(np.sum(changed2))
        self.abl_lbl.config(
            text=f"[{name}]\n"
                 f"Foto1: {n1}/{n} window berubah ({n1/max(n,1)*100:.1f}%)  sign-sim akhir={sim1:.4f}\n"
                 f"Foto2: {n2}/{n} window berubah ({n2/max(n,1)*100:.1f}%)  sign-sim akhir={sim2:.4f}\n"
                 f"Overlap posisi berubah (Jaccard): {jacc:.3f}\n"
                 f"({'Konsisten ✓' if jacc>0.4 else 'Tidak konsisten ✗'})",
            fg=GREEN if jacc>0.4 else RED)


if __name__=="__main__":
    app=App()
    app.after(200, app._redraw_all)  # trigger setelah window render
    app.mainloop()
