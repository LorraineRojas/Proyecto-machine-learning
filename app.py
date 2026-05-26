import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import torch
import timm
import os
from PIL import Image

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Writer Identification",
    page_icon="✍️",
    layout="centered",
)

# ──────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background-color: #F5F0E8;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(180,160,120,0.12) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(100,80,60,0.08) 0%, transparent 50%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 720px; }

.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: #8B7355; margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem; line-height: 1.1; color: #1C1410; margin: 0 0 0.4rem;
}
.hero-title em { font-style: italic; color: #6B4F2A; }
.hero-subtitle {
    font-size: 0.95rem; color: #6B5E4E; font-weight: 300;
    max-width: 420px; margin: 0 auto; line-height: 1.6;
}
.ink-divider {
    text-align: center; color: #C4A882; font-size: 1.4rem;
    letter-spacing: 0.5rem; margin: 1.2rem 0;
}
[data-testid="stFileUploader"] > div {
    background: #FBF7EF !important; border: 2px dashed #C4A882 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] > div:hover { border-color: #6B4F2A !important; }
.stButton > button {
    width: 100%; background: #3D2B1A !important; color: #F5F0E8 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'DM Serif Display', serif !important; font-size: 1.1rem !important;
    letter-spacing: 0.04em !important; padding: 0.75rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(61,43,26,0.25) !important; margin-top: 0.5rem;
}
.stButton > button:hover {
    background: #6B4F2A !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(61,43,26,0.35) !important;
}
.result-card {
    background: #1C1410; border-radius: 16px; padding: 2rem 2.2rem;
    margin-top: 1.4rem; box-shadow: 0 8px 40px rgba(28,20,16,0.25);
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(180,140,80,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.result-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: #9C8060; margin-bottom: 0.4rem;
}
.result-writer {
    font-family: 'DM Serif Display', serif; font-size: 3.6rem;
    color: #F5F0E8; line-height: 1; margin-bottom: 0.2rem;
}
.result-writer.unknown { color: #C4A882; font-style: italic; }
.result-conf-row { display: flex; align-items: center; gap: 0.8rem; margin-top: 1rem; }
.conf-bar-bg {
    flex: 1; height: 6px; background: rgba(255,255,255,0.1);
    border-radius: 3px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #C4A882, #F5D8A0);
}
.conf-pct {
    font-family: 'DM Mono', monospace; font-size: 0.85rem;
    color: #C4A882; min-width: 44px; text-align: right;
}
.result-model-tag {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6B5040; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;
    padding: 0.2rem 0.5rem; margin-top: 0.9rem;
}
.unknown-note { font-size: 0.82rem; color: #9C8060; margin-top: 0.6rem; font-style: italic; }
.info-box {
    background: #FFF8EE; border: 1.5px solid #D4C4A8; border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.83rem; color: #6B5040;
    margin-bottom: 1rem; line-height: 1.6;
}
.err-box {
    background: #FFF0EC; border: 1.5px solid #E8B4A0; border-radius: 10px;
    padding: 1rem 1.2rem; color: #8B3A20; font-size: 0.88rem; margin-top: 1rem;
}
[data-testid="stImage"] img { border-radius: 10px; border: 1.5px solid #D4C4A8; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "effnet_b0_fold0.pth")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE      = 224
NUM_CLASSES   = 36

LABEL_MAP = {i: f"Writer #{i}" for i in range(NUM_CLASSES)}


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return m.eval()


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    img  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    s    = max(h, w)
    canvas = np.ones((s, s), dtype=np.uint8) * 255
    canvas[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = gray
    resized = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img3 = np.stack([resized]*3, axis=-1).astype(np.float32) / 255.0
    img3 = (img3 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img3.transpose(2, 0, 1)).unsqueeze(0).float()


def run_inference(pil_image: Image.Image, threshold: float):
    tensor = preprocess(pil_image)
    with torch.no_grad():
        out  = load_model()(tensor)
        prob = torch.softmax(out, dim=1).numpy()[0]
    conf   = float(prob.max())
    idx    = int(prob.argmax())
    writer = LABEL_MAP[idx] if conf >= threshold else "-1"
    return writer, conf


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">ISIS4219 2026-1</div>
    <div class="hero-title">Writer<br><em>Identification</em></div>
    <div class="hero-subtitle">
        Sube una imagen de escritura y el modelo identificará al escritor.
    </div>
</div>
<div class="ink-divider">· · ·</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    El <b>umbral de confianza</b> controla cuándo el modelo dice "desconocido"
    en vez de forzar una predicción. Si ves muchos resultados desconocidos, bájalo;
    si ves predicciones erradas, súbelo.
</div>
""", unsafe_allow_html=True)

threshold = st.slider("Umbral de confianza", min_value=0.10, max_value=0.95,
                      value=0.50, step=0.05)

uploaded = st.file_uploader("Imagen de escritura", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded:
    pil_img = Image.open(uploaded)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(pil_img, use_container_width=True)
    with c2:
        identificar = st.button("✦  Identificar\nescritor")

    if identificar:
        if not os.path.exists(MODEL_PATH):
            st.markdown(
                '<div class="err-box">⚠️ Modelo no encontrado: '
                '<code>models/effnet_b0_fold0.pth</code></div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Analizando trazos..."):
                try:
                    writer, conf = run_inference(pil_img, threshold)
                    pct = int(conf * 100)

                    if writer == "-1":
                        name_block = '<div class="result-writer unknown">Desconocido</div>'
                        note_block = (
                            '<div class="unknown-note">'
                            f'Confianza ({pct}%) por debajo del umbral ({int(threshold*100)}%). '
                            'El escritor puede no estar en el conjunto de entrenamiento, '
                            'o intenta bajar el umbral.'
                            '</div>'
                        )
                    else:
                        name_block = f'<div class="result-writer">{writer}</div>'
                        note_block = ''

                    html = (
                        '<div class="result-card">'
                        '<div class="result-label">Escritor identificado</div>'
                        + name_block + note_block
                        + '<div class="result-conf-row">'
                        '<div class="conf-bar-bg">'
                        f'<div class="conf-bar-fill" style="width:{pct}%"></div>'
                        '</div>'
                        f'<div class="conf-pct">{pct}%</div>'
                        '</div>'
                        '<div class="result-model-tag">Modelo · EfficientNet-B0</div>'
                        '</div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)
                    components.html("""
                    <script>
                        window.parent.document.querySelector(
                            '[data-testid="stVerticalBlock"]'
                        ).lastElementChild.scrollIntoView({behavior: "smooth", block: "start"});
                    </script>
                    """, height=0)

                except Exception as e:
                    st.markdown(
                        f'<div class="err-box">❌ Error: <code>{e}</code></div>',
                        unsafe_allow_html=True
                    )
