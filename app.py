import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

APP_NAME = "SonoTriage"
TAGLINE = "AI that triages breast ultrasounds, explained."
MODEL_PATH = "busi_model.pt"
CLASS_NAMES = ["benign", "malignant", "normal"]
LOW_CONF = 0.60

MODEL_FACTS = {
    "Architecture": "EfficientNet-B0 (transfer learning + fine-tuning)",
    "Training data": "BUSI dataset, about 780 images",
    "Test accuracy": "about 83%",
    "AUC (one-vs-rest)": "about 0.94",
    "Malignant recall": "about 87%",
}

CLASS_INFO = {
    "normal": {
        "title": "Normal",
        "emoji": "🟢",
        "color": "#16a34a",
        "msg": "No suspicious mass pattern was detected in this image. "
               "If you have symptoms or concerns, please still see a doctor.",
    },
    "benign": {
        "title": "Benign-looking",
        "emoji": "🟡",
        "color": "#d97706",
        "msg": "This image resembles a benign (non-cancerous) lesion. "
               "A clinician should still confirm and decide on follow-up.",
    },
    "malignant": {
        "title": "Suspicious (malignant-looking)",
        "emoji": "🔴",
        "color": "#dc2626",
        "msg": "This image has features resembling a malignant lesion. "
               "Please consult a radiologist or doctor promptly for further evaluation.",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title=APP_NAME, page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
        padding: 2rem 2.2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.92; color: white;}
    .card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 16px;
        padding: 1.4rem 1.5rem; background: rgba(128,128,128,0.06);
    }
    .badge {
        display: inline-block; padding: 0.45rem 1rem; border-radius: 999px;
        color: white; font-weight: 700; font-size: 1.05rem;
    }
    .conf {font-size: 2.6rem; font-weight: 800; margin: 0.6rem 0 0.2rem 0; line-height: 1;}
    .muted {opacity: 0.7; font-size: 0.9rem;}
    .footer {text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


class BreastUltrasoundModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


@st.cache_resource
def load_model():
    model = BreastUltrasoundModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, img):
    x = test_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return probs


def prob_bar(name, p, color):
    return f"""
    <div style="margin:10px 0">
        <div style="display:flex;justify-content:space-between;font-size:0.92rem">
            <span>{name}</span><span><b>{p*100:.1f}%</b></span>
        </div>
        <div style="background:rgba(128,128,128,0.2);border-radius:8px;height:10px">
            <div style="width:{p*100:.1f}%;background:{color};height:10px;border-radius:8px"></div>
        </div>
    </div>
    """


with st.sidebar:
    st.markdown(f"## 🩺 {APP_NAME}")
    st.caption(TAGLINE)
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a breast ultrasound image\n"
        "2. The model classifies it as **normal**, **benign** or **malignant-looking**\n"
        "3. You see the confidence for every class"
    )
    st.markdown("---")
    st.markdown("### Model details")
    for k, v in MODEL_FACTS.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown("---")
    st.markdown("### Limitations")
    st.markdown(
        "- Trained on a small public dataset\n"
        "- Not clinically validated\n"
        "- Not a diagnosis; biopsy and a doctor decide"
    )

st.markdown(
    f"""
    <div class="hero">
        <h1>🩺 {APP_NAME}</h1>
        <p>{TAGLINE} A free decision-support tool for breast ultrasound screening.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Place it in the same folder as app.py.")
    st.stop()

model = load_model()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 1. Upload ultrasound image")
    file = st.file_uploader(
        "PNG or JPG", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )
    if file is not None:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=file.name)
    else:
        st.info("Drag and drop an image here, or click Browse files.")

with right:
    st.markdown("### 2. Result")
    if file is None:
        st.markdown(
            '<div class="card"><span class="muted">'
            "Results will appear here after you upload an image.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing image..."):
            probs = predict(model, img)

        pred = int(probs.argmax())
        name = CLASS_NAMES[pred]
        info = CLASS_INFO[name]
        conf = float(probs[pred])

        st.markdown(
            f"""
            <div class="card">
                <span class="badge" style="background:{info['color']}">{info['emoji']} {info['title']}</span>
                <div class="conf">{conf*100:.1f}%</div>
                <div class="muted">model-estimated probability for this class</div>
                <p style="margin-top:1rem">{info['msg']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if conf < LOW_CONF:
            st.warning(
                "Low confidence: the model is unsure about this image. "
                "Treat this result with extra caution."
            )

        st.markdown("#### Probability breakdown")
        bars = ""
        for cname, p in zip(CLASS_NAMES, probs):
            bars += prob_bar(CLASS_INFO[cname]["title"], float(p), CLASS_INFO[cname]["color"])
        st.markdown(bars, unsafe_allow_html=True)

st.markdown("---")
st.warning(
    "**Medical disclaimer:** This tool is for screening support and education only. "
    "It is not a diagnosis and does not replace a doctor. "
    "Please consult a qualified healthcare professional for actual diagnosis."
)

st.markdown(
    f'<div class="footer">{APP_NAME} · Built for the CodeTheCure Hackathon 2026</div>',
    unsafe_allow_html=True,
)