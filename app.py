import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "model_pretrained_mobilenetv2.h5"
IMG_SIZE = (128, 128)          # sesuai input model
CLASS_NAMES = ["Apple", "Orange"]  # index 0 -> Apple, index 1 -> Orange
# Catatan: urutan ini mengikuti urutan folder alfabetis (apple, orange)
# saat training. Kalau hasil prediksi kebalik, tinggal tukar urutan list ini.

st.set_page_config(page_title="Apple vs Orange Classifier", page_icon="🍎🍊", layout="centered")

st.markdown("""
<style>
h1 { color: #2F4B3C; }
.stButton>button { background: #2F4B3C; color: #FFFFFF; border: none; }
.stButton>button:hover { background: #22362A; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL (cache supaya tidak reload tiap interaksi)
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ============================================================
# UI
# ============================================================
st.title("Apple vs Orange Classifier")
st.write("Upload gambar apel atau jeruk, lalu model akan memprediksi kelasnya.")

uploaded_file = st.file_uploader(
    "Pilih gambar...", type=["jpg", "jpeg", "png"],
    help="Gunakan foto satu buah dengan latar belakang polos untuk hasil terbaik.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):
        with st.spinner("Model sedang memproses..."):
            processed = preprocess_image(image)
            pred = model.predict(processed)[0][0]  # output sigmoid, skalar 0-1

            # threshold 0.5: >0.5 -> kelas index 1 (Orange), <=0.5 -> kelas index 0 (Apple)
            pred_class = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
            confidence = pred if pred > 0.5 else 1 - pred

        st.success(f"Prediksi: **{pred_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
else:
    st.info("Silakan upload gambar terlebih dahulu.")

st.markdown("---")
st.caption("Model transfer learning MobileNetV2.")
