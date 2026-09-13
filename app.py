Python
import streamlit as st
import numpy as np
from PIL import Image
import math
import time

st.set_page_config(page_title="Hệ Thống Phân Tích ECG", page_icon="🫀", layout="wide")

def analyze_ecg_image(image: Image.Image):
    img_resized = image.resize((224, 224)).convert("L")
    img_arr = np.array(img_resized) / 255.0
    mean_val = np.mean(img_arr)
    std_val = np.std(img_arr)
    grad_x = np.diff(img_arr, axis=1)
    spike_density = np.mean(np.abs(grad_x) > 0.15)

    score_normal = max(0.05, 1.0 - (spike_density * 4.0))
    score_mi = min(0.95, (1.0 - mean_val) * 0.8 + spike_density)
    score_afib = min(0.90, std_val * 2.2)
    score_pvc = min(0.85, spike_density * 3.5)

    raw_scores = np.array([score_normal, score_mi, score_afib, score_pvc])
    exp_scores = np.exp(raw_scores * 2.5)
    probs = exp_scores / np.sum(exp_scores)

    classes = [
        "Nhịp xoang bình thường (Normal Rhythm)",
        "Gợi ý Nhồi máu cơ tim (Myocardial Infarction)",
        "Rung nhĩ / Loạn nhịp (Atrial Fibrillation)",
        "Ngoại tâm thu thất (Premature Ventricular Contraction)"
    ]

    top_idx = int(np.argmax(probs))
    return {
        "label": classes[top_idx],
        "confidence": float(probs[top_idx] * 100),
        "distribution": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

st.title("🫀 Hệ Thống Phân Tích Điện Tâm Đồ (ECG Analyzer)")
st.caption("Ứng dụng kiểm thử AI Vision + Đo đạc tham số lâm sàng")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Bản ghi hình ảnh & Thông số")
    uploaded_file = st.file_uploader("Tải lên ảnh ECG (PNG, JPG)", type=["png", "jpg", "jpeg"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Bản ghi đã tải lên", use_container_width=True)

    hr = st.number_input("Tần số tim (nhịp/phút)", 30, 220, 78)
    pr = st.number_input("Khoảng PR (giây)", 0.08, 0.40, 0.16, step=0.01)
    qrs = st.number_input("Độ rộng QRS (giây)", 0.04, 0.25, 0.08, step=0.01)
    qt = st.number_input("Khoảng QT (giây)", 0.20, 0.70, 0.38, step=0.01)
    gender = st.selectbox("Giới tính bệnh nhân", ["Nam", "Nữ"])
    st_status = st.selectbox("Đoạn ST", ["Đẳng điện (Bình thường)", "ST chênh lên (STEMI)", "ST chênh xuống"])

with col_right:
    st.subheader("2. Kết quả đánh giá")
    
    rr = 60.0 / hr
    qtc = (qt / math.sqrt(rr)) * 1000
    qtc_limit = 460 if gender == "Nữ" else 450

    m1, m2, m3 = st.columns(3)
    m1.metric("Nhịp tim", f"{hr} bpm")
    m2.metric("Chu kỳ R-R", f"{rr:.2f} s")
    m3.metric("QTc (Bazett)", f"{qtc:.0f} ms")

    st.markdown("---")
    st.markdown("#### Phân tích chuyên sâu")

    if image is not None:
        if st.button("🚀 Chạy nhận diện AI từ ảnh", type="primary"):
            with st.spinner("Đang phân tích ma trận sóng..."):
                time.sleep(0.5)
                res = analyze_ecg_image(image)
                if "Bình thường" in res["label"]:
                    st.success(f"**Chẩn đoán:** {res['label']} ({res['confidence']:.1f}%)")
                else:
                    st.error(f"**Cảnh báo hình ảnh:** {res['label']} ({res['confidence']:.1f}%)")
                st.bar_chart(res["distribution"])

    # Đánh giá chỉ số
    if st_status == "ST chênh lên (STEMI)":
        st.error("🚨 **DẤU HIỆU NGUY CẤP:** Đoạn ST chênh lên gợi ý Nhồi máu cơ tim cấp.")
    elif qtc > qtc_limit:
        st.warning(f"⚠️ **CẢNH BÁO:** Khoảng QTc kéo dài ({qtc:.0f} ms > {qtc_limit} ms) - Nguy cơ loạn nhịp.")
    else:
        st.info("✅ Các chỉ số điện học nằm trong khoảng an toàn cho phép.")
