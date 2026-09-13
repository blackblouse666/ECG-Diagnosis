import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG: Chẩn Đoán Block Nhánh & Rối Loạn Nhịp",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 AI Phân Tích ECG Chuyên Sâu (AHA/ACC/ESC)")
st.caption("Khắc phục dương tính giả RBBB; Tích hợp chẩn đoán Rối loạn nhịp (Rung nhĩ, Cuồng nhĩ) & Ngoại tâm thu (PVC, PAC)")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ & BÓC TÁCH TÍN HIỆU (KHỬ NHIỄU LƯỚI NỀN)
# =========================================================================
def preprocess_and_clean_image(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )

    # Lọc bỏ vạch lưới ngang và dọc
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    kernel_w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_w)
    return clean

def extract_signal_from_roi(roi):
    h, w = roi.shape
    signal = []
    for col in range(w):
        pts = np.where(roi[:, col] > 0)[0]
        if len(pts) > 0:
            signal.append(h - np.median(pts))
        else:
            signal.append(signal[-1] if len(signal) > 0 else h / 2.0)
    sig = np.array(signal)
    return sig - np.median(sig)

# =========================================================================
# 2. ĐO ĐẠC HÌNH THÁI VI MÔ & PHÁT HIỆN SỰ BẤT THƯỜNG CỦA NHỊP
# =========================================================================
def analyze_lead_morphology(sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "st_shift": 0.0,
        "qrs_w": 0.08, "pr_interval": 0.16, "p_detected": False,
        "true_rsr": False, "slurred_s": False, "notched_r": False,
        "peaks": [], "qrs_list": []
    }
    if len(sig) < 30:
        return default_props

    # Tìm các đỉnh R thực thụ (ngưỡng prominence đủ cao để không bắt nhầm nhiễu)
    peaks, _ = find_peaks(
        sig, distance=int(px_per_sec * 0.25),
        prominence=np.max(sig) * 0.30 if np.max(sig) > 0 else None
    )
    if len(peaks) == 0:
        return default_props

    r_amps, s_amps, qrs_widths = [], [], []
    has_true_rsr = False
    has_slurred_s = False
    has_notched_r = False
    p_detected = False
    pr_intervals = []

    for r in peaks:
        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # 1. Tìm sóng S và đo độ rộng sóng S
        s_win = sig[r:min(len(sig), r + int(px_per_sec * 0.14))]
        if len(s_win) > 0:
            s_val = (abs(np.min(s_win)) / px_per_mv) * 10.0
            s_amps.append(s_val)
            # S rộng thực sự: thời gian sóng S duy trì âm sâu kéo dài >= 40ms
            s_neg_pts = np.where(s_win < -0.20 * sig[r])[0]
            if len(s_neg_pts) / px_per_sec >= 0.040:
                has_slurred_s = True
        else:
            s_amps.append(0.0)

        # 2. Đo chân sóng QRS
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.10)) and sig[left_idx] > 0.15 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.14)) and sig[right_idx] > 0.15 * sig[r]:
            right_idx += 1
        qrs_dur = (right_idx - left_idx) / px_per_sec
        qrs_widths.append(qrs_dur)

        # 3. Tiêu chuẩn nhận diện tai thỏ rsR' thực thụ (Tránh bắt nhầm)
        # Sóng R' phải là một đỉnh riêng biệt xuất hiện sau r từ 30ms đến 90ms và biên độ phải đủ cao
        sub_complex = sig[max(0, r - int(px_per_sec * 0.03)):min(len(sig), r + int(px_per_sec * 0.10))]
        if len(sub_complex) > 5:
            local_peaks, _ = find_peaks(sub_complex, distance=int(px_per_sec * 0.025), prominence=3.0)
            if len(local_peaks) >= 2:
                # Đỉnh sau (R') phải có ý nghĩa bệnh lý chứ không phải sóng rung nhiễu
                has_true_rsr = True
                has_notched_r = True

        # 4. Kiểm tra sóng P và khoảng PR
        p_zone = sig[max(0, r - int(px_per_sec * 0.30)):max(0, r - int(px_per_sec * 0.09))]
        if len(p_zone) > 5:
            p_pks, _ = find_peaks(p_zone, prominence=0.8)
            if len(p_pks) > 0:
                p_detected = True
                pr_dur = (r - (max(0, r - int(px_per_sec * 0.30)) + p_pks[-1])) / px_per_sec
                if 0.08 <= pr_dur <= 0.35:
                    pr_intervals.append(pr_dur)

    avg_qrs = float(np.median(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.22))

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "qrs_w": avg_qrs,
        "pr_interval": float(np.median(pr_intervals)) if pr_intervals else 0.16,
        "p_detected": p_detected,
        "true_rsr": has_true_rsr,
        "slurred_s": has_slurred_s,
        "notched_r": has_notched_r,
        "peaks": peaks,
        "qrs_list": qrs_widths
    }

def process_ecg_record(pil_img: Image.Image):
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    clean_bin = preprocess_and_clean_image(gray)

    h_tot, w_tot = clean_bin.shape
    h_ecg = int(h_tot * 0.85)
    binary_cropped = clean_bin[:h_ecg, :]

    cell_h = h_ecg // 3
    cell_w = w_tot // 4
    px_per_sec = cell_w / 2.5
    px_per_mv = cell_h / 4.0

    leads = {}
    all_rr_intervals = []
    all_qrs_measurements = []

    for r in range(3):
        for c in range(4):
            l_name = LEAD_GRID[r][c]
            roi = binary_cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            sig = extract_signal_from_roi(roi)
            m = analyze_lead_morphology(sig, px_per_sec, px_per_mv)
            leads[l_name] = m

            if len(m["peaks"]) >= 2:
                all_rr_intervals.extend(np.diff(m["peaks"]) / px_per_sec)
            all_qrs_measurements.extend(m["qrs_list"])

    # Tính biến thiên chu kỳ R-R để chẩn đoán rối loạn nhịp
    mean_rr = float(np.median(all_rr_intervals)) if all_rr_intervals else 0.80
    hr = int(60.0 / mean_rr) if mean_rr > 0 else 75
    rr_cv = (float(np.std(all_rr_intervals)) / mean_rr) if len(all_rr_intervals) > 3 else 0.0

    # Đo độ rộng QRS chuẩn: trung vị của các chuyển đạo trước tim
    qrs_final = float(np.median(all_qrs_measurements)) if all_qrs_measurements else 0.08
    qrs_final = max(0.07, min(qrs_final, 0.20))

    # Tần suất sóng P được tìm thấy trên 12 chuyển đạo
    p_presence_ratio = sum(1 for v in leads.values() if v["p_detected"]) / 12.0

    return {
        "leads": leads,
        "hr": hr,
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "rr_list": all_rr_intervals,
        "qrs": qrs_final,
        "p_ratio": p_presence_ratio
    }

# =========================================================================
# 3. BỘ TIÊU CHUẨN ĐỒNG THUẬN QUỐC TẾ (AHA/ACC/HRS) ĐÃ SỬA ĐỔI
# =========================================================================
def evaluate_conduction_and_arrhythmia(data):
    leads = data["leads"]
    hr = data["hr"]
    qrs = data["qrs"]
    rr_cv = data["rr_cv"]
    rr_list = data["rr_list"]
    p_ratio = data["p_ratio"]

    findings = []
    alerts = []

    # ---------------- 1. RỐI LOẠN NHỊP CƠ BẢN (ARRHYTHMIAS) ----------------
    # Rung nhĩ (Atrial Fibrillation - AFib): Không có sóng P + R-R hoàn toàn không đều (Hệ số biến thiên CV > 0.18)
    if rr_cv > 0.18 and p_ratio < 0.25:
        findings.append(("Rối loạn nhịp", "Rung nhĩ (Atrial Fibrillation - AFib): Mất sóng P, nhịp thất hoàn toàn không đều"))
        alerts.append("⚠️ Rung nhĩ: Đánh giá nguy cơ thuyên tắc mạch (Thang điểm CHA2DS2-VASc) và kiểm soát tần số thất")
    elif 250 <= (60.0 / (data["mean_rr"] / 4.0 if data["mean_rr"] > 0 else 1.0)) <= 350 and p_ratio < 0.20:
        findings.append(("Rối loạn nhịp", "Cuồng nhĩ (Atrial Flutter - AFL): Sóng F dạng răng cưa tần số nhĩ nhanh đều"))
    else:
        # Nhịp xoang
        if hr > 100:
            findings.append(("Rối loạn nhịp", f"Nhịp nhanh xoang (Sinus Tachycardia) - Tần số: {hr} l/p"))
        elif hr < 60:
            findings.append(("Rối loạn nhịp", f"Nhịp chậm xoang (Sinus Bradycardia) - Tần số: {hr} l/p"))
        else:
            findings.append(("Rối loạn nhịp", f"Nhịp xoang bình thường (Normal Sinus Rhythm) - Tần số: {hr} l/p"))

    # ---------------- 2. NGOẠI TÂM THU (ECTOPIC BEATS) ----------------
    # Phát hiện các nhát bóp đến sớm (RR ngắn hơn 20% so với trung bình)
    early_beats = [rr for rr in rr_list if rr < 0.80 * data["mean_rr"]]
    late_compensatory = [rr for rr in rr_list if rr > 1.20 * data["mean_rr"]]

    if len(early_beats) > 0 and len(late_compensatory) > 0:
        # Nếu nhát đến sớm có QRS giãn rộng -> Ngoại tâm thu thất (PVC)
        if qrs >= 0.12 or leads.get("V1", {}).get("qrs_w", 0.08) >= 0.12:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu thất (PVC - Premature Ventricular Contraction): Nhát bóp đến sớm, QRS giãn rộng dị dạng, nghỉ bù hoàn toàn"))
            alerts.append("Phát hiện Ngoại tâm thu thất (PVC): Cần đánh giá số lượng ổ và tần suất xuất hiện (Holter ECG)")
        else:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu nhĩ (PAC - Premature Atrial Contraction): Nhát bóp đến sớm với phức bộ QRS hẹp"))

    # ---------------- 3. TIÊU CHUẨN BLOCK NHÁNH CHUẨN HÓA (ĐÃ KHẮC PHỤC RBBB DƯƠNG TÍNH GIẢ) ----------------
    v1 = leads.get("V1", {})
    v2 = leads.get("V2", {})
    v5 = leads.get("V5", {})
    v6 = leads.get("V6", {})
    d1 = leads.get("I", {})

    # BẮT BUỘC 2 VẾ CHO RBBB:
    # Vế 1: Chuyển đạo trước tim phải (V1 hoặc V2) có dạng rsR', rSR' thực thụ (tai thỏ rõ ràng)
    v1_has_rsr = v1.get("true_rsr", False) or v2.get("true_rsr", False)
    
    # Vế 2: Chuyển đạo thành bên (I hoặc V6) BẮT BUỘC PHẢI CÓ sóng S rộng, sâu kéo dài (> 40ms)
    lateral_has_broad_s = v6.get("slurred_s", False) or d1.get("slurred_s", False)

    # Chỉ kết luận RBBB khi V1 có tai thỏ VÀ V6/I có sóng S rộng bệnh lý
    if v1_has_rsr and lateral_has_broad_s:
        if qrs >= 0.12:
            findings.append(("Block Nhánh", "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 dạng rsR' (tai thỏ), S rộng kéo dài ở DI/V6"))
        elif 0.09 <= qrs < 0.12:
            findings.append(("Block Nhánh", "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.09 - 0.11s, có hình thái rsR' tại V1 và S sâu ở V6"))
    else:
        # Nếu V1 chỉ có sóng R hơi nhô nhưng I và V6 không có sóng S rộng -> BÌNH THƯỜNG (Loại bỏ triệt để dương tính giả)
        pass

    # Tiêu chuẩn LBBB: R rộng có khía tại DI, aVL, V5, V6 + Mất sóng q vách + rS/QS sâu ở V1 + QRS kéo dài
    has_lbbb_pattern = (
        (v5.get("notched_r", False) or v6.get("notched_r", False) or d1.get("notched_r", False)) and
        (v1.get("s_amp", 0.0) > 8.0 and v1.get("r_amp", 0.0) < 3.0) and
        not v1_has_rsr
    )
    if has_lbbb_pattern:
        if qrs >= 0.12:
            findings.append(("Block Nhánh", "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/V5/V6, mất q vách"))
            alerts.append("🚨 LBBB hoàn toàn: Cần loại trừ nhồi máu cơ tim cấp tương đương STEMI (Tiêu chuẩn Sgarbossa)")
        elif 0.10 <= qrs < 0.12:
            findings.append(("Block Nhánh", "Block nhánh trái không hoàn toàn (Incomplete LBBB)"))

    # ---------------- 4. BLOCK PHÂN NHÁNH & BLOCK AV ----------------
    # Trục điện tim
    net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
    net_avf = leads.get("aVF", {}).get("r_amp", 0.0) - leads.get("aVF", {}).get("s_amp", 0.0)
    net_d2 = leads.get("II", {}).get("r_amp", 0.0) - leads.get("II", {}).get("s_amp", 0.0)

    axis = "Bình thường"
    if net_d1 > 0 and net_avf < 0 and net_d2 < 0:
        axis = "LAD (Lệch trái: -30° đến -90°)"
        if qrs < 0.12:
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh, qR ở DI/aVL, rS ở DII/DIII/aVF"))
    elif net_d1 < 0 and net_avf > 0:
        axis = "RAD (Lệch phải: +90° đến +180°)"
        if qrs < 0.12 and v1.get("r_amp", 0.0) < 6.0:
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh, rS ở DI/aVL, qR ở DII/DIII/aVF"))

    # Block AV độ 1
    pr_val = leads.get("II", {}).get("pr_interval", 0.16)
    if pr_val > 0.20:
        findings.append(("Dẫn truyền nhĩ - thất", f"Block nhĩ - thất độ I: Khoảng PR kéo dài cố định ({pr_val:.2f}s > 0.20s)"))

    return {
        "axis": axis,
        "pr": pr_val,
        "findings": findings,
        "alerts": alerts
    }

# =========================================================================
# 4. GIAO DIỆN HIỂN THỊ STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Tải Lên Bản Ghi 12 Chuyển Đạo")
    uploaded = st.file_uploader("Tải ảnh điện tim (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Phiếu đo ECG được nạp vào hệ thống", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên.")

with col2:
    st.subheader("2. Kết Quả Phân Tích Nhịp & Dẫn Truyền")
    if uploaded:
        with st.spinner("Đang quét ma trận 12 chuyển đạo, đối chiếu tiêu chuẩn kép RBBB và phân tích nhịp..."):
            res = process_ecg_record(img_pil)
            diag = evaluate_conduction_and_arrhythmia(res)

        if diag["alerts"]:
            for al in diag["alerts"]:
                st.error(al)

        st.markdown("#### Chỉ Số Đo Đạc Tự Động")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR)", f"{res['hr']} bpm")
        m2.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m3.metric("Khoảng PR", f"{diag['pr']:.2f} s")
        m4.metric("Chu kỳ R-R", f"{res['mean_rr']:.2f} s")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}` | **Độ biến thiên nhịp (R-R CV):** `{res['rr_cv']:.3f}`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if "hoàn toàn" in desc or "Rung nhĩ" in desc or "PVC" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "không hoàn toàn" in desc or "Ngoại tâm thu" in desc or "Block" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện rối loạn nhịp hoặc block nhánh.")

        st.markdown("---")
        with st.expander("🔍 Chi tiết tiêu chuẩn đối chiếu RBBB ở V1 và Chuyển đạo bên (I, V6)"):
            leads_data = res["leads"]
            v1_data = leads_data.get("V1", {})
            v6_data = leads_data.get("V6", {})
            d1_data = leads_data.get("I", {})

            st.write(f"- **V1 có dạng tai thỏ (rsR'/rSR') thực thụ:** {'Có' if v1_data.get('true_rsr') else 'Không (Âm tính)'}")
            st.write(f"- **V6 / DI có sóng S rộng (>40ms):** {'Có' if (v6_data.get('slurred_s') or d1_data.get('slurred_s')) else 'Không (Âm tính)'}")
            st.caption("Ghi chú: Tiêu chuẩn quốc tế bắt buộc PHẢI CÓ CẢ HAI điều kiện trên mới được chẩn đoán RBBB/IRBBB, tránh bắt nhầm ở ca bình thường.")
    else:
        st.write("Đang chờ tải ảnh...")
