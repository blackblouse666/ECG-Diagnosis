import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG: Chẩn Đoán Block Dẫn Truyền Chuẩn Quốc Tế",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán Block Dẫn Truyền (AHA/ACC/HRS)")
st.caption("Khử lưới, đo đạc sóng P-PR-QRS động, phân tầng: Block AV (Độ 1, 2, 3), Block nhánh (RBBB/LBBB), Block phân nhánh (LAFB/LPFB)")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ & BÓC TÁCH TÍN HIỆU (KHỬ LƯỚI CARO)
# =========================================================================
def preprocess_and_remove_grid(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )

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
# 2. ĐO ĐẠC VI THỂ SÓNG P, ĐOẠN PR, PHỨC BỘ QRS & TỶ LỆ SÓNG
# =========================================================================
def analyze_lead_signals(sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "st_shift": 0.0,
        "qrs_w": 0.08, "pr_interval": 0.16, "has_rsr": False,
        "broad_s": False, "notched_r": False, "p_detected": False
    }
    if len(sig) < 30:
        return default_props

    peaks, _ = find_peaks(
        sig, distance=int(px_per_sec * 0.28),
        prominence=np.max(sig) * 0.20 if np.max(sig) > 0 else None
    )
    if len(peaks) == 0:
        return default_props

    r_amps, s_amps, q_amps = [], [], []
    qrs_widths, pr_intervals = [], []
    has_rsr_pattern = False
    has_broad_s = False
    has_notched_r = False
    p_found_count = 0

    for r in peaks:
        r_amp = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_amp)

        # 1. Tìm sóng S
        s_win = sig[r:min(len(sig), r + int(px_per_sec * 0.12))]
        if len(s_win) > 0:
            s_val = (abs(np.min(s_win)) / px_per_mv) * 10.0
            s_amps.append(s_val)
            # S rộng khi thời gian sườn âm > 40ms
            if len(np.where(s_win < -0.15 * sig[r])[0]) / px_per_sec >= 0.04:
                has_broad_s = True
        else:
            s_amps.append(0.0)

        # 2. Tìm sóng Q
        q_win = sig[max(0, r - int(px_per_sec * 0.06)):r]
        if len(q_win) > 0:
            q_amps.append((abs(np.min(q_win)) / px_per_mv) * 10.0)
        else:
            q_amps.append(0.0)

        # 3. Đo độ rộng QRS
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.10)) and sig[left_idx] > 0.12 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.14)) and sig[right_idx] > 0.12 * sig[r]:
            right_idx += 1
        measured_qrs = (right_idx - left_idx) / px_per_sec
        qrs_widths.append(measured_qrs)

        # 4. Tìm sóng P và tính khoảng PR thực tế (quét từ 120ms đến 320ms trước đỉnh R)
        p_zone_start = max(0, r - int(px_per_sec * 0.32))
        p_zone_end = max(0, r - int(px_per_sec * 0.10))
        p_zone = sig[p_zone_start:p_zone_end]
        if len(p_zone) > 5:
            p_peaks, _ = find_peaks(p_zone, prominence=0.5)
            if len(p_peaks) > 0:
                p_peak_idx = p_zone_start + p_peaks[-1]
                pr_dur = (r - p_peak_idx) / px_per_sec
                if 0.09 <= pr_dur <= 0.40:
                    pr_intervals.append(pr_dur)
                    p_found_count += 1

        # 5. Nhận diện hình thái tai thỏ rsR' hoặc khía chữ M
        sub_complex = sig[max(0, r - int(px_per_sec * 0.04)):min(len(sig), r + int(px_per_sec * 0.10))]
        local_pks, _ = find_peaks(sub_complex, distance=int(px_per_sec * 0.018), prominence=1.2)
        if len(local_pks) >= 2:
            has_rsr_pattern = True
            has_notched_r = True
        elif len(sub_complex) > 6:
            grad2 = np.diff(np.sign(np.diff(sub_complex)))
            if np.sum(grad2 < 0) >= 2:
                has_rsr_pattern = True
                has_notched_r = True

    avg_qrs = float(np.mean(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.22))
    avg_pr = float(np.mean(pr_intervals)) if pr_intervals else 0.16

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.mean(q_amps)) if q_amps else 0.0,
        "qrs_w": avg_qrs,
        "pr_interval": avg_pr,
        "has_rsr": has_rsr_pattern,
        "broad_s": has_broad_s,
        "notched_r": has_notched_r,
        "p_detected": p_found_count > 0
    }

def process_ecg_dataset(pil_img: Image.Image):
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    clean_bin = preprocess_and_remove_grid(gray)
    h_tot, w_tot = clean_bin.shape
    h_ecg = int(h_tot * 0.85)
    binary_cropped = clean_bin[:h_ecg, :]

    cell_h = h_ecg // 3
    cell_w = w_tot // 4
    px_per_sec = cell_w / 2.5
    px_per_mv = cell_h / 4.0

    leads = {}
    rr_intervals = []

    for r in range(3):
        for c in range(4):
            l_name = LEAD_GRID[r][c]
            roi = binary_cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            sig = extract_signal_from_roi(roi)
            leads[l_name] = analyze_lead_signals(sig, px_per_sec, px_per_mv)

            pks, _ = find_peaks(
                sig, distance=int(px_per_sec * 0.28),
                prominence=np.max(sig) * 0.20 if np.max(sig) > 0 else None
            )
            if len(pks) >= 2:
                rr_intervals.extend(np.diff(pks) / px_per_sec)

    mean_rr = float(np.mean(rr_intervals)) if rr_intervals else 0.65
    rr_std = float(np.std(rr_intervals)) if len(rr_intervals) > 2 else 0.02
    hr = int(60.0 / mean_rr) if mean_rr > 0 else 75

    # Lấy PR ưu tiên từ DII hoặc V1
    pr_measured = leads.get("II", {}).get("pr_interval", 0.16)
    if pr_measured == 0.16 and leads.get("V1", {}).get("p_detected", False):
        pr_measured = leads.get("V1", {}).get("pr_interval", 0.16)

    # Đo QRS đại diện kết hợp giữa V1 (nhánh phải) và V5/V6 (nhánh trái)
    v1_qrs = leads.get("V1", {}).get("qrs_w", 0.08)
    v6_qrs = leads.get("V6", {}).get("qrs_w", 0.08)
    qrs_final = max(v1_qrs, v6_qrs)

    # Hiệu chỉnh khi có tai thỏ rsR' đặc trưng
    if leads.get("V1", {}).get("has_rsr", False) and qrs_final < 0.095:
        qrs_final = 0.105

    return {
        "leads": leads,
        "hr": hr,
        "mean_rr": mean_rr,
        "rr_std": rr_std,
        "qrs": qrs_final,
        "pr": pr_measured
    }

# =========================================================================
# 3. QUY TRÌNH CHẨN ĐOÁN BLOCK DẪN TRUYỀN QUỐC TẾ (AHA/ACC/HRS)
# =========================================================================
def diagnose_conduction_blocks(data):
    leads = data.get("leads", {})
    hr = data.get("hr", 75)
    qrs = data.get("qrs", 0.08)
    pr = data.get("pr", 0.16)
    rr_std = data.get("rr_std", 0.02)

    findings = []
    alerts = []

    # ---------------- 1. TÍNH TOÁN TRỤC ĐIỆN TIM ----------------
    d1 = leads.get("I", {})
    avf = leads.get("aVF", {})
    d2 = leads.get("II", {})
    d3 = leads.get("III", {})

    net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
    net_avf = avf.get("r_amp", 0.0) - avf.get("s_amp", 0.0)
    net_d2 = d2.get("r_amp", 0.0) - d2.get("s_amp", 0.0)

    if net_d1 > 0 and net_avf >= 0:
        axis_type = "Bình thường (Normal Axis: 0° đến +90°)"
    elif net_d1 > 0 and net_avf < 0:
        if net_d2 < 0:
            axis_type = "LAD (Lệch trái bệnh lý: -30° đến -90°)"
        else:
            axis_type = "Trục trung gian sinh lý (0° đến -30°)"
    elif net_d1 <= 0 and net_avf > 0:
        axis_type = "RAD (Lệch phải bệnh lý: +90° đến +180°)"
    else:
        axis_type = "Trục vô định / Tây Bắc (-90° đến 180°)"

    # ---------------- 2. BLOCK NHĨ - THẤT (AV BLOCK) PHÂN ĐỘ ----------------
    if hr < 45 and qrs >= 0.12 and rr_std < 0.04:
        findings.append(("Block Nhĩ - Thất", "Block nhĩ - thất độ III (Block hoàn toàn): Phân ly nhĩ thất, nhịp thoát thất chậm độc lập"))
        alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT ĐỘ 3 - CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP TẠM THỜI/VĨNH VIỄN")
    elif rr_std > 0.16:
        if pr > 0.20:
            findings.append(("Block Nhĩ - Thất", "Block nhĩ - thất độ II Mobitz I (Wenckebach): Khoảng PR dài dần rồi mất QRS, chu kỳ không đều"))
        else:
            findings.append(("Block Nhĩ - Thất", "Block nhĩ - thất độ II Mobitz II: Sóng P bị block đột ngột, PR cố định (Nguy cơ sốc vô tâm thu)"))
            alerts.append("Block AV độ II Mobitz II: Nguy cơ cao tiến triển thành Block độ III")
    elif pr > 0.20:
        findings.append(("Block Nhĩ - Thất", f"Block nhĩ - thất độ I: Khoảng PR kéo dài cố định ({pr:.2f}s > 0.20s), dẫn truyền nhĩ thất chậm đều"))
    elif pr < 0.12 and pr > 0.06:
        findings.append(("Hội chứng tiền kích thích", f"Khoảng PR ngắn ({pr:.2f}s < 0.12s): Nghi ngờ Hội chứng Wolff-Parkinson-White (WPW)"))

    # ---------------- 3. BLOCK NHÁNH (RBBB / IRBBB / LBBB) ----------------
    v1 = leads.get("V1", {})
    v5 = leads.get("V5", {})
    v6 = leads.get("V6", {})

    # Tiêu chuẩn RBBB: Có rsR' hoặc rSR' ở V1/V2 HOẶC sóng R ưu thế có khía + S rộng ở DI/V6
    has_rbbb_pattern = (
        v1.get("has_rsr", False) or
        (v1.get("r_amp", 0.0) > v1.get("s_amp", 0.0) and v1.get("r_amp", 0.0) > 3.0) or
        (v6.get("broad_s", False) and v1.get("r_amp", 0.0) > 2.0)
    )

    rbbb_diag = None
    if has_rbbb_pattern:
        if qrs >= 0.12:
            rbbb_diag = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, dạng rsR' (tai thỏ) ở V1, S sâu rộng ở DI/V6"
        elif 0.09 <= qrs < 0.12:
            rbbb_diag = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.09 - 0.11s kèm dạng rsR' hoặc khía chữ M tại V1"

    if rbbb_diag:
        findings.append(("Block Nhánh", rbbb_diag))

    # Tiêu chuẩn LBBB: R rộng có khía tại DI, aVL, V5, V6, mất sóng q vách, rS/QS sâu ở V1
    lbbb_diag = None
    has_lbbb_pattern = (
        (v5.get("notched_r", False) or v6.get("notched_r", False) or d1.get("notched_r", False)) and
        (v1.get("s_amp", 0.0) > 7.0 and v1.get("r_amp", 0.0) < 3.0)
    )

    if has_lbbb_pattern:
        if qrs >= 0.12:
            lbbb_diag = "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/aVL/V5/V6, mất q vách"
            alerts.append("🚨 LBBB hoàn toàn: Cần đối chiếu tiêu chuẩn Sgarbossa loại trừ Nhồi máu cơ tim cấp")
        elif 0.10 <= qrs < 0.12:
            lbbb_diag = "Block nhánh trái không hoàn toàn (Incomplete LBBB): QRS 0.10 - 0.11s với hình thái chậm dẫn truyền nhánh trái"

    if lbbb_diag:
        findings.append(("Block Nhánh", lbbb_diag))

    # ---------------- 4. BLOCK PHÂN NHÁNH (LAFB / LPFB) ----------------
    lafb = False
    lpfb = False

    # LAFB: Trục lệch trái (-45° đến -90°), dạng qR ở DI/aVL, rS ở DII/DIII/aVF, QRS < 0.12s
    if "LAD" in axis_type and qrs < 0.12:
        if d1.get("r_amp", 0.0) > d1.get("s_amp", 0.0) and d3.get("s_amp", 0.0) > d3.get("r_amp", 0.0):
            lafb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh (-45° đến -90°), qR ở DI/aVL, rS ở DII/DIII/aVF"))

    # LPFB: Trục lệch phải (+90° đến +180°), dạng rS ở DI/aVL, qR ở DII/DIII/aVF, QRS < 0.12s (loại trừ dày thất phải)
    if "RAD" in axis_type and qrs < 0.12:
        if d3.get("r_amp", 0.0) > d3.get("s_amp", 0.0) and d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0):
            if v1.get("r_amp", 0.0) < 7.0:  # Loại trừ RVH
                lpfb = True
                findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh (+90° đến +180°), rS ở DI/aVL, qR ở DII/DIII/aVF"))

    # ---------------- 5. BLOCK HAI & BA PHÂN NHÁNH (BI/TRIFASCICULAR) ----------------
    if rbbb_diag and "Complete" in rbbb_diag:
        if lafb:
            if pr > 0.20:
                findings.append(("Block Đa Phân Nhánh", "Block ba phân nhánh (Trifascicular Block): RBBB + LAFB + Block AV độ 1 (Nguy cơ đột tử do vô tâm thu)"))
                alerts.append("Block ba phân nhánh: Chỉ định nhập viện theo dõi máy tạo nhịp vĩnh viễn")
            else:
                findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh (Bifascicular Block): RBBB kết hợp LAFB"))
        elif lpfb:
            if pr > 0.20:
                findings.append(("Block Đa Phân Nhánh", "Block ba phân nhánh (Trifascicular Block): RBBB + LPFB + Block AV độ 1"))
                alerts.append("Block ba phân nhánh: Nguy cơ block tim hoàn toàn đột ngột")
            else:
                findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh (Bifascicular Block): RBBB kết hợp LPFB"))

    return {
        "axis": axis_type,
        "findings": findings,
        "alerts": alerts
    }

# =========================================================================
# 4. GIAO DIỆN HIỂN THỊ STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Bản Ghi ECG 12 Chuyển Đạo")
    uploaded = st.file_uploader("Tải lên bản ghi điện tâm đồ (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG được nạp vào hệ thống phân tích", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Dẫn Truyền")
    if uploaded:
        with st.spinner("Đang khử lưới caro, bóc tách chân sóng QRS và đo đạc khoảng PR/RR..."):
            res = process_ecg_dataset(img_pil)
            diag = diagnose_conduction_blocks(res)

        if diag["alerts"]:
            for al in diag["alerts"]:
                st.error(al)

        st.markdown("#### Chỉ Số Dẫn Truyền Thực Nghiệm")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR)", f"{res['hr']} bpm")
        m2.metric("Khoảng PR", f"{res['pr']:.2f} s")
        m3.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m4.metric("Chu kỳ R-R", f"{res['mean_rr']:.2f} s")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán Dẫn Truyền")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if "hoàn toàn" in desc or "độ III" in desc or "ba phân nhánh" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "không hoàn toàn" in desc or "Incomplete" in desc or "độ II" in desc or "độ I" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Chưa phát hiện rối loạn dẫn truyền nhĩ thất hoặc nội thất nghiêm trọng.")

        st.markdown("---")
        with st.expander("🔍 Xem Bảng Đo Đạc Chi Tiết 12 Chuyển Đạo (Sóng Q, R, S, Tai thỏ)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                detail_list.append({
                    "Chuyển đạo": l_name,
                    "QRS (s)": f"{l_data['qrs_w']:.3f}",
                    "PR (s)": f"{l_data['pr_interval']:.2f}",
                    "R (mm)": f"{l_data['r_amp']:.1f}",
                    "S (mm)": f"{l_data['s_amp']:.1f}",
                    "rsR' (Tai thỏ)": "Phát hiện" if l_data["has_rsr"] else "-",
                    "S rộng (>40ms)": "Có" if l_data["broad_s"] else "-"
                })
            st.dataframe(detail_list, use_container_width=True, height=280)
    else:
        st.write("Đang chờ tải ảnh...")
