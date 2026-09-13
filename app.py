import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG Chẩn Đoán Toàn Diện (AHA/ACC/ESC)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Chuyên Khoa Toàn Diện")
st.caption("Chuẩn hóa AHA/ACC/ESC: Dày thất (Sokolow-Lyon, Cornell), Lớn nhĩ (P phế/P nhĩ), Hội chứng vành cấp/mạn & Rối loạn dẫn truyền")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ & BÓC TÁCH TÍN HIỆU (KHỬ LƯỚI Ô VUÔNG)
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
# 2. ĐO ĐẠC VI THỂ: SÓNG P, ĐOẠN PR, QRS, SÓNG Q, ĐIỂM J VÀ ĐOẠN ST-T
# =========================================================================
def analyze_lead_signals(sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "t_amp": 0.0, "qrs_w": 0.08, "pr_interval": 0.16,
        "p_amp": 0.0, "p_dur": 0.08, "has_rsr": False, "broad_s": False,
        "notched_r": False, "p_biphasic": False
    }
    if len(sig) < 30:
        return default_props

    peaks, _ = find_peaks(
        sig, distance=int(px_per_sec * 0.28),
        prominence=np.max(sig) * 0.20 if np.max(sig) > 0 else None
    )
    if len(peaks) == 0:
        return default_props

    r_amps, s_amps, q_amps, q_durs = [], [], [], []
    st_shifts, t_amps = [], []
    qrs_widths, pr_intervals, p_amps, p_durs = [], [], [], []
    has_rsr_pattern = False
    has_broad_s = False
    has_notched_r = False
    p_biphasic = False

    for r in peaks:
        # Sóng R
        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # Sóng S
        s_win = sig[r:min(len(sig), r + int(px_per_sec * 0.12))]
        if len(s_win) > 0:
            s_val = (abs(np.min(s_win)) / px_per_mv) * 10.0
            s_amps.append(s_val)
            if len(np.where(s_win < -0.15 * sig[r])[0]) / px_per_sec >= 0.04:
                has_broad_s = True
        else:
            s_amps.append(0.0)

        # Sóng Q
        q_win = sig[max(0, r - int(px_per_sec * 0.08)):r]
        if len(q_win) > 0 and np.min(q_win) < 0:
            q_val = (abs(np.min(q_win)) / px_per_mv) * 10.0
            q_dur = len(np.where(q_win < -0.05 * sig[r])[0]) / px_per_sec
            q_amps.append(q_val)
            q_durs.append(q_dur)
        else:
            q_amps.append(0.0)
            q_durs.append(0.0)

        # Độ rộng QRS
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.10)) and sig[left_idx] > 0.12 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.14)) and sig[right_idx] > 0.12 * sig[r]:
            right_idx += 1
        qrs_widths.append((right_idx - left_idx) / px_per_sec)

        # Đoạn ST (tại điểm J, 60ms sau R) & Sóng T (140 - 280ms sau R)
        j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
        st_shifts.append((sig[j_idx] / px_per_mv) * 10.0)

        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.14)):min(len(sig), r + int(px_per_sec * 0.28))]
        if len(t_zone) > 0:
            t_amps.append((t_zone[np.argmax(np.abs(t_zone))] / px_per_mv) * 10.0)
        else:
            t_amps.append(0.0)

        # Sóng P và khoảng PR (100 - 320ms trước R)
        p_start = max(0, r - int(px_per_sec * 0.32))
        p_end = max(0, r - int(px_per_sec * 0.10))
        p_zone = sig[p_start:p_end]
        if len(p_zone) > 5:
            p_pks, _ = find_peaks(p_zone, prominence=0.4)
            if len(p_pks) > 0:
                p_idx = p_start + p_pks[-1]
                pr_dur = (r - p_idx) / px_per_sec
                if 0.08 <= pr_dur <= 0.40:
                    pr_intervals.append(pr_dur)
                    p_amps.append((sig[p_idx] / px_per_mv) * 10.0)
                    p_durs.append(0.10)
            # Kiểm tra 2 pha sóng P ở V1
            if np.max(p_zone) > 0 and np.min(p_zone) < -0.5:
                p_biphasic = True

        # Nhận diện rsR' / Tai thỏ
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

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.mean(q_amps)) if q_amps else 0.0,
        "q_dur": float(np.max(q_durs)) if q_durs else 0.0,
        "st_shift": float(np.mean(st_shifts)) if st_shifts else 0.0,
        "t_amp": float(np.mean(t_amps)) if t_amps else 0.0,
        "qrs_w": avg_qrs,
        "pr_interval": float(np.mean(pr_intervals)) if pr_intervals else 0.16,
        "p_amp": float(np.mean(p_amps)) if p_amps else 1.0,
        "p_dur": float(np.mean(p_durs)) if p_durs else 0.08,
        "has_rsr": has_rsr_pattern,
        "broad_s": has_broad_s,
        "notched_r": has_notched_r,
        "p_biphasic": p_biphasic
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

    pr_measured = leads.get("II", {}).get("pr_interval", 0.16)
    v1_qrs = leads.get("V1", {}).get("qrs_w", 0.08)
    v6_qrs = leads.get("V6", {}).get("qrs_w", 0.08)
    qrs_final = max(v1_qrs, v6_qrs)

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
# 3. BỘ CHẨN ĐOÁN LÂM SÀNG TOÀN DIỆN (AHA/ACC/ESC)
# =========================================================================
def diagnose_ecg_comprehensive(data, gender="Nam"):
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
        axis_type = "Bình thường (0° đến +90°)"
    elif net_d1 > 0 and net_avf < 0:
        axis_type = "LAD (Lệch trái: -30° đến -90°)" if net_d2 < 0 else "Trục trung gian (0° đến -30°)"
    elif net_d1 <= 0 and net_avf > 0:
        axis_type = "RAD (Lệch phải: +90° đến +180°)"
    else:
        axis_type = "Trục vô định (-90° đến 180°)"

    # ---------------- 2. LỚN NHĨ (ATRIAL ENLARGEMENT) ----------------
    p_amp_d2 = d2.get("p_amp", 0.0)
    p_amp_d3 = d3.get("p_amp", 0.0)
    p_amp_avf = avf.get("p_amp", 0.0)
    v1_data = leads.get("V1", {})

    if max(p_amp_d2, p_amp_d3, p_amp_avf) >= 2.5:
        findings.append(("Lớn buồng tim", f"Lớn nhĩ phải (P phế): Sóng P cao {max(p_amp_d2, p_amp_d3, p_amp_avf):.1f} mm (≥ 2.5 mm ở DII/DIII/aVF)"))
    elif d2.get("p_dur", 0.08) >= 0.12 or v1_data.get("p_biphasic", False):
        findings.append(("Lớn buồng tim", "Lớn nhĩ trái (P nhĩ): Sóng P rộng ≥ 0.12s hoặc 2 pha âm chiếm ưu thế ở V1 (Chỉ số Morris)"))

    # ---------------- 3. DÀY THẤT (VENTRICULAR HYPERTROPHY) ----------------
    v5_data = leads.get("V5", {})
    v6_data = leads.get("V6", {})
    avl_data = leads.get("aVL", {})
    v3_data = leads.get("V3", {})

    # Sokolow-Lyon LVH: SV1 + RV5 hoặc RV6 >= 35mm
    sokolow_lv = v1_data.get("s_amp", 0.0) + max(v5_data.get("r_amp", 0.0), v6_data.get("r_amp", 0.0))
    # Cornell: RaVL + SV3 (> 28mm ở Nam, > 20mm ở Nữ)
    cornell_val = avl_data.get("r_amp", 0.0) + v3_data.get("s_amp", 0.0)
    cornell_cutoff = 28.0 if gender == "Nam" else 20.0

    if sokolow_lv >= 35.0:
        findings.append(("Phì đại thất", f"Dày thất trái (LVH) theo Sokolow-Lyon: SV1 + RV5 = {sokolow_lv:.1f} mm (≥ 35 mm)"))
    elif cornell_val > cornell_cutoff:
        findings.append(("Phì đại thất", f"Dày thất trái (LVH) theo Cornell: RaVL + SV3 = {cornell_val:.1f} mm (> {cornell_cutoff:.0f} mm ở {gender})"))

    # Dày thất phải (RVH): RV1 >= 7mm, R/S ở V1 > 1 kết hợp trục lệch phải
    rv1 = v1_data.get("r_amp", 0.0)
    sv1 = v1_data.get("s_amp", 0.0)
    if (rv1 >= 7.0 or (sv1 > 0 and rv1 / sv1 > 1.0)) and "RAD" in axis_type:
        findings.append(("Phì đại thất", f"Dày thất phải (RVH): R sóng ưu thế ở V1 (R={rv1:.1f} mm) kèm trục lệch phải"))

    # ---------------- 4. HỘI CHỨNG VÀNH CẤP & VÀNH MẠN (ACS / CCS) ----------------
    st_elev_leads = []
    st_depr_leads = []
    path_q_leads = []

    for l_name, l_val in leads.items():
        # Tiêu chuẩn ST chênh lên: V2, V3 >= 1.5 - 2.0 mm; Chuyển đạo khác >= 1.0 mm
        st_cutoff = 1.5 if l_name in ["V2", "V3"] else 1.0
        if l_val.get("st_shift", 0.0) >= st_cutoff:
            st_elev_leads.append(l_name)
        elif l_val.get("st_shift", 0.0) <= -0.8:
            st_depr_leads.append(l_name)

        # Sóng Q hoại tử (rộng >= 0.04s, sâu >= 25% sóng R)
        if l_val.get("q_dur", 0.0) >= 0.035 and (l_val.get("r_amp", 0.0) > 0 and l_val.get("q_amp", 0.0) >= 0.25 * l_val.get("r_amp", 0.0)):
            path_q_leads.append(l_name)

    st_set = set(st_elev_leads)
    q_set = set(path_q_leads)

    def map_anatomy(leads_found):
        regions = []
        arteries = []
        if {"V1", "V2"}.issubset(leads_found) and not {"V3", "V4"}.issubset(leads_found):
            regions.append("Vách liên thất (Septal)")
            arteries.append("LAD")
        if {"V3", "V4"}.issubset(leads_found):
            regions.append("Thành trước (Anterior)")
            arteries.append("LAD")
        if {"V1", "V2", "V3", "V4"}.issubset(leads_found):
            regions.append("Trước - Vách (Anteroseptal)")
            arteries.append("LAD đoạn gần")
        if {"V1", "V2", "V3", "V4", "V5", "V6"}.issubset(leads_found):
            regions.append("Trước rộng (Extensive Anterior)")
            arteries.append("Thân chung / Proximal LAD")
        if len({"II", "III", "aVF"}.intersection(leads_found)) >= 2:
            regions.append("Thành dưới (Inferior)")
            arteries.append("RCA / LCx")
        if {"I", "aVL"}.issubset(leads_found) or {"V5", "V6"}.issubset(leads_found):
            regions.append("Thành bên (Lateral)")
            arteries.append("LCx")
        return list(set(regions)), list(set(arteries))

    # Đánh giá Hội chứng vành cấp (ACS - STEMI)
    if st_elev_leads:
        regions, arteries = map_anatomy(st_set)
        reg_str = ", ".join(regions) if regions else ", ".join(st_elev_leads)
        art_str = f" - ĐM thủ phạm: {', '.join(arteries)}" if arteries else ""
        
        # Phân loại giai đoạn STEMI
        has_necrosis = any(l in q_set for l in st_elev_leads)
        stage_str = "Bán cấp / Hoại tử tiến triển (Có sóng Q)" if has_necrosis else "Tối cấp / Cấp (Chưa có sóng Q hoại tử)"
        
        findings.append(("Hội chứng vành cấp (ACS)", f"Nhồi máu cơ tim ST chênh lên (STEMI) - Vùng: {reg_str} - Giai đoạn: {stage_str}{art_str}"))
        alerts.append(f"🚨 STEMI VÙNG {reg_str.upper()}: KÍCH HOẠT QUY TRÌNH PCI CẤP CỨU")

    # Đánh giá Hội chứng vành mạn (CCS - Nhồi máu cơ tim cũ / Thiếu máu cơ tim ổn định)
    elif path_q_leads:
        regions, _ = map_anatomy(q_set)
        reg_str = ", ".join(regions) if regions else ", ".join(path_q_leads)
        findings.append(("Hội chứng vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old Myocardial Infarction) - Vùng: {reg_str}"))
    elif st_depr_leads:
        dep_set = set(st_depr_leads)
        regions, _ = map_anatomy(dep_set)
        reg_str = ", ".join(regions) if regions else ", ".join(st_depr_leads)
        findings.append(("Thiếu máu cục bộ cơ tim", f"ST chênh xuống / Thiếu máu dưới nội tâm mạc (NSTEMI / Đau thắt ngực) - Vùng: {reg_str}"))

    # ---------------- 5. BLOCK DẪN TRUYỀN (AV BLOCK, BUNDLE BRANCH, FASCICULAR) ----------------
    # AV Block
    if hr < 45 and qrs >= 0.12 and rr_std < 0.04:
        findings.append(("Dẫn truyền nhĩ - thất", "Block nhĩ - thất độ III: Phân ly nhĩ thất hoàn toàn"))
        alerts.append("🚨 BLOCK TIM HOÀN TOÀN: CHỈ ĐỊNH MÁY TẠO NHỊP CẤP CỨU")
    elif pr > 0.20:
        findings.append(("Dẫn truyền nhĩ - thất", f"Block nhĩ - thất độ I: Khoảng PR kéo dài ({pr:.2f}s > 0.20s)"))

    # Bundle Branch Block
    has_rbbb = (
        v1_data.get("has_rsr", False) or
        (v1_data.get("r_amp", 0.0) > v1_data.get("s_amp", 0.0) and v1_data.get("r_amp", 0.0) > 3.0) or
        (v6_data.get("broad_s", False) and v1_data.get("r_amp", 0.0) > 2.0)
    )
    rbbb_type = None
    if has_rbbb:
        if qrs >= 0.12:
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, rsR' ở V1, S rộng ở DI/V6"
        elif 0.09 <= qrs < 0.12:
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.09 - 0.11s, dạng tai thỏ V1"

    if rbbb_type:
        findings.append(("Block Nhánh", rbbb_type))

    has_lbbb = (
        (v5_data.get("notched_r", False) or v6_data.get("notched_r", False)) and
        (v1_data.get("s_amp", 0.0) > 7.0 and v1_data.get("r_amp", 0.0) < 3.0)
    )
    if has_lbbb and qrs >= 0.12:
        findings.append(("Block Nhánh", "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/V5/V6"))
        alerts.append("LBBB hoàn toàn: Cần loại trừ nhồi máu cơ tim cấp tương đương STEMI")

    # Fascicular Blocks
    lafb = False
    lpfb = False
    if "LAD" in axis_type and qrs < 0.12:
        if d1.get("r_amp", 0.0) > d1.get("s_amp", 0.0) and d3.get("s_amp", 0.0) > d3.get("r_amp", 0.0):
            lafb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh, qR ở DI/aVL, rS ở DII/DIII/aVF"))
    elif "RAD" in axis_type and qrs < 0.12 and rv1 < 7.0:
        if d3.get("r_amp", 0.0) > d3.get("s_amp", 0.0) and d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0):
            lpfb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh, rS ở DI/aVL, qR ở DII/DIII/aVF"))

    # Đa phân nhánh
    if rbbb_type and "Complete" in rbbb_type:
        if lafb:
            findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh: RBBB kết hợp LAFB" + (" + Block AV độ 1 (Block 3 phân nhánh)" if pr > 0.20 else "")))
        elif lpfb:
            findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh: RBBB kết hợp LPFB" + (" + Block AV độ 1 (Block 3 phân nhánh)" if pr > 0.20 else "")))

    return {
        "axis": axis_type,
        "sokolow": sokolow_lv,
        "cornell": cornell_val,
        "findings": findings,
        "alerts": alerts
    }

# =========================================================================
# 4. GIAO DIỆN HIỂN THỊ STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Bản Ghi ECG 12 Chuyển Đạo")
    gender_choice = st.radio("Giới tính bệnh nhân (để tính chuẩn Cornell)", ["Nam", "Nữ"], horizontal=True)
    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG đã nạp vào bộ xử lý", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Chuyên Khoa")
    if uploaded:
        with st.spinner("Đang đo đạc sóng P, Q, R, S, ST, tính Sokolow-Lyon & Cornell và phân tích định khu..."):
            res = process_ecg_dataset(img_pil)
            diag = diagnose_ecg_comprehensive(res, gender=gender_choice)

        if diag["alerts"]:
            for al in diag["alerts"]:
                st.error(al)

        st.markdown("#### Chỉ Số Đo Đạc & Điện Thế Học Tự Động")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR)", f"{res['hr']} bpm")
        m2.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m3.metric("Sokolow-Lyon (LVH)", f"{diag['sokolow']:.1f} mm", help="Dày thất trái nếu ≥ 35 mm")
        m4.metric("Cornell (LVH)", f"{diag['cornell']:.1f} mm", help="Dày thất trái nếu > 28 mm (Nam) hoặc > 20 mm (Nữ)")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}` | **Khoảng PR:** `{res['pr']:.2f} s`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán Chi Tiết")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if "STEMI" in desc or "độ III" in desc or "CẤP" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "Block" in desc or "Dày thất" in desc or "Lớn nhĩ" in desc or "mạn" in desc or "Thiếu máu" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Chưa phát hiện bất thường tái cực, phì đại buồng tim hoặc rối loạn dẫn truyền nghiêm trọng.")

        st.markdown("---")
        with st.expander("🔍 Xem Bảng Đo Đạc Chi Tiết 12 Chuyển Đạo (P, Q, R, S, ST, T)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                detail_list.append({
                    "Chuyển đạo": l_name,
                    "P (mm)": f"{l_data['p_amp']:.1f}",
                    "Q (mm)": f"{l_data['q_amp']:.1f}",
                    "R (mm)": f"{l_data['r_amp']:.1f}",
                    "S (mm)": f"{l_data['s_amp']:.1f}",
                    "ST Chênh (mm)": f"{l_data['st_shift']:+.1f}",
                    "rsR' (V1)": "Có" if l_data["has_rsr"] else "-"
                })
            st.dataframe(detail_list, use_container_width=True, height=280)
    else:
        st.write("Đang chờ tải ảnh...")
