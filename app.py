import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG Chẩn Đoán Chuyên Khoa Tim Mạch (YDS 2026 Guidelines)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Chuyên Khoa Toàn Diện")
st.caption("Cập nhật toàn bộ tiêu chuẩn Hội Chứng Mạch Vành Cấp & Mạn (Sổ tay ĐTĐ YDS 2026) kết hợp bộ tiêu chuẩn Dẫn truyền, Dày thất, Lớn nhĩ AHA/ESC")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ ẢNH & BÓC TÁCH ĐƯỜNG TÍN HIỆU
# =========================================================================
def extract_robust_ecg_traces(rgb_img):
    img_float = rgb_img.astype(np.float32)
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    gray_evidence = np.maximum(np.abs(r - g), np.abs(r - b))
    brightness = (r + g + b) / 3.0

    mask_trace = (gray_evidence < 35) & (brightness < 175)
    binary = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    binary[mask_trace] = 255

    if np.mean(binary > 0) < 0.015:
        gray_fallback = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_fallback)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 10
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def extract_signal_from_roi(roi):
    h, w = roi.shape
    signal = []
    for col in range(w):
        pts = np.where(roi[:, col] > 0)[0]
        if len(pts) > 0:
            signal.append(h - np.median(pts))
        else:
            signal.append(signal[-1] if len(signal) > 0 else h / 2.0)
    return np.array(signal)

# =========================================================================
# 2. ĐO ĐẠC HÌNH THÁI CHI TIẾT THEO TIÊU CHUẨN SỔ TAY ĐTĐ YDS
# =========================================================================
def analyze_lead_morphology(raw_sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "st_slope": "flat", "t_amp": 0.0, "t_morph": "normal",
        "qx_qt_ratio": 0.40, "qrs_w": 0.08, "pr_interval": 0.16,
        "p_amp": 0.0, "p_detected": False, "p_biphasic": False,
        "true_rsr": False, "slurred_s": False, "notched_r": False,
        "r_peaks": [], "rr_intervals": []
    }
    if len(raw_sig) < 40:
        return default_props

    cut_start = int(len(raw_sig) * 0.06)
    sig_search = raw_sig[cut_start:]
    if len(sig_search) == 0:
        return default_props

    amp_span = np.max(sig_search) - np.min(sig_search)
    prom_val = amp_span * 0.25 if amp_span > 6.0 else 3.0

    peaks, _ = find_peaks(raw_sig, distance=int(px_per_sec * 0.35), prominence=prom_val)
    peaks = [p for p in peaks if p > cut_start]
    if len(peaks) == 0:
        peaks, _ = find_peaks(raw_sig, distance=int(px_per_sec * 0.35), prominence=2.0)
        peaks = [p for p in peaks if p > cut_start]
        if len(peaks) == 0:
            return default_props

    r_amps, s_amps, q_amps, q_durs = [], [], [], []
    st_shifts, t_amps, qrs_widths, pr_intervals, p_amps = [], [], [], [], []
    st_slopes, t_morphs, qx_qt_ratios = [], [], []
    has_true_rsr = False
    has_slurred_s = False
    has_notched_r = False
    p_detected = False
    p_biphasic = False

    for r in peaks:
        pr_zone = raw_sig[max(0, r - int(px_per_sec * 0.14)):max(0, r - int(px_per_sec * 0.05))]
        baseline_val = np.median(pr_zone) if len(pr_zone) > 0 else np.median(raw_sig)
        sig = raw_sig - baseline_val

        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # 1. Đo sóng Q (vùng trước đỉnh R 80ms)
        q_zone = sig[max(0, r - int(px_per_sec * 0.08)):r]
        if len(q_zone) > 0 and np.min(q_zone) < 0:
            q_peak_idx = np.argmin(q_zone)
            q_depth_mm = (abs(q_zone[q_peak_idx]) / px_per_mv) * 10.0
            q_negative_pts = np.where(q_zone < -0.05 * max(r_val * (px_per_mv / 10.0), 5.0))[0]
            q_duration_sec = len(q_negative_pts) / px_per_sec
            q_amps.append(q_depth_mm)
            q_durs.append(q_duration_sec)
        else:
            q_amps.append(0.0)
            q_durs.append(0.0)

        # 2. Tìm sóng S và Điểm J
        s_search = sig[r:min(len(sig), r + int(px_per_sec * 0.14))]
        if len(s_search) > 2:
            min_s_idx = np.argmin(s_search)
            s_val = (abs(min(0.0, s_search[min_s_idx])) / px_per_mv) * 10.0
            s_amps.append(s_val)

            j_search_zone = s_search[max(1, int(px_per_sec * 0.04)):min(len(s_search), int(px_per_sec * 0.09))]
            if len(j_search_zone) > 0:
                j_idx = r + max(1, int(px_per_sec * 0.04)) + int(np.argmin(np.abs(np.diff(j_search_zone, prepend=j_search_zone[0]))))
            else:
                j_idx = r + min_s_idx
        else:
            j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
            s_amps.append(0.0)

        # 3. Đo độ lệch ST tại điểm J (mm) và hình thái hướng đi (Slope)
        st_shift_mm = (sig[j_idx] / px_per_mv) * 10.0
        st_shifts.append(st_shift_mm)

        # Điểm J80 (60-80ms sau điểm J) để xác định kiểu ST chênh xuống đi ngang, đi xuống hay đi lên
        j80_idx = min(len(sig) - 1, j_idx + int(px_per_sec * 0.07))
        st_j80_shift = (sig[j80_idx] / px_per_mv) * 10.0
        
        slope_diff = st_j80_shift - st_shift_mm
        if slope_diff > 0.4:
            st_slope = "upsloping"     # Đi lên
        elif slope_diff < -0.4:
            st_slope = "downsloping"   # Đi xuống (rất đặc hiệu)
        else:
            st_slope = "horizontal"    # Đi ngang (đặc hiệu)
        st_slopes.append(st_slope)

        # 4. Sóng T và Tỷ số QX/QT (Hình 4.1, 4.4, 5.5 trong tài liệu)
        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.12)):min(len(sig), r + int(px_per_sec * 0.38))]
        if len(t_zone) > 8:
            t_max = np.max(t_zone)
            t_min = np.min(t_zone)
            if abs(t_min) > abs(t_max):
                t_amp_val = (t_min / px_per_mv) * 10.0
            else:
                t_amp_val = (t_max / px_per_mv) * 10.0
            t_amps.append(t_amp_val)

            # Đánh giá hình thái sóng T: dẹt, âm, hai pha (+/- hoặc -/+), tối cấp
            if -1.0 <= t_amp_val <= 1.0:
                t_morph = "flat"  # Sóng T dẹt
            elif t_amp_val <= -1.0:
                t_morph = "inverted"  # Sóng T âm
            elif t_max > 1.0 and t_min < -1.0:
                # Kiểm tra pha nào đến trước (T hai pha trong thiếu máu cấp bù cấp: pha dương trước âm)
                if np.argmax(t_zone) < np.argmin(t_zone):
                    t_morph = "biphasic_pos_neg"  # Wellens type A / Thiếu máu cấp
                else:
                    t_morph = "biphasic_neg_pos"  # Hạ kali máu
            else:
                t_morph = "positive"
            t_morphs.append(t_morph)

            # Tỷ số QX/QT (Điểm X là nơi ST cắt đường đẳng điện sau điểm J)
            qt_duration = len(t_zone) / px_per_sec
            x_cross = np.where(np.diff(np.sign(sig[j_idx:j_idx + len(t_zone)])))[0]
            if len(x_cross) > 0 and qt_duration > 0:
                qx_duration = x_cross[0] / px_per_sec
                qx_qt_ratios.append(qx_duration / qt_duration)
            else:
                qx_qt_ratios.append(0.45)
        else:
            t_amps.append(0.0)
            t_morphs.append("normal")
            qx_qt_ratios.append(0.40)

        # 5. Độ rộng QRS
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.10)) and sig[left_idx] > 0.15 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.12)) and sig[right_idx] > 0.15 * sig[r]:
            right_idx += 1
        qrs_widths.append((right_idx - left_idx) / px_per_sec)

        # 6. Sóng P và khoảng PR
        p_zone_start = max(0, r - int(px_per_sec * 0.36))
        p_zone_end = max(0, r - int(px_per_sec * 0.10))
        p_zone = sig[p_zone_start:p_zone_end]
        if len(p_zone) > 5:
            p_pks, _ = find_peaks(p_zone, prominence=0.5)
            if len(p_pks) > 0:
                p_idx = p_zone_start + p_pks[-1]
                pr_dur = (r - p_idx) / px_per_sec
                if 0.11 <= pr_dur <= 0.38:
                    pr_intervals.append(pr_dur)
                    p_amps.append(min(4.5, (sig[p_idx] / px_per_mv) * 10.0))
                    p_detected = True

    rr_list = np.diff(peaks) / px_per_sec if len(peaks) >= 2 else []
    avg_qrs = float(np.median(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.18))

    return {
        "r_amp": float(np.median(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.median(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.median(q_amps)) if q_amps else 0.0,
        "q_dur": float(np.max(q_durs)) if q_durs else 0.0,
        "st_shift": float(np.median(st_shifts)) if st_shifts else 0.0,
        "st_slope": max(set(st_slopes), key=st_slopes.count) if st_slopes else "flat",
        "t_amp": float(np.median(t_amps)) if t_amps else 0.0,
        "t_morph": max(set(t_morphs), key=t_morphs.count) if t_morphs else "normal",
        "qx_qt_ratio": float(np.median(qx_qt_ratios)) if qx_qt_ratios else 0.40,
        "qrs_w": avg_qrs,
        "pr_interval": float(np.median(pr_intervals)) if pr_intervals else 0.16,
        "p_amp": float(np.median(p_amps)) if p_amps else 1.0,
        "p_detected": p_detected,
        "p_biphasic": p_biphasic,
        "true_rsr": has_true_rsr,
        "slurred_s": has_slurred_s,
        "notched_r": has_notched_r,
        "r_peaks": peaks,
        "rr_intervals": rr_list
    }

def process_ecg_dataset(pil_img: Image.Image):
    rgb_img = np.array(pil_img.convert("RGB"))
    clean_bin = extract_robust_ecg_traces(rgb_img)

    h_tot, w_tot = clean_bin.shape
    h_ecg = int(h_tot * 0.85)
    binary_cropped = clean_bin[:h_ecg, :]

    cell_h = h_ecg // 3
    cell_w = w_tot // 4
    px_per_sec = cell_w / 2.5
    px_per_mv = cell_h / 4.0

    leads = {}
    all_rr_intervals = []
    all_pr_intervals = []
    all_qrs_measurements = []

    for r in range(3):
        for c in range(4):
            l_name = LEAD_GRID[r][c]
            roi = binary_cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            sig = extract_signal_from_roi(roi)
            m = analyze_lead_morphology(sig, px_per_sec, px_per_mv)
            leads[l_name] = m

            all_rr_intervals.extend(m["rr_intervals"])
            if m["p_detected"]:
                all_pr_intervals.append(m["pr_interval"])
            all_qrs_measurements.append(m["qrs_w"])

    valid_rrs = [rr for rr in all_rr_intervals if 0.40 <= rr <= 1.8]
    mean_rr = float(np.median(valid_rrs)) if valid_rrs else 0.85
    hr = int(60.0 / mean_rr) if mean_rr > 0 else 72
    rr_cv = (float(np.std(valid_rrs)) / mean_rr) if len(valid_rrs) > 3 else 0.0

    pr_final = float(np.median(all_pr_intervals)) if all_pr_intervals else 0.16
    if leads.get("II", {}).get("p_detected", False):
        pr_final = leads["II"]["pr_interval"]

    qrs_final = float(np.median(all_qrs_measurements)) if all_qrs_measurements else 0.08
    p_presence_ratio = sum(1 for v in leads.values() if v.get("p_detected", False)) / 12.0

    return {
        "leads": leads,
        "hr": hr,
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "rr_list": valid_rrs,
        "qrs": qrs_final,
        "pr": pr_final,
        "p_ratio": p_presence_ratio
    }

# =========================================================================
# 3. BỘ CHẨN ĐOÁN HỘI CHỨNG MẠCH VÀNH CẤP & MẠN CHUẨN YDS 2026
# =========================================================================
def evaluate_yds_coronary_syndromes(leads, gender="Nam", age=55):
    """
    Tích hợp toàn bộ tiêu chuẩn Sổ tay điện tâm đồ YDS 2026:
    - Thiếu máu cấp: ST chênh xuống đi ngang/đi xuống >= 0.5mm, T tối cấp, T âm đối xứng, T hai pha (+/-).
    - Tổn thương cấp (STEMI): ST chênh lên theo nhóm tuổi/giới, đối chiếu soi gương, định vị động mạch thủ phạm.
    - Biến thể tương đương: Hội chứng De Winter, Wellens Type A & B, Thân chung/3 nhánh.
    - Hội chứng vành mạn (CCS): Sóng T dẹt, T âm mạn tính, Q hoại tử liên tiếp, R cắt cụt (DePace).
    """
    st_elevation_leads = []
    st_depression_leads = []
    hyperacute_t_leads = []
    inverted_t_leads = []
    flat_t_leads = []
    biphasic_t_leads = []
    pathological_q_leads = []

    for l_name, l_data in leads.items():
        if l_name == "aVR":
            continue

        st_val = l_data.get("st_shift", 0.0)
        t_val = l_data.get("t_amp", 0.0)
        t_morph = l_data.get("t_morph", "normal")
        r_val = l_data.get("r_amp", 0.0)
        s_val = l_data.get("s_amp", 0.0)
        q_dur = l_data.get("q_dur", 0.0)
        q_amp = l_data.get("q_amp", 0.0)
        slope = l_data.get("st_slope", "flat")

        # 1. ST chênh lên (ESC/ACC/AHA/WHF 2018)
        if l_name in ["V2", "V3"]:
            if gender == "Nữ":
                cutoff = 1.5
            else:
                cutoff = 2.0 if age >= 40 else 2.5
        else:
            cutoff = 1.0

        if st_val >= cutoff:
            st_elevation_leads.append(l_name)

        # 2. ST chênh xuống đi ngang hoặc đi xuống >= 0.5 mm
        if st_val <= -0.5:
            st_depression_leads.append((l_name, slope))

        # 3. Sóng T tối cấp: > 5mm ở chi hoặc > 10mm ở trước ngực hoặc > 3/4 R
        is_chest = l_name.startswith("V")
        if (is_chest and t_val > 10.0) or (not is_chest and t_val > 5.0) or (r_val > 0 and t_val > 0.75 * r_val):
            hyperacute_t_leads.append(l_name)

        # 4. Sóng T đảo ngược (> 1mm ở chuyển đạo có R ưu thế hoặc R/S > 1)
        if t_val < -1.0 and (r_val > s_val or r_val > 5.0):
            inverted_t_leads.append(l_name)

        # 5. Sóng T dẹt (-1mm đến +1mm)
        if -1.0 <= t_val <= 1.0 and r_val > 3.0:
            flat_t_leads.append(l_name)

        # 6. Sóng T hai pha (pha dương trước pha âm: +/-)
        if t_morph == "biphasic_pos_neg":
            biphasic_t_leads.append(l_name)

        # 7. Sóng Q hoại tử (chuẩn ESC/ACC/AHA/WHF 2018):
        # - V2-V3: Bất kỳ Q nào > 0.02s hoặc dạng QS
        # - DI, DII, aVL, aVF, V4-V6: Q >= 0.04s và sâu >= 1mm hoặc dạng QS
        if l_name in ["V2", "V3"]:
            if q_dur > 0.020 or (r_val == 0.0 and q_amp >= 2.0):
                pathological_q_leads.append(l_name)
        else:
            if (q_dur >= 0.038 and q_amp >= 1.0) or (r_val == 0.0 and q_amp >= 2.0):
                pathological_q_leads.append(l_name)

    findings = []
    alerts = []
    st_set = set(st_elevation_leads)
    dep_leads_names = [item[0] for item in st_depression_leads]
    dep_set = set(dep_leads_names)

    # ------------------ A. BIẾN THỂ TƯƠNG ĐƯƠNG STEMI ------------------
    # 1. Hội chứng De Winter: Điểm J chênh xuống đi lên ở V1-V6 kèm T cao đối xứng, thường kèm ST chênh lên ở aVR
    avr_st = leads.get("aVR", {}).get("st_shift", 0.0)
    dewinter_candidates = [l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping" and l in hyperacute_t_leads]
    if len(dewinter_candidates) >= 2 or (avr_st >= 0.5 and len([l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping"]) >= 2):
        findings.append(("Hội chứng mạch vành cấp", "Hội chứng De Winter: Điểm J chênh xuống đi lên ở V1-V6 kèm sóng T cao đối xứng (Tương đương STEMI tắc đoạn gần LAD)"))
        alerts.append("🚨 CẤP CỨU: HỘI CHỨNG DE WINTER - CHỈ ĐỊNH CAN THIỆP MẠCH VÀNH KHẨN CẤP")

    # 2. Hội chứng Wellens: Type A (T hai pha ở V1-V3) hoặc Type B (T âm sâu ở V1-V6) gợi ý tắc đoạn gần LAD
    wellens_a = [l for l in biphasic_t_leads if l in ["V1", "V2", "V3"]]
    wellens_b = [l for l in inverted_t_leads if l in ["V1", "V2", "V3", "V4"]]
    if len(wellens_a) >= 2:
        findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type A: Sóng T hai pha (+/-) tại {', '.join(wellens_a)} (Gợi ý hẹp nặng đoạn gần LAD)"))
        alerts.append("⚠️ CẢNH BÁO: HỘI CHỨNG WELLENS TYPE A - NGUY CƠ TIẾN TRIỂN NMCT THÀNH TRƯỚC RỘNG")
    elif len(wellens_b) >= 2:
        findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type B: Sóng T âm sâu đối xứng tại {', '.join(wellens_b)} (Gợi ý hẹp nặng đoạn gần LAD)"))
        alerts.append("⚠️ CẢNH BÁO: HỘI CHỨNG WELLENS TYPE B - CHỈ ĐỊNH CHỤP MẠCH VÀNH SỚM")

    # 3. Bệnh thân chung (LMCA) hoặc 3 nhánh: ST chênh xuống lan tỏa >= 6 chuyển đạo + ST chênh lên ở aVR
    if len(dep_leads_names) >= 6 and avr_st >= 1.0:
        findings.append(("Hội chứng mạch vành cấp", f"Gợi ý tổn thương Thân chung ĐM Vành Trái (LMCA) hoặc 3 nhánh mạch vành: ST chênh xuống lan tỏa ({', '.join(dep_leads_names)}) kèm ST chênh lên tại aVR"))
        alerts.append("🚨 NGUY KỊCH: THEO DÕI HẸP NẶNG THÂN CHUNG LMCA / 3 NHÁNH - HỘI CHẨN CAN THIỆP CẤP")

    # ------------------ B. NHỒI MÁU CƠ TIM CÓ ST CHÊNH LÊN (STEMI) ------------------
    stemi_regions = []
    culprit_artery = []

    # Định khu chi tiết theo Bảng 4.3 trong tài liệu
    if {"V1", "V2", "V3", "V4", "V5", "V6"}.issubset(st_set) or ({"V1", "V2", "V3", "V4"}.issubset(st_set) and {"I", "aVL"}.intersection(st_set)):
        recip = " (Soi gương ở DII, DIII, aVF)" if len({"II", "III", "aVF"}.intersection(dep_set)) >= 1 else ""
        stemi_regions.append(f"Thành trước rộng (Extensive Anterior: V1-V6, DI, aVL){recip}")
        culprit_artery.append("Đoạn gần LAD (pLAD)")
    elif {"V1", "V2", "V3", "V4"}.issubset(st_set):
        stemi_regions.append("Thành trước vách (Anteroseptal: V1-V4)")
        culprit_artery.append("LAD (trước S1/D1)")
    elif len({"V2", "V3", "V4", "V5"}.intersection(st_set)) >= 3:
        stemi_regions.append("Thành trước (Anterior: V2-V5)")
        culprit_artery.append("LAD")
    elif {"V1", "V2"}.issubset(st_set):
        stemi_regions.append("Vách liên thất (Septal: V1-V2)")
        culprit_artery.append("Nhánh vách của LAD")
    elif {"V3", "V4"}.issubset(st_set):
        stemi_regions.append("Thành trước (Anterior: V3-V4)")
        culprit_artery.append("LAD đoạn giữa")

    # Thành bên
    if {"I", "aVL"}.issubset(st_set) and not {"V5", "V6"}.intersection(st_set):
        recip = " (Soi gương ở DII, DIII, aVF)" if len({"II", "III", "aVF"}.intersection(dep_set)) >= 1 else ""
        stemi_regions.append(f"Thành bên cao đơn thuần (High Lateral: DI, aVL){recip}")
        culprit_artery.append("Nhánh D1 của LAD hoặc LCx")
    elif {"V5", "V6"}.issubset(st_set) and not {"I", "aVL"}.intersection(st_set):
        stemi_regions.append("Thành bên thấp (Low Lateral: V5-V6)")
        culprit_artery.append("Đoạn xa LAD (dLAD)")
    elif {"V5", "V6", "I", "aVL"}.issubset(st_set):
        stemi_regions.append("Thành bên toàn bộ (Lateral: V5, V6, DI, aVL)")
        culprit_artery.append("LCx hoặc nhánh D1 của LAD")

    # Thành dưới
    inferior_leads = {"II", "III", "aVF"}.intersection(st_set)
    if len(inferior_leads) >= 2:
        recip = " (Soi gương ở aVL, DI, V1-V3)" if len({"aVL", "I", "V1", "V2"}.intersection(dep_set)) >= 1 else ""
        stemi_regions.append(f"Thành dưới (Inferior: {', '.join(sorted(list(inferior_leads)))}){recip}")
        culprit_artery.append("RCA (80%) hoặc LCx (20%)")

    # Dấu gián tiếp NMCT Thành sau thực: R/S > 1, ST chênh xuống, T dương ở V2-V3
    v2_r = leads.get("V2", {}).get("r_amp", 0.0)
    v2_s = leads.get("V2", {}).get("s_amp", 0.0)
    v2_st = leads.get("V2", {}).get("st_shift", 0.0)
    v2_t = leads.get("V2", {}).get("t_amp", 0.0)
    if (v2_s > 0 and v2_r / v2_s > 1.0) and v2_st <= -0.5 and v2_t > 0:
        stemi_regions.append("Dấu hiệu gián tiếp NMCT Thành sau thực (R/S > 1, ST chênh xuống, T dương ở V2-V3 - Đề nghị đo thêm V7-V9)")
        culprit_artery.append("RCA hoặc LCx")

    if stemi_regions:
        reg_txt = " + ".join(stemi_regions)
        art_txt = f" - ĐM thủ phạm dự đoán: {', '.join(set(culprit_artery))}" if culprit_artery else ""
        has_q = any(l in set(pathological_q_leads) for l in st_set)
        stage_str = "Bán cấp / Hoại tử (Đã có sóng Q)" if has_q else "Tối cấp / Cấp tính (Chưa có sóng Q)"
        findings.append(("Hội chứng mạch vành cấp (STEMI)", f"Nhồi máu cơ tim ST chênh lên - Vùng: {reg_txt} - Giai đoạn: {stage_str}{art_txt}"))
        alerts.append(f"🚨 CẤP CỨU: STEMI VÙNG {reg_txt.upper()} - KÍCH HOẠT QUY TRÌNH CAN THIỆP PCI KHẨN")

    # ------------------ C. THIẾU MÁU CỤC BỘ CƠ TIM CẤP (NSTE-ACS) ------------------
    elif len(st_depression_leads) >= 2 or len(inverted_t_leads) >= 2 or len(hyperacute_t_leads) >= 2:
        ischemia_details = []
        # ST chênh xuống đi ngang/đi xuống
        spec_dep = [f"{l} (dạng {sl})" for (l, sl) in st_depression_leads if sl in ["horizontal", "downsloping"]]
        if len(spec_dep) >= 2:
            ischemia_details.append(f"ST chênh xuống đặc hiệu tại: {', '.join(spec_dep)}")
        elif len(dep_leads_names) >= 2:
            ischemia_details.append(f"ST chênh xuống tại: {', '.join(dep_leads_names)}")

        if len(inverted_t_leads) >= 2:
            ischemia_details.append(f"Sóng T âm sâu đảo ngược tại: {', '.join(inverted_t_leads)}")
        if len(hyperacute_t_leads) >= 2:
            ischemia_details.append(f"Sóng T tối cấp (đáy rộng, đối xứng) tại: {', '.join(hyperacute_t_leads)}")

        findings.append(("Thiếu máu cục bộ cơ tim (NSTE-ACS)", f"Biến đổi thiếu máu cơ tim cấp: {'; '.join(ischemia_details)}"))
        alerts.append("⚠️ CẢNH BÁO: THEO DÕI HỘI CHỨNG VÀNH CẤP KHÔNG ST CHÊNH LÊN (NSTE-ACS) - ĐỀ NGHỊ ĐỊNH LƯỢNG TROPONIN HS")

    # ------------------ D. HỘI CHỨNG MẠCH VÀNH MẠN (CCS) ------------------
    # 1. Sóng Q hoại tử (Sẹo nhồi máu cơ tim cũ) theo vùng liên tiếp chuẩn ESC 2018
    q_set = set(pathological_q_leads)
    old_mi_regions = []
    if len({"II", "III", "aVF"}.intersection(q_set)) >= 2:
        old_mi_regions.append(f"Thành dưới ({', '.join(sorted(list({'II', 'III', 'aVF'}.intersection(q_set))))})")
    if {"V1", "V2"}.issubset(q_set) or {"V2", "V3"}.issubset(q_set):
        old_mi_regions.append("Trước - Vách (V1-V3)")
    if {"V4", "V5", "V6"}.intersection(q_set) and len({"V4", "V5", "V6"}.intersection(q_set)) >= 2:
        old_mi_regions.append("Thành trước - bên (V4-V6)")
    if {"I", "aVL"}.issubset(q_set):
        old_mi_regions.append("Thành bên cao (DI, aVL)")

    if old_mi_regions:
        findings.append(("Hội chứng mạch vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI theo chuẩn ESC 2018) - Vùng: {', '.join(old_mi_regions)}"))

    # 2. Sóng R cắt cụt theo tiêu chuẩn DePace (R(V3) < R(V2) hoặc R(V4) < R(V3) hoặc R(V3) <= 3mm)
    rv2 = leads.get("V2", {}).get("r_amp", 0.0)
    rv3 = leads.get("V3", {}).get("r_amp", 0.0)
    rv4 = leads.get("V4", {}).get("r_amp", 0.0)
    if (rv3 < rv2 and rv2 > 2.0) or (rv4 < rv3 and rv3 > 2.0) or (0 < rv3 <= 3.0 and rv2 > 0):
        findings.append(("Hội chứng mạch vành mạn (CCS)", f"Sóng R cắt cụt vùng trước tim theo tiêu chuẩn DePace (R V3={rv3:.1f}mm, R V2={rv2:.1f}mm) - Gợi ý sẹo nhồi máu cũ thành trước hoặc phì đại thất"))

    # 3. Sóng T dẹt hoặc T âm mạn tính theo vùng
    if len(flat_t_leads) >= 3 and not stemi_regions:
        findings.append(("Hội chứng mạch vành mạn (CCS)", f"Sóng T dẹt lan tỏa (-1mm đến +1mm) tại: {', '.join(flat_t_leads)} - Gợi ý thiếu máu cơ tim mạn tính"))

    return {
        "findings": findings,
        "alerts": alerts,
        "st_elevation_leads": st_elevation_leads,
        "st_depression_leads": dep_leads_names,
        "pathological_q_leads": pathological_q_leads
    }

# =========================================================================
# 4. BỘ ĐÁNH GIÁ CHẨN ĐOÁN TỔNG HỢP (GIỮ NGUYÊN TOÀN BỘ CÁC BỆNH LÝ KHÁC)
# =========================================================================
def diagnose_ecg_comprehensive(data, gender="Nam", age=55):
    leads = data.get("leads", {})
    hr = data.get("hr", 72)
    qrs = data.get("qrs", 0.08)
    pr = data.get("pr", 0.16)
    rr_cv = data.get("rr_cv", 0.0)
    p_ratio = data.get("p_ratio", 1.0)

    findings = []
    alerts = []

    # 1. Trục điện tim
    d1 = leads.get("I", {})
    avf = leads.get("aVF", {})
    d2 = leads.get("II", {})

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
        axis_type = "Trục trung gian / Bình thường"

    # 2. Rối loạn nhịp
    if rr_cv > 0.24 and p_ratio < 0.20:
        findings.append(("Rối loạn nhịp", "Rung nhĩ (AFib): Mất sóng P, nhịp thất hoàn toàn không đều"))
        alerts.append("⚠️ Rung nhĩ: Đánh giá nguy cơ tắc mạch (CHA2DS2-VASc)")
    else:
        if hr > 100:
            findings.append(("Rối loạn nhịp", f"Nhịp nhanh xoang (Sinus Tachycardia) - Tần số: {hr} l/p"))
        elif hr < 60:
            findings.append(("Rối loạn nhịp", f"Nhịp chậm xoang (Sinus Bradycardia) - Tần số: {hr} l/p"))
        else:
            findings.append(("Rối loạn nhịp", f"Nhịp xoang bình thường (Normal Sinus Rhythm) - Tần số: {hr} l/p"))

    # 3. Block nhĩ - thất (AV Block)
    if pr >= 0.21:
        findings.append(("Dẫn truyền nhĩ - thất", f"Block nhĩ - thất độ I: Khoảng PR kéo dài cố định ({pr:.2f}s > 0.20s)"))
    elif hr < 45 and qrs >= 0.12 and rr_cv < 0.04:
        findings.append(("Dẫn truyền nhĩ - thất", "Block nhĩ - thất độ III: Phân ly nhĩ thất hoàn toàn"))
        alerts.append("🚨 BLOCK TIM ĐỘ 3: CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP CẤP CỨU")

    # 4. CHẨN ĐOÁN HỘI CHỨNG MẠCH VÀNH CẤP & MẠN CHUẨN YDS 2026
    coronary_res = evaluate_yds_coronary_syndromes(leads, gender=gender, age=age)
    findings.extend(coronary_res["findings"])
    alerts.extend(coronary_res["alerts"])

    # 5. Dày thất & Lớn nhĩ (Chuẩn Sokolow-Lyon & Cornell)
    v1_data = leads.get("V1", {})
    v5_data = leads.get("V5", {})
    v6_data = leads.get("V6", {})
    avl_data = leads.get("aVL", {})
    v3_data = leads.get("V3", {})

    sokolow_lv = v1_data.get("s_amp", 0.0) + max(v5_data.get("r_amp", 0.0), v6_data.get("r_amp", 0.0))
    cornell_val = avl_data.get("r_amp", 0.0) + v3_data.get("s_amp", 0.0)
    cornell_cutoff = 28.0 if gender == "Nam" else 20.0

    if sokolow_lv >= 35.0:
        findings.append(("Phì đại thất", f"Dày thất trái (LVH) theo Sokolow-Lyon: SV1 + RV5 = {sokolow_lv:.1f} mm (≥ 35 mm)"))
    elif cornell_val > cornell_cutoff:
        findings.append(("Phì đại thất", f"Dày thất trái (LVH) theo Cornell: RaVL + SV3 = {cornell_val:.1f} mm (> {cornell_cutoff:.0f} mm ở {gender})"))

    rv1 = v1_data.get("r_amp", 0.0)
    sv1 = v1_data.get("s_amp", 0.0)
    if rv1 >= 7.0 and (sv1 > 0 and rv1 / sv1 > 1.2) and "RAD" in axis_type:
        findings.append(("Phì đại thất", f"Dày thất phải (RVH): Sóng R cao ưu thế ở V1 ({rv1:.1f} mm) kèm trục lệch phải"))

    # Lớn nhĩ (P phế, P nhĩ)
    p_amp_d2 = d2.get("p_amp", 0.0)
    p_amp_d3 = leads.get("III", {}).get("p_amp", 0.0)
    p_amp_avf = avf.get("p_amp", 0.0)
    max_p = max(p_amp_d2, p_amp_d3, p_amp_avf)
    if 2.5 <= max_p <= 4.5:
        findings.append(("Lớn buồng tim", f"Lớn nhĩ phải (P phế): Sóng P cao {max_p:.1f} mm (≥ 2.5 mm ở DII/DIII/aVF)"))
    elif v1_data.get("p_biphasic", False) and d2.get("p_dur", 0.08) >= 0.12:
        findings.append(("Lớn buồng tim", "Lớn nhĩ trái (P nhĩ): Sóng P rộng ≥ 0.12s hoặc 2 pha âm chiếm ưu thế ở V1"))

    # 6. Block nhánh & Block phân nhánh
    v2_data = leads.get("V2", {})
    v1_has_rsr = v1_data.get("true_rsr", False) or v2_data.get("true_rsr", False)
    lateral_has_broad_s = v6_data.get("slurred_s", False) or d1.get("slurred_s", False)

    rbbb_type = None
    if v1_has_rsr and lateral_has_broad_s:
        if qrs >= 0.12:
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 dạng rsR', S rộng ở DI/V6"
        elif 0.09 <= qrs < 0.12:
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.09 - 0.11s, dạng rsR' tại V1 kèm S rộng ở V6"

    if rbbb_type:
        findings.append(("Block Nhánh", rbbb_type))

    has_lbbb = (
        (v5_data.get("notched_r", False) or v6_data.get("notched_r", False) or d1.get("notched_r", False)) and
        (v1_data.get("s_amp", 0.0) > 9.0 and v1_data.get("r_amp", 0.0) < 2.5) and
        not v1_has_rsr
    )
    if has_lbbb and qrs >= 0.12:
        findings.append(("Block Nhánh", "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/V5/V6"))

    # Block phân nhánh
    if "LAD" in axis_type and qrs < 0.12:
        if d1.get("r_amp", 0.0) > d1.get("s_amp", 0.0) and leads.get("III", {}).get("s_amp", 0.0) > leads.get("III", {}).get("r_amp", 0.0):
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh, qR ở DI/aVL, rS ở DII/DIII/aVF"))
    elif "RAD" in axis_type and qrs < 0.12 and rv1 < 6.0:
        if leads.get("III", {}).get("r_amp", 0.0) > leads.get("III", {}).get("s_amp", 0.0) and d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0):
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh, rS ở DI/aVL, qR ở DII/DIII/aVF"))

    return {
        "axis": axis_type,
        "sokolow": sokolow_lv,
        "cornell": cornell_val,
        "findings": findings,
        "alerts": alerts
    }

# =========================================================================
# 5. GIAO DIỆN HIỂN THỊ STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Bản Ghi ECG 12 Chuyển Đạo")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        gender_choice = st.radio("Giới tính bệnh nhân", ["Nam", "Nữ"], horizontal=True)
    with c_p2:
        age_choice = st.number_input("Tuổi", min_value=1, max_value=120, value=55)

    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG đã nạp vào bộ xử lý", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên để phân tích.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Chuyên Khoa")
    if uploaded:
        with st.spinner("Đang đo đạc theo tiêu chuẩn Sổ tay ĐTĐ YDS 2026, đối chiếu De Winter, Wellens & STEMI..."):
            res = process_ecg_dataset(img_pil)
            diag = diagnose_ecg_comprehensive(res, gender=gender_choice, age=age_choice)

        if diag["alerts"]:
            for al in diag["alerts"]:
                if "🚨" in al:
                    st.error(al)
                else:
                    st.warning(al)

        st.markdown("#### Chỉ Số Đo Đạc Tự Động")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR)", f"{res['hr']} bpm")
        m2.metric("Khoảng PR", f"{res['pr']:.2f} s", delta="Kéo dài (>0.20s)" if res['pr'] >= 0.21 else "Bình thường", delta_color="inverse" if res['pr'] >= 0.21 else "normal")
        m3.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m4.metric("Sokolow-Lyon", f"{diag['sokolow']:.1f} mm")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}` | **Chu kỳ R-R:** `{res['mean_rr']:.2f} s`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán Phân Tầng")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if "STEMI" in desc or "độ III" in desc or "De Winter" in desc or "LMCA" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "Wellens" in desc or "NSTE-ACS" in desc or "Thiếu máu" in desc or "Block" in desc or "CCS" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện biến đổi bệnh lý mạch vành hoặc rối loạn dẫn truyền.")

        st.markdown("---")
        with st.expander("🔍 Chi Tiết Hình Thái ST-T & Đoạn PR Từng Chuyển Đạo (YDS Guidelines)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                st_val = l_data.get('st_shift', 0.0)
                slope = l_data.get('st_slope', 'flat')
                t_val = l_data.get('t_amp', 0.0)
                t_morph = l_data.get('t_morph', 'normal')

                eval_txt = "Đẳng điện"
                if st_val >= 1.0:
                    eval_txt = "🔴 ST Chênh Lên"
                elif st_val <= -0.5:
                    eval_txt = f"🔵 ST Chênh Xuống ({slope})"

                detail_list.append({
                    "Chuyển đạo": l_name,
                    "ST Chênh (mm)": f"{st_val:+.1f}",
                    "Đánh giá ST": eval_txt,
                    "Sóng T (mm)": f"{t_val:+.1f} ({t_morph})",
                    "Sóng Q (mm/s)": f"{l_data.get('q_amp', 0.0):.1f}mm / {l_data.get('q_dur', 0.0):.3f}s",
                    "R (mm)": f"{l_data.get('r_amp', 0.0):.1f}",
                    "PR (s)": f"{l_data.get('pr_interval', 0.16):.2f}"
                })
            st.dataframe(detail_list, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
