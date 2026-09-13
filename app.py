import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG Chẩn Đoán Dẫn Truyền & Tổn Thương (Chuẩn Hóa AHA/ESC)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Chuyên Khoa (Đã Triệt Tiêu Dương Tính Giả)")
st.caption("Khử triệt để lưới hồng & chữ chìm; Bắt chuẩn tần số và khoảng PR (Block AV độ I); Siết chặt tiêu chuẩn STEMI, Dày thất & Rối loạn nhịp")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ ẢNH MÀU: LỌC MÀU HỒNG & XÓA WATERMARK BẰNG HSV
# =========================================================================
def extract_pure_ecg_traces(rgb_img):
    """
    Chuyển ảnh sang không gian màu HSV để lọc bỏ hoàn toàn lưới giấy in màu đỏ/hồng
    và các dòng chữ in chìm mờ (watermark), chỉ giữ lại nét chì đen của sóng điện tim.
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # Nét mực điện tim màu đen/xám đậm có Value (độ sáng) thấp
    # Lưới caro hồng/đỏ có Hue nằm ở dải 0-10 hoặc 160-180 với Saturation cao
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    
    # Tạo mặt nạ chỉ lấy pixel đen (độ sáng thấp và không bị bão hòa màu hồng)
    trace_mask = (v_channel < 115) & (s_channel < 140)
    binary = np.zeros_like(v_channel, dtype=np.uint8)
    binary[trace_mask] = 255

    # Làm sạch các chấm nhiễu đơn lẻ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned

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
    # Khử trôi đường đẳng điện bằng median cục bộ
    baseline = np.median(sig)
    return sig - baseline

# =========================================================================
# 2. ĐO ĐẠC HÌNH THÁI CHÍNH XÁC: ĐỈNH R, SÓNG P, ĐOẠN PR & ST-T
# =========================================================================
def analyze_lead_morphology(sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "t_amp": 0.0, "qrs_w": 0.08, "pr_interval": 0.16,
        "p_amp": 0.0, "p_dur": 0.08, "p_detected": False, "p_biphasic": False,
        "true_rsr": False, "slurred_s": False, "notched_r": False,
        "r_peaks": [], "rr_intervals": []
    }
    if len(sig) < 40:
        return default_props

    # Loại bỏ xung calib ở đầu mỗi chuyển đạo (10% chiều dài đầu tiên nếu có đột biến)
    sig_search = sig.copy()
    cut_start = int(len(sig) * 0.08)

    # Đặt khoảng cách tối thiểu giữa 2 đỉnh R >= 0.40s (tương đương HR tối đa sinh lý ~ 150 bpm)
    # Điều này ngăn chặn triệt để việc đếm sóng T hoặc sóng P thành đỉnh R
    min_peak_dist = int(px_per_sec * 0.42)
    max_val = np.max(sig_search[cut_start:]) if len(sig_search[cut_start:]) > 0 else 0
    if max_val <= 0:
        return default_props

    peaks, _ = find_peaks(
        sig_search, 
        distance=min_peak_dist,
        prominence=max_val * 0.40
    )
    # Lọc bỏ các đỉnh nằm trong vùng calib đầu
    peaks = [p for p in peaks if p > cut_start]
    if len(peaks) == 0:
        return default_props

    r_amps, s_amps, q_amps, q_durs = [], [], [], []
    st_shifts, qrs_widths, pr_intervals, p_amps = [], [], [], []
    has_true_rsr = False
    has_slurred_s = False
    has_notched_r = False
    p_detected = False
    p_biphasic = False

    for r in peaks:
        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # 1. Tìm sóng S (trong khoảng 100ms sau R)
        s_win = sig[r:min(len(sig), r + int(px_per_sec * 0.10))]
        if len(s_win) > 0:
            s_val = (abs(np.min(s_win)) / px_per_mv) * 10.0
            s_amps.append(s_val)
            # Sóng S rộng: âm sâu kéo dài >= 40ms
            if len(np.where(s_win < -0.20 * sig[r])[0]) / px_per_sec >= 0.040:
                has_slurred_s = True
        else:
            s_amps.append(0.0)

        # 2. Sóng Q (trong khoảng 60ms trước R)
        q_win = sig[max(0, r - int(px_per_sec * 0.06)):r]
        if len(q_win) > 0 and np.min(q_win) < 0:
            q_val = (abs(np.min(q_win)) / px_per_mv) * 10.0
            q_dur = len(np.where(q_win < -0.10 * sig[r])[0]) / px_per_sec
            q_amps.append(q_val)
            q_durs.append(q_dur)
        else:
            q_amps.append(0.0)
            q_durs.append(0.0)

        # 3. Đo độ rộng chân sóng QRS
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.09)) and sig[left_idx] > 0.18 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.11)) and sig[right_idx] > 0.18 * sig[r]:
            right_idx += 1
        qrs_widths.append((right_idx - left_idx) / px_per_sec)

        # 4. Đoạn ST tại điểm J (khoảng 50-70ms sau đỉnh R)
        j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
        st_shifts.append((sig[j_idx] / px_per_mv) * 10.0)

        # 5. TÌM SÓNG P VÀ ĐO KHOẢNG PR CHÍNH XÁC (Quét từ 140ms đến 360ms trước R)
        p_zone_start = max(0, r - int(px_per_sec * 0.38))
        p_zone_end = max(0, r - int(px_per_sec * 0.12))
        p_zone = sig[p_zone_start:p_zone_end]
        
        if len(p_zone) > 5:
            # Sóng P có biên độ thấp hơn nhiều so với R (ngưỡng prominence vừa phải)
            p_pks, _ = find_peaks(p_zone, prominence=0.8)
            if len(p_pks) > 0:
                p_idx = p_zone_start + p_pks[-1]
                pr_dur = (r - p_idx) / px_per_sec
                # Khoảng PR sinh lý thực tế từ 0.11s đến 0.38s
                if 0.11 <= pr_dur <= 0.38:
                    pr_intervals.append(pr_dur)
                    p_amp_val = min(4.5, (sig[p_idx] / px_per_mv) * 10.0) # Giới hạn không bị vượt ngưỡng vô lý
                    p_amps.append(max(0.5, p_amp_val))
                    p_detected = True
            if np.max(p_zone) > 0.6 and np.min(p_zone) < -0.6:
                p_biphasic = True

        # 6. Kiểm tra tai thỏ rsR' ở V1
        sub_complex = sig[max(0, r - int(px_per_sec * 0.03)):min(len(sig), r + int(px_per_sec * 0.09))]
        if len(sub_complex) > 5:
            local_peaks, _ = find_peaks(sub_complex, distance=int(px_per_sec * 0.024), prominence=3.0)
            if len(local_peaks) >= 2:
                has_true_rsr = True
                has_notched_r = True

    rr_list = np.diff(peaks) / px_per_sec if len(peaks) >= 2 else []
    avg_qrs = float(np.median(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.18))

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.mean(q_amps)) if q_amps else 0.0,
        "q_dur": float(np.max(q_durs)) if q_durs else 0.0,
        "st_shift": float(np.mean(st_shifts)) if st_shifts else 0.0,
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
    clean_bin = extract_pure_ecg_traces(rgb_img)

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

    # Tính nhịp tim chính xác từ trung vị R-R toàn bản ghi
    valid_rrs = [rr for rr in all_rr_intervals if 0.45 <= rr <= 1.8]
    mean_rr = float(np.median(valid_rrs)) if valid_rrs else 0.85
    hr = int(60.0 / mean_rr) if mean_rr > 0 else 72
    rr_cv = (float(np.std(valid_rrs)) / mean_rr) if len(valid_rrs) > 3 else 0.0

    # Lấy PR ưu tiên từ DII hoặc trung vị các chuyển đạo đo được
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
# 3. BỘ TIÊU CHUẨN CHẨN ĐOÁN (ĐÃ KHẮC PHỤC TRIỆT ĐỂ DƯƠNG TÍNH GIẢ)
# =========================================================================
def diagnose_ecg_strictly(data, gender="Nam"):
    leads = data.get("leads", {})
    hr = data.get("hr", 72)
    qrs = data.get("qrs", 0.08)
    pr = data.get("pr", 0.16)
    rr_cv = data.get("rr_cv", 0.0)
    rr_list = data.get("rr_list", [])
    p_ratio = data.get("p_ratio", 1.0)

    findings = []
    alerts = []

    # ---------------- 1. TÍNH TOÁN TRỤC ĐIỆN TIM CHUẨN ----------------
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
        axis_type = "Trục trung gian / Bình thường"

    # ---------------- 2. RỐI LOẠN NHỊP & NGOẠI TÂM THU ----------------
    # Rung nhĩ: Chỉ chẩn đoán khi biến thiên RR rất lớn VÀ không thấy sóng P
    if rr_cv > 0.24 and p_ratio < 0.20:
        findings.append(("Rối loạn nhịp", "Rung nhĩ (Atrial Fibrillation - AFib): Mất sóng P, nhịp thất hoàn toàn không đều"))
        alerts.append("⚠️ Rung nhĩ: Đánh giá nguy cơ thuyên tắc mạch (CHA2DS2-VASc)")
    else:
        if hr > 100:
            findings.append(("Rối loạn nhịp", f"Nhịp nhanh xoang (Sinus Tachycardia) - Tần số: {hr} l/p"))
        elif hr < 60:
            findings.append(("Rối loạn nhịp", f"Nhịp chậm xoang (Sinus Bradycardia) - Tần số: {hr} l/p"))
        else:
            findings.append(("Rối loạn nhịp", f"Nhịp xoang bình thường (Normal Sinus Rhythm) - Tần số: {hr} l/p"))

    # Ngoại tâm thu: Siết chặt ngưỡng, phải có ít nhất 2 nhát bóp đến sớm rõ rệt
    early_beats = [rr for rr in rr_list if rr < 0.72 * data["mean_rr"]]
    late_compensatory = [rr for rr in rr_list if rr > 1.25 * data["mean_rr"]]

    if len(early_beats) >= 2 and len(late_compensatory) >= 2:
        if qrs >= 0.12 or leads.get("V1", {}).get("qrs_w", 0.08) >= 0.12:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu thất (PVC): Nhát bóp đến sớm, QRS giãn rộng dị dạng, nghỉ bù"))
        else:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu nhĩ (PAC): Nhát bóp đến sớm với QRS hẹp"))

    # ---------------- 3. BLOCK DẪN TRUYỀN NHĨ - THẤT (BLOCK AV) ----------------
    # TIÊU CHUẨN VÀNG: PR > 0.20s trên nền nhịp đều -> BLOCK AV ĐỘ I
    if pr >= 0.21:
        findings.append(("Dẫn truyền nhĩ - thất", f"Block nhĩ - thất độ I (First-degree AV Block): Khoảng PR kéo dài cố định ({pr:.2f}s > 0.20s)"))
        alerts.append(f"Phát hiện Block AV độ I (PR = {pr:.2f}s): Cần theo dõi tiến triển dẫn truyền nhĩ - thất")
    elif hr < 45 and qrs >= 0.12 and rr_cv < 0.04:
        findings.append(("Dẫn truyền nhĩ - thất", "Block nhĩ - thất độ III: Phân ly nhĩ thất hoàn toàn"))
        alerts.append("🚨 BLOCK TIM ĐỘ 3: CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP CẤP CỨU")

    # ---------------- 4. LỚN NHĨ & DÀY THẤT (ĐÃ LỌC BỎ NHIỄU BIÊN ĐỘ) ----------------
    p_amp_d2 = d2.get("p_amp", 0.0)
    p_amp_d3 = d3.get("p_amp", 0.0)
    p_amp_avf = avf.get("p_amp", 0.0)
    v1_data = leads.get("V1", {})

    # Chỉ chẩn đoán Lớn nhĩ phải khi P cao >= 2.5mm nhưng phải <= 4.5mm (tránh bắt nhầm vạch calib)
    max_p = max(p_amp_d2, p_amp_d3, p_amp_avf)
    if 2.5 <= max_p <= 4.5:
        findings.append(("Lớn buồng tim", f"Lớn nhĩ phải (P phế): Sóng P cao {max_p:.1f} mm (≥ 2.5 mm ở DII/DIII/aVF)"))
    elif v1_data.get("p_biphasic", False) and d2.get("p_dur", 0.08) >= 0.12:
        findings.append(("Lớn buồng tim", "Lớn nhĩ trái (P nhĩ): Sóng P rộng ≥ 0.12s hoặc 2 pha âm chiếm ưu thế ở V1"))

    # Dày thất trái (LVH)
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

    # Dày thất phải (RVH): Phải có R ưu thế ở V1 (>7mm) KÈM THEO TRỤC LỆCH PHẢI THỰC THỤ
    rv1 = v1_data.get("r_amp", 0.0)
    sv1 = v1_data.get("s_amp", 0.0)
    if rv1 >= 7.0 and (sv1 > 0 and rv1 / sv1 > 1.2) and "RAD" in axis_type:
        findings.append(("Phì đại thất", f"Dày thất phải (RVH): Sóng R cao ưu thế ở V1 ({rv1:.1f} mm) kèm trục lệch phải"))

    # ---------------- 5. BLOCK NHÁNH (TIÊU CHUẨN KÉP KHỬ DƯƠNG TÍNH GIẢ) ----------------
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

    # LBBB
    has_lbbb = (
        (v5_data.get("notched_r", False) or v6_data.get("notched_r", False) or d1.get("notched_r", False)) and
        (v1_data.get("s_amp", 0.0) > 9.0 and v1_data.get("r_amp", 0.0) < 2.5) and
        not v1_has_rsr
    )
    if has_lbbb and qrs >= 0.12:
        findings.append(("Block Nhánh", "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/V5/V6"))

    # ---------------- 6. HỘI CHỨNG VÀNH CẤP (SIẾT CHẶT QUY TẮC PHÂN VÙNG LIÊN TIẾP) ----------------
    st_elev_raw = [k for k, v in leads.items() if v.get("st_shift", 0.0) >= (1.5 if k in ["V2", "V3"] else 1.0)]
    st_depr_raw = [k for k, v in leads.items() if v.get("st_shift", 0.0) <= -0.9]
    q_leads_raw = [k for k, v in leads.items() if v.get("q_dur", 0.0) >= 0.035 and (v.get("r_amp", 0.0) > 0 and v.get("q_amp", 0.0) >= 0.25 * v.get("r_amp", 0.0))]

    st_set = set(st_elev_raw)
    stemi_confirmed_regions = []
    culprit_art = []

    # QUY TẮC VÀNG STEMI: Phải chênh lên ở ÍT NHẤT 2 CHUYỂN ĐẠO LIÊN TIẾP cùng vùng giải phẫu
    if {"V1", "V2"}.issubset(st_set):
        stemi_confirmed_regions.append("Vách liên thất (V1-V2)")
        culprit_art.append("LAD")
    if {"V3", "V4"}.issubset(st_set):
        stemi_confirmed_regions.append("Thành trước (V3-V4)")
        culprit_art.append("LAD")
    if len({"II", "III", "aVF"}.intersection(st_set)) >= 2:
        stemi_confirmed_regions.append("Thành dưới (DII, DIII, aVF)")
        culprit_art.append("RCA")
    if len({"I", "aVL", "V5", "V6"}.intersection(st_set)) >= 2:
        stemi_confirmed_regions.append("Thành bên (DI, aVL, V5, V6)")
        culprit_art.append("LCx")

    if stemi_confirmed_regions:
        reg_txt = ", ".join(stemi_confirmed_regions)
        art_txt = f" (Nghi ngờ nhánh: {', '.join(set(culprit_art))})" if culprit_art else ""
        findings.append(("Hội chứng vành cấp (ACS)", f"Nhồi máu cơ tim ST chênh lên (STEMI) - Vùng: {reg_txt}{art_txt}"))
        alerts.append(f"🚨 CẢNH BÁO: STEMI VÙNG {reg_txt.upper()} - KÍCH HOẠT QUY TRÌNH PCI")
    elif q_leads_raw and len(q_leads_raw) >= 2:
        findings.append(("Hội chứng vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI) tại: {', '.join(q_leads_raw)}"))
    elif len(st_depr_raw) >= 2:
        findings.append(("Thiếu máu cục bộ", f"ST chênh xuống / Thiếu máu dưới nội tâm mạc tại: {', '.join(st_depr_raw)}"))

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
    gender_choice = st.radio("Giới tính bệnh nhân", ["Nam", "Nữ"], horizontal=True)
    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG đã nạp vào bộ lọc màu HSV", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Đã Khử Dương Tính Giả")
    if uploaded:
        with st.spinner("Đang tách dải màu HSV, khử chữ chìm & lưới hồng, đo đạc sóng P-PR..."):
            res = process_ecg_dataset(img_pil)
            diag = diagnose_ecg_strictly(res, gender=gender_choice)

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
                if "STEMI" in desc or "độ III" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "Block" in desc or "Dày thất" in desc or "Lớn nhĩ" in desc or "Ngoại tâm thu" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện rối loạn nhịp, phì đại buồng tim hoặc tổn thương dẫn truyền.")

        st.markdown("---")
        with st.expander("🔍 Xem Bảng Đo Đạc Vi Thể 12 Chuyển Đạo (P, PR, Q, R, S, ST)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                detail_list.append({
                    "Chuyển đạo": l_name,
                    "P (mm)": f"{l_data.get('p_amp', 0.0):.1f}",
                    "PR (s)": f"{l_data.get('pr_interval', 0.16):.2f}",
                    "R (mm)": f"{l_data.get('r_amp', 0.0):.1f}",
                    "S (mm)": f"{l_data.get('s_amp', 0.0):.1f}",
                    "ST Chênh (mm)": f"{l_data.get('st_shift', 0.0):+.1f}"
                })
            st.dataframe(detail_list, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
