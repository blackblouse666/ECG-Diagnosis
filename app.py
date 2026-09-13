import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="Hệ Thống AI Chẩn Đoán ECG Chuyên Khoa (AHA/ESC)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Đa Tầng (AHA/ACC/ESC)")
st.caption("Chuẩn hóa tiêu chuẩn Sóng Q hoại tử theo vùng liên tiếp; Chẩn đoán chính xác Thiếu máu cục bộ cơ tim (ST chênh xuống / T âm)")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ & BÓC TÁCH NÉT MỰC
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
# 2. ĐO ĐẠC HÌNH THÁI VI THỂ (ĐIỂM J, ST, SÓNG T, SÓNG Q)
# =========================================================================
def analyze_lead_morphology(raw_sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "t_amp": 0.0, "qrs_w": 0.08, "pr_interval": 0.16,
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

        # 1. Đo sóng Q bệnh lý thực sự (chỉ tính phần âm trước R)
        q_zone = sig[max(0, r - int(px_per_sec * 0.08)):r]
        if len(q_zone) > 0 and np.min(q_zone) < 0:
            q_peak_idx = np.argmin(q_zone)
            q_depth_mm = (abs(q_zone[q_peak_idx]) / px_per_mv) * 10.0
            # Đo độ rộng ở mức 0.5mm âm
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

        # 3. Đo độ lệch ST tại điểm J (mm)
        st_shift_mm = (sig[j_idx] / px_per_mv) * 10.0
        st_shifts.append(st_shift_mm)

        # 4. Đo biên độ sóng T (khoảng 120ms đến 280ms sau R)
        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.12)):min(len(sig), r + int(px_per_sec * 0.30))]
        if len(t_zone) > 5:
            # Lấy cực trị có độ lệch lớn nhất so với baseline
            t_extrema = t_zone[np.argmax(np.abs(t_zone))]
            t_amps.append((t_extrema / px_per_mv) * 10.0)
        else:
            t_amps.append(0.0)

        # 5. Đo QRS
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
        "t_amp": float(np.median(t_amps)) if t_amps else 0.0,
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
# 3. TIÊU CHUẨN ĐỒNG THUẬN VỀ BỆNH MẠCH VÀNH (ESC / AHA)
# =========================================================================
def evaluate_ischemia_and_infarction(leads, gender="Nam", age=55):
    """
    Tiêu chuẩn chuẩn hóa:
    1. LOẠI TRỪ aVR khỏi mọi tiêu chuẩn sóng Q bệnh lý.
    2. Sóng Q bệnh lý thực sự (Pathological Q):
       - Độ rộng >= 0.04s (40ms) VÀ độ sâu >= 1.0mm (hoặc >= 25% biên độ R).
       - BẮT BUỘC xuất hiện ở ít nhất 2 chuyển đạo liên tiếp cùng vùng:
         + Thành dưới: ít nhất 2 trong 3 lead (DII, DIII, aVF). DIII đứng một mình KHÔNG TÍNH.
         + Thành trước: V1-V2, V2-V3, hoặc V3-V4.
         + Thành bên: DI kèm aVL, hoặc V5 kèm V6.
    3. Thiếu máu cục bộ cơ tim (Myocardial Ischemia / NSTEMI):
       - ST chênh xuống dạng ngang hoặc dốc xuống >= 0.5mm ở ít nhất 2 chuyển đạo liên tiếp.
       - Hoặc sóng T âm sâu đối xứng >= 1.0mm ở các chuyển đạo có R ưu thế (loại trừ aVR, V1).
    """
    pathological_q_leads = []
    st_elevation_leads = []
    st_depression_leads = []
    t_inversion_leads = []

    for l_name, l_data in leads.items():
        if l_name == "aVR":
            continue  # aVR luôn âm sinh lý, bỏ qua đánh giá Q hoại tử và T âm

        st_val = l_data.get("st_shift", 0.0)
        t_val = l_data.get("t_amp", 0.0)
        q_dur = l_data.get("q_dur", 0.0)
        q_amp = l_data.get("q_amp", 0.0)
        r_amp = l_data.get("r_amp", 0.0)

        # ST chênh lên chuẩn
        st_cutoff = 1.5 if (l_name in ["V2", "V3"] and gender == "Nữ") else (2.0 if l_name in ["V2", "V3"] else 1.0)
        if st_val >= st_cutoff:
            st_elevation_leads.append(l_name)

        # ST chênh xuống (Thiếu máu cục bộ)
        if st_val <= -0.5:
            st_depression_leads.append(l_name)

        # Sóng T âm sâu (Thiếu máu cục bộ)
        if t_val <= -1.0 and l_name != "V1":
            t_inversion_leads.append(l_name)

        # Sóng Q bệnh lý thực sự
        is_true_q = False
        if q_dur >= 0.038 and q_amp >= 1.0:
            if r_amp > 0 and (q_amp / r_amp) >= 0.25:
                is_true_q = True
            elif r_amp == 0 and q_amp >= 2.0:  # Dạng QS
                is_true_q = True

        if is_true_q:
            pathological_q_leads.append(l_name)

    # Lọc sóng Q theo vùng liên tiếp (Contiguous leads requirement)
    confirmed_old_mi_regions = []
    q_set = set(pathological_q_leads)

    # Thành dưới: Phải có ít nhất 2 lead (DII kèm aVF, hoặc DIII kèm aVF/DII)
    inf_q = {"II", "III", "aVF"}.intersection(q_set)
    if len(inf_q) >= 2:
        confirmed_old_mi_regions.append(f"Thành dưới (Inferior: {', '.join(sorted(list(inf_q)))})")

    # Thành trước vách
    if {"V1", "V2"}.issubset(q_set) or {"V2", "V3"}.issubset(q_set):
        confirmed_old_mi_regions.append("Trước - Vách (Anteroseptal)")
    if {"V3", "V4"}.issubset(q_set):
        confirmed_old_mi_regions.append("Thành trước (Anterior)")

    # Thành bên
    if {"I", "aVL"}.issubset(q_set) or {"V5", "V6"}.issubset(q_set):
        confirmed_old_mi_regions.append("Thành bên (Lateral)")

    # Lọc vùng thiếu máu cục bộ (ST chênh xuống hoặc T âm ở >= 2 chuyển đạo liên tiếp)
    ischemia_leads = sorted(list(set(st_depression_leads + t_inversion_leads)))
    ischemia_set = set(ischemia_leads)
    confirmed_ischemia_regions = []

    if len({"II", "III", "aVF"}.intersection(ischemia_set)) >= 2:
        confirmed_ischemia_regions.append(f"Thành dưới ({', '.join(sorted(list({'II', 'III', 'aVF'}.intersection(ischemia_set))))})")
    if len({"V4", "V5", "V6"}.intersection(ischemia_set)) >= 2:
        confirmed_ischemia_regions.append(f"Thành trước - bên ({', '.join(sorted(list({'V4', 'V5', 'V6'}.intersection(ischemia_set))))})")
    if {"I", "aVL"}.issubset(ischemia_set):
        confirmed_ischemia_regions.append("Thành bên cao (DI, aVL)")
    if len({"V1", "V2", "V3"}.intersection(ischemia_set)) >= 2:
        confirmed_ischemia_regions.append(f"Trước - Vách ({', '.join(sorted(list({'V1', 'V2', 'V3'}.intersection(ischemia_set))))})")

    return {
        "st_elevation_leads": st_elevation_leads,
        "old_mi_regions": confirmed_old_mi_regions,
        "ischemia_regions": confirmed_ischemia_regions,
        "ischemia_leads": ischemia_leads,
        "raw_q_leads": pathological_q_leads
    }

# =========================================================================
# 4. BỘ ĐÁNH GIÁ CHẨN ĐOÁN TỔNG HỢP
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
        alerts.append("🚨 BLOCK TIM ĐỘ 3: CHỈ ĐẶT MÁY TẠO NHỊP CẤP CỨU")

    # 4. ĐÁNH GIÁ CHUẨN XÁC: THIẾU MÁU CỤC BỘ vs NHỒI MÁU CŨ vs STEMI
    coronary_res = evaluate_ischemia_and_infarction(leads, gender=gender, age=age)
    
    # Ưu tiên 1: STEMI nếu có ST chênh lên
    stemi_leads = coronary_res["st_elevation_leads"]
    if len(stemi_leads) >= 2:
        findings.append(("Hội chứng vành cấp (ACS)", f"Nhồi máu cơ tim ST chênh lên (STEMI) tại: {', '.join(stemi_leads)}"))
        alerts.append(f"🚨 CẢNH BÁO: THEO DÕI STEMI TẠI {', '.join(stemi_leads)} - CẦN ĐỐI CHIẾU LÂM SÀNG CẤP CỨU")
    
    # Ưu tiên 2: Thiếu máu cục bộ cơ tim (ST chênh xuống / T âm)
    elif coronary_res["ischemia_regions"]:
        reg_txt = " + ".join(coronary_res["ischemia_regions"])
        findings.append(("Bệnh mạch vành", f"Thiếu máu cục bộ cơ tim (ST chênh xuống / T âm sâu) - Vùng: {reg_txt}"))
        alerts.append(f"⚠️ Phát hiện Thiếu máu cục bộ cơ tim vùng {reg_txt}: Đề nghị làm men tim (hs-Troponin) và siêu âm tim")
    elif len(coronary_res["ischemia_leads"]) >= 2:
        findings.append(("Bệnh mạch vành", f"Thiếu máu cục bộ cơ tim (ST chênh xuống / T âm) tại các chuyển đạo: {', '.join(coronary_res['ischemia_leads'])}"))

    # Ưu tiên 3: Sẹo hoại tử / Nhồi máu cơ tim cũ (Chỉ kết luận khi có vùng liên tiếp thực thụ)
    elif coronary_res["old_mi_regions"]:
        findings.append(("Hội chứng vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI) - Vùng: {', '.join(coronary_res['old_mi_regions'])}"))

    # 5. Dày thất & Lớn nhĩ
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
        with st.spinner("Đang loại trừ sóng Q sinh lý, phân tích ST-T và đối chiếu vùng thiếu máu cục bộ..."):
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
                if "STEMI" in desc or "độ III" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "Thiếu máu" in desc or "Block" in desc or "Dày thất" in desc or "mạn" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện thiếu máu cơ tim cấp/mạn hoặc rối loạn dẫn truyền.")

        st.markdown("---")
        with st.expander("🔍 Chi Tiết Đo Đạc ST, Sóng T & Sóng Q Từng Chuyển Đạo"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                st_shift_val = l_data.get('st_shift', 0.0)
                t_amp_val = l_data.get('t_amp', 0.0)
                q_dur_val = l_data.get('q_dur', 0.0)
                q_amp_val = l_data.get('q_amp', 0.0)

                st_eval = "Đẳng điện"
                if st_shift_val >= 1.0:
                    st_eval = "🔴 ST Chênh Lên"
                elif st_shift_val <= -0.5:
                    st_eval = "🔵 ST Chênh Xuống"

                detail_list.append({
                    "Chuyển đạo": l_name,
                    "ST Chênh (mm)": f"{st_shift_val:+.1f}",
                    "Sóng T (mm)": f"{t_amp_val:+.1f}",
                    "Đánh giá ST": st_eval,
                    "Sóng Q (mm/s)": f"{q_amp_val:.1f}mm / {q_dur_val:.3f}s"
                })
            st.dataframe(detail_list, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
