import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI ECG Chẩn Đoán Chuyên Khoa Toàn Diện (YDS 2026)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Chuyên Khoa Toàn Diện (YDS 2026)")
st.caption("Khử nhiễu CAPS SCAN, Hội chứng vành cấp & mạn, Block nhĩ thất, Dẫn truyền nội thất, Kích thích sớm, SVT, VT, Lớn nhĩ & Dày thất")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ ẢNH & BÓC TÁCH NÉT MỰC
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
# 2. ĐO ĐẠC HÌNH THÁI VI THỂ ĐA THAM SỐ
# =========================================================================
def analyze_lead_morphology(raw_sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "st_slope": "flat", "t_amp": 0.0, "t_morph": "normal",
        "qrs_w": 0.08, "pr_interval": 0.16, "pr_list": [], "rp_interval": 0.0,
        "p_amp": 0.0, "p_pos_amp": 0.0, "p_neg_amp": 0.0, "p_neg_dur": 0.0,
        "p_dur": 0.08, "p_peaks_dist": 0.0, "p_notched": False, "p_detected": False,
        "p_inverted": False, "pseudo_r_prime": False, "vat": 0.03,
        "has_delta": False, "true_rsr": False, "slurred_s": False, "notched_r": False,
        "r_to_s_time": 0.0, "has_josephson": False, "left_rabbit_ear": False,
        "is_flatline": False, "has_notch_sign": False,
        "r_peaks": [], "rr_intervals": [], "qrs_list": []
    }
    if len(raw_sig) < 40:
        return default_props

    amp_span_raw = np.max(raw_sig) - np.min(raw_sig)
    if amp_span_raw < 1.0:
        default_props["is_flatline"] = True
        return default_props

    cut_start = int(len(raw_sig) * 0.05)
    sig_search = raw_sig[cut_start:]
    if len(sig_search) == 0:
        return default_props

    amp_span = np.max(sig_search) - np.min(sig_search)
    prom_val = amp_span * 0.22 if amp_span > 5.0 else 2.5

    peaks, _ = find_peaks(raw_sig, distance=int(px_per_sec * 0.20), prominence=prom_val)
    peaks = [p for p in peaks if p > cut_start]
    if len(peaks) == 0:
        peaks, _ = find_peaks(raw_sig, distance=int(px_per_sec * 0.18), prominence=1.8)
        peaks = [p for p in peaks if p > cut_start]
        if len(peaks) == 0:
            return default_props

    r_amps, s_amps, q_amps, q_durs = [], [], [], []
    st_shifts, t_amps, qrs_widths, pr_intervals, p_amps = [], [], [], [], []
    p_pos_amps, p_neg_amps, p_neg_durs, p_durs_list = [], [], [], []
    rp_intervals, r_to_s_times, vats = [], [], []
    st_slopes, t_morphs = [], []
    has_delta = False
    has_true_rsr = False
    has_slurred_s = False
    has_notched_r = False
    p_detected = False
    p_inverted = False
    pseudo_r_prime = False
    has_josephson = False
    left_rabbit_ear = False
    p_notched = False
    p_peaks_dist_val = 0.0

    for r in peaks:
        pr_zone = raw_sig[max(0, r - int(px_per_sec * 0.14)):max(0, r - int(px_per_sec * 0.04))]
        baseline_val = np.median(pr_zone) if len(pr_zone) > 0 else np.median(raw_sig)
        sig = raw_sig - baseline_val

        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # Sóng Q
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

        # Sóng S & Điểm J
        s_search = sig[r:min(len(sig), r + int(px_per_sec * 0.14))]
        if len(s_search) > 2:
            min_s_idx = np.argmin(s_search)
            s_val = (abs(min(0.0, s_search[min_s_idx])) / px_per_mv) * 10.0
            s_amps.append(s_val)
            r_to_s_times.append(min_s_idx / px_per_sec)

            if min_s_idx > 3:
                s_downslope = s_search[:min_s_idx]
                if np.sum(np.diff(np.sign(np.diff(s_downslope))) > 0) >= 1:
                    has_josephson = True

            s_neg_pts = np.where(s_search < -0.15 * max(r_val * (px_per_mv / 10.0), 4.0))[0]
            if len(s_neg_pts) / px_per_sec >= 0.038:
                has_slurred_s = True

            j_search_zone = s_search[max(1, int(px_per_sec * 0.04)):min(len(s_search), int(px_per_sec * 0.09))]
            if len(j_search_zone) > 0:
                j_idx = r + max(1, int(px_per_sec * 0.04)) + int(np.argmin(np.abs(np.diff(j_search_zone, prepend=j_search_zone[0]))))
            else:
                j_idx = r + min_s_idx
        else:
            j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
            s_amps.append(0.0)
            r_to_s_times.append(0.06)

        # ST chênh
        st_shift_mm = (sig[j_idx] / px_per_mv) * 10.0
        st_shifts.append(st_shift_mm)
        j80_idx = min(len(sig) - 1, j_idx + int(px_per_sec * 0.07))
        st_j80_shift = (sig[j80_idx] / px_per_mv) * 10.0
        slope_diff = st_j80_shift - st_shift_mm
        st_slopes.append("upsloping" if slope_diff > 0.4 else ("downsloping" if slope_diff < -0.4 else "horizontal"))

        # Sóng T
        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.10)):min(len(sig), r + int(px_per_sec * 0.35))]
        if len(t_zone) > 6:
            t_max = np.max(t_zone)
            t_min = np.min(t_zone)
            t_amp_val = (t_min / px_per_mv) * 10.0 if abs(t_min) > abs(t_max) else (t_max / px_per_mv) * 10.0
            t_amps.append(t_amp_val)
            if -1.0 <= t_amp_val <= 1.0:
                t_morphs.append("flat")
            elif t_amp_val <= -1.0:
                t_morphs.append("inverted")
            elif t_max > 1.0 and t_min < -1.0:
                t_morphs.append("biphasic_pos_neg" if np.argmax(t_zone) < np.argmin(t_zone) else "biphasic_neg_pos")
            else:
                t_morphs.append("positive")
        else:
            t_amps.append(0.0)
            t_morphs.append("normal")

        # Độ rộng QRS và VAT
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.12)) and sig[left_idx] > 0.15 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.16)) and sig[right_idx] > 0.15 * sig[r]:
            right_idx += 1
        measured_qrs = (right_idx - left_idx) / px_per_sec
        qrs_widths.append(measured_qrs)
        vats.append((r - left_idx) / px_per_sec)

        # Sóng Delta
        delta_zone = sig[left_idx:r]
        if len(delta_zone) > 5:
            first_half = delta_zone[:len(delta_zone)//2]
            second_half = delta_zone[len(delta_zone)//2:]
            s1 = np.mean(np.diff(first_half)) if len(first_half) > 1 else 0
            s2 = np.mean(np.diff(second_half)) if len(second_half) > 1 else 0
            if 0 < s1 < 0.55 * s2 and len(delta_zone) / px_per_sec >= 0.035:
                has_delta = True

        # Sóng P và khoảng PR
        p_zone_start = max(0, r - int(px_per_sec * 0.38))
        p_zone_end = max(0, r - int(px_per_sec * 0.06))
        p_zone = sig[p_zone_start:p_zone_end]
        if len(p_zone) > 5:
            p_pks, _ = find_peaks(p_zone, prominence=0.35)
            if len(p_pks) > 0:
                p_idx = p_zone_start + p_pks[-1]
                pr_dur = (r - p_idx) / px_per_sec
                if 0.06 <= pr_dur <= 0.45:
                    pr_intervals.append(pr_dur)
                    p_amp_val = (sig[p_idx] / px_per_mv) * 10.0
                    p_amps.append(p_amp_val)
                    p_detected = True

                p_pts = np.where(np.abs(p_zone) > 0.15 * (px_per_mv / 10.0))[0]
                if len(p_pts) > 0:
                    p_durs_list.append(len(p_pts) / px_per_sec)

                if len(p_pks) >= 2:
                    p_notched = True
                    p_peaks_dist_val = (p_pks[-1] - p_pks[-2]) / px_per_sec

                p_pos_part = np.max(p_zone)
                p_neg_part = np.min(p_zone)
                p_pos_amps.append(max(0.0, (p_pos_part / px_per_mv) * 10.0))
                if p_neg_part < 0:
                    p_neg_amps.append((abs(p_neg_part) / px_per_mv) * 10.0)
                    neg_pts = np.where(p_zone < -0.10 * (px_per_mv / 10.0))[0]
                    p_neg_durs.append(len(neg_pts) / px_per_sec)
                else:
                    p_neg_amps.append(0.0)
                    p_neg_durs.append(0.0)

        # Sóng P retro & Khoảng RP
        retro_p_zone = sig[right_idx:min(len(sig), right_idx + int(px_per_sec * 0.22))]
        if len(retro_p_zone) > 4:
            retro_pks, _ = find_peaks(np.abs(retro_p_zone), prominence=0.45)
            if len(retro_pks) > 0:
                rp_dur = (right_idx + retro_pks[0] - r) / px_per_sec
                rp_intervals.append(rp_dur)
                if retro_p_zone[retro_pks[0]] < 0:
                    p_inverted = True
                if retro_p_zone[retro_pks[0]] > 0 and rp_dur < 0.09:
                    pseudo_r_prime = True

        # Tai thỏ rsR' ở V1
        sub_complex = sig[max(0, r - int(px_per_sec * 0.03)):min(len(sig), r + int(px_per_sec * 0.10))]
        if len(sub_complex) > 5:
            local_pks, _ = find_peaks(sub_complex, distance=int(px_per_sec * 0.020), prominence=2.0)
            if len(local_pks) >= 2:
                has_true_rsr = True
                has_notched_r = True
                if sub_complex[local_pks[0]] > sub_complex[local_pks[1]]:
                    left_rabbit_ear = True

    rr_list = np.diff(peaks) / px_per_sec if len(peaks) >= 2 else []
    avg_qrs = float(np.median(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.25))

    has_notch_sign = len(peaks) >= 4 and (float(np.std(rr_list)) / (np.median(rr_list) if np.median(rr_list) > 0 else 1.0) < 0.10)

    return {
        "r_amp": float(np.median(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.median(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.median(q_amps)) if q_amps else 0.0,
        "q_dur": float(np.max(q_durs)) if q_durs else 0.0,
        "st_shift": float(np.median(st_shifts)) if st_shifts else 0.0,
        "st_slope": max(set(st_slopes), key=st_slopes.count) if st_slopes else "flat",
        "t_amp": float(np.median(t_amps)) if t_amps else 0.0,
        "t_morph": max(set(t_morphs), key=t_morphs.count) if t_morphs else "normal",
        "qrs_w": avg_qrs,
        "pr_interval": float(np.median(pr_intervals)) if pr_intervals else 0.16,
        "pr_list": pr_intervals,
        "rp_interval": float(np.median(rp_intervals)) if rp_intervals else 0.0,
        "p_amp": float(np.median(p_amps)) if p_amps else 1.0,
        "p_pos_amp": float(np.median(p_pos_amps)) if p_pos_amps else 0.0,
        "p_neg_amp": float(np.median(p_neg_amps)) if p_neg_amps else 0.0,
        "p_neg_dur": float(np.median(p_neg_durs)) if p_neg_durs else 0.0,
        "p_dur": float(np.median(p_durs_list)) if p_durs_list else 0.08,
        "p_peaks_dist": p_peaks_dist_val,
        "p_notched": p_notched,
        "p_detected": p_detected,
        "p_inverted": p_inverted,
        "pseudo_r_prime": pseudo_r_prime,
        "vat": float(np.median(vats)) if vats else 0.03,
        "has_delta": has_delta,
        "true_rsr": has_true_rsr,
        "slurred_s": has_slurred_s,
        "notched_r": has_notched_r,
        "r_to_s_time": float(np.median(r_to_s_times)) if r_to_s_times else 0.06,
        "has_josephson": has_josephson,
        "left_rabbit_ear": left_rabbit_ear,
        "is_flatline": False,
        "has_notch_sign": has_notch_sign,
        "r_peaks": peaks,
        "rr_intervals": rr_list,
        "qrs_list": qrs_widths
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
                all_pr_intervals.extend(m["pr_list"])
            all_qrs_measurements.extend(m["qrs_list"])

    valid_rrs = [rr for rr in all_rr_intervals if 0.15 <= rr <= 2.2]
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
        "all_prs": all_pr_intervals,
        "p_ratio": p_presence_ratio
    }

# =========================================================================
# 3. CHẨN ĐOÁN NHIỄU & SAI LỆCH ĐIỆN CỰC (YDS 2026 CHƯƠNG 2)
# =========================================================================
def evaluate_artifacts_and_lead_reversals(leads):
    findings = []
    alerts = []

    d1 = leads.get("I", {})
    d2 = leads.get("II", {})
    d3 = leads.get("III", {})
    avr = leads.get("aVR", {})
    v2 = leads.get("V2", {})

    limb_signals_clean = [leads.get(ld, {}).get("p_detected", False) for ld in ["I", "II", "III"]]
    has_notch = any(l.get("has_notch_sign", False) for l in leads.values())
    if any(limb_signals_clean) and has_notch:
        findings.append(("Kiểm tra kỹ thuật", "Dấu hiệu Notch / Sinus dương tính: Phát hiện phức bộ QRS hẹp diễu hành xuyên qua nhiễu dao động (Nhiễu do run cơ Parkinson hoặc chuyển động, không phải rung thất/nhanh thất)"))

    d1_all_neg = (d1.get("r_amp", 0.0) < d1.get("s_amp", 0.0)) and (d1.get("t_amp", 0.0) < 0) and (d1.get("p_amp", 0.0) < 0 or not d1.get("p_detected", True))
    avr_pos = (avr.get("r_amp", 0.0) > avr.get("s_amp", 0.0)) and (avr.get("t_amp", 0.0) > 0)
    if d1_all_neg and avr_pos:
        findings.append(("Sai lệch điện cực", "Nghi ngờ Đảo ngược điện cực tay phải - tay trái (LA/RA): DI đảo ngược hoàn toàn (P, QRS, T âm), aVR dương tính (Cần phân biệt với Đảo ngược phủ tạng)"))
        alerts.append("⚠️ CẢNH BÁO KỸ THUẬT: ĐẢO DÂY ĐIỆN CỰC TAY LA/RA - ĐỀ NGHỊ ĐO LẠI ECG TRƯỚC KHI ĐỌC KẾT QUẢ")

    if d1.get("p_amp", 0.0) > d2.get("p_amp", 0.0) and d3.get("t_amp", 0.0) < -0.5 and (d3.get("p_amp", 0.0) < 0):
        findings.append(("Sai lệch điện cực", "Gợi ý Đảo ngược điện cực tay trái - chân trái (LA/LL): Sóng P ở DI > P ở DII, P và T ở DIII đảo ngược"))

    if d2.get("is_flatline", False):
        findings.append(("Sai lệch điện cực", "Lỗi đảo ngược điện cực RA/RL: Chuyển đạo DII là một đường thẳng đẳng điện"))
        alerts.append("⚠️ LỖI KỸ THUẬT: CHUYỂN ĐẠO DII ĐẲNG ĐIỆN DO ĐẢO DÂY ĐẤT RA/RL")
    elif d3.get("is_flatline", False):
        findings.append(("Sai lệch điện cực", "Lỗi đảo ngược điện cực LA/RL: Chuyển đạo DIII là một đường thẳng đẳng điện"))
        alerts.append("⚠️ LỖI KỸ THUẬT: CHUYỂN ĐẠO DIII ĐẲNG ĐIỆN DO ĐẢO DÂY ĐẤT LA/RL")

    if v2.get("p_amp", 0.0) < 0 or v2.get("t_morph") in ["biphasic_pos_neg", "biphasic_neg_pos"] and (v2.get("r_amp", 0.0) < 3.0):
        findings.append(("Sai lệch điện cực", "Gợi ý Đặt điện cực V1-V2 quá cao (KLS 2 hoặc 3): Sóng P ở V2 âm hoặc hai pha (Dễ gây hình ảnh giả sóng Q hoại tử vách, giả RBBB hoặc giả Brugada)"))

    v_r_amps = [leads.get(f"V{i}", {}).get("r_amp", 0.0) for i in range(1, 7)]
    if len(v_r_amps) == 6:
        if v_r_amps[1] > 12.0 and v_r_amps[2] < 4.0 and v_r_amps[3] > 8.0:
            findings.append(("Sai lệch điện cực", "Nghi ngờ Đảo lộn thứ tự dây trước tim V2 và V3: Sóng R ở V2 cao vọt bất thường rồi V3 tụt giảm đột ngột"))

    return {"findings": findings, "alerts": alerts}

# =========================================================================
# 4. CHẨN ĐOÁN BLOCK DẪN TRUYỀN & NHỊP CHẬM (YDS 2026 CHƯƠNG 6, 7, 8)
# =========================================================================
def evaluate_conduction_and_bradycardia_yds(data):
    leads = data["leads"]
    hr = data["hr"]
    qrs = data["qrs"]
    pr = data["pr"]
    all_prs = data.get("all_prs", [])
    rr_list = data["rr_list"]
    rr_cv = data["rr_cv"]
    mean_rr = data["mean_rr"]

    findings = []
    alerts = []

    d1 = leads.get("I", {})
    d2 = leads.get("II", {})
    d3 = leads.get("III", {})
    avf = leads.get("aVF", {})

    net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
    net_d2 = d2.get("r_amp", 0.0) - d2.get("s_amp", 0.0)
    net_avf = avf.get("r_amp", 0.0) - avf.get("s_amp", 0.0)

    is_extreme_lad = False
    is_extreme_rad = False
    axis_txt = "Trục trung gian sinh lý"

    if net_d1 > 0 and net_avf >= 0:
        axis_txt = "Bình thường (0° đến +90°)"
    elif net_d1 > 0 and net_avf < 0:
        if net_d2 < 0:
            is_extreme_lad = True
            axis_txt = "Trục lệch quá trái (-30° đến -90°)"
        else:
            axis_txt = "Trục lệch trái sinh lý (0° đến -30°)"
    elif net_d1 <= 0 and net_avf > 0:
        net_d3 = d3.get("r_amp", 0.0) - d3.get("s_amp", 0.0)
        if net_d3 > 0 and abs(net_d1) > 2.0:
            is_extreme_rad = True
            axis_txt = "Trục lệch quá phải (≥ +120°)"
        else:
            axis_txt = "Trục lệch phải (+90° đến +120°)"
    else:
        axis_txt = "Trục vô định"

    long_rrs = [r for r in rr_list if r > 1.6 * mean_rr]
    av_diag = None

    if hr <= 40 and qrs >= 0.12 and rr_cv < 0.05:
        av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát thất (QRS rộng ≥ 0.12s, tần số ≤ 40 l/p)"
        alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT ĐỘ 3 DƯỚI NÚT - CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP")
    elif hr < 60 and qrs < 0.12 and rr_cv < 0.05 and hr <= 45:
        av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát bộ nối (QRS hẹp, tần số 40-60 l/p)"
        alerts.append("🚨 BLOCK NHĨ THẤT ĐỘ 3 TẠI NÚT: CHỈ ĐỊNH NHẬP VIỆN HỒI SỨC TIM MẠCH")
    elif len(long_rrs) > 0:
        if len(all_prs) >= 3 and (max(all_prs) - min(all_prs) >= 0.05):
            av_diag = "Block nhĩ thất độ II Mobitz type 1 (Chu kỳ Wenckebach): Khoảng PR tăng dần cho đến khi có 1 sóng P không dẫn, khoảng RR ngắn dần"
        elif any(abs(r - 2.0 * mean_rr) < 0.15 for r in long_rrs):
            av_diag = "Block nhĩ thất độ II Mobitz type 2: Khoảng PR cố định ở các nhịp được dẫn, có sóng P không dẫn đột ngột (RR không dẫn = 2 x RR bình thường)"
            alerts.append("⚠️ CẢNH BÁO: BLOCK NHĨ THẤT ĐỘ II MOBITZ 2 - NGUY CƠ TIẾN TRIỂN THÀNH BLOCK CAO ĐỘ")
        elif any(r >= 3.0 * mean_rr for r in long_rrs):
            av_diag = "Block nhĩ thất cao độ: Có ít nhất hai sóng P liên tiếp không dẫn truyền (tỉ lệ P/QRS ≥ 3:1), khoảng PR cố định"
            alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT CAO ĐỘ (≥ 3:1) - NGUY CƠ NGẤT")
    elif any(abs(r - 2.0 * mean_rr) < 0.12 for r in rr_list) and len(rr_list) >= 4 and rr_cv < 0.08 and hr < 55:
        av_diag = "Block nhĩ thất 2:1: Sóng P dẫn và P không dẫn xen kẽ nhau (P/QRS = 2:1), khoảng RR đều nhau"
    elif pr > 0.20:
        av_diag = f"Block nhĩ thất độ I: Khoảng PR cố định và kéo dài ({pr:.2f}s > 0.20s), theo sau mỗi sóng P là một phức bộ QRS"

    if av_diag:
        findings.append(("Block Nhĩ Thất", av_diag))

    # Block xoang nhĩ & Ngưng xoang
    if len(long_rrs) > 0 and not av_diag:
        is_multiple_pp = any(abs((r / mean_rr) - round(r / mean_rr)) < 0.10 for r in long_rrs)
        max_pause = max(long_rrs)
        if max_pause >= 3.0:
            findings.append(("Hội chứng suy nút xoang", f"Khoảng ngưng xoang kéo dài > 3 giây (Pause = {max_pause:.2f}s)"))
            alerts.append("🚨 CẢNH BÁO: NGƯNG XOANG > 3 GIÂY - CHỈ ĐỊNH TẠO NHỊP")
        elif is_multiple_pp:
            findings.append(("Block Xoang Nhĩ", f"Block xoang nhĩ độ II type 2: Khoảng PP cố định, có khoảng nghỉ đột ngột bằng bội số nguyên của PP (Khoảng nghỉ = {round(max_pause/mean_rr)} x PP)"))
        elif max_pause < 2.0 * mean_rr:
            findings.append(("Block Xoang Nhĩ", "Block xoang nhĩ độ II type 1: Khoảng PP và RR ngắn dần cho đến khi có khoảng nghỉ mất hẳn sóng P (< 2 x PP)"))
        else:
            findings.append(("Rối loạn chức năng nút xoang", f"Khoảng ngưng xoang (Sinus Pause = {max_pause:.2f}s): Khoảng ngưng không bằng bội số của PP cơ bản"))

    # Block dẫn truyền nội thất
    v1 = leads.get("V1", {})
    v5 = leads.get("V5", {})
    v6 = leads.get("V6", {})
    v1_has_m_pattern = v1.get("true_rsr", False) or (v1.get("r_amp", 0.0) > v1.get("s_amp", 0.0) and v1.get("r_amp", 0.0) > 4.0)
    lateral_has_broad_s = v6.get("slurred_s", False) or d1.get("slurred_s", False)

    rbbb_type = None
    if v1_has_m_pattern and lateral_has_broad_s:
        if qrs >= 0.12:
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 có dạng chữ M (rsR'/rSR'), DI và V6 có sóng S rộng ≥ 0.04s"
        else:
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS < 0.12s, V1 có dạng chữ M (rsR') và S rộng ở DI/V6"

    if rbbb_type:
        findings.append(("Block Dẫn Truyền Nội Thất", rbbb_type))

    v5_v6_notched_r = v5.get("notched_r", False) or v6.get("notched_r", False) or d1.get("notched_r", False)
    v1_qs_rs = (v1.get("s_amp", 0.0) > 8.0 and v1.get("r_amp", 0.0) < 2.5)
    no_q_lateral = (v5.get("q_amp", 0.0) == 0 and v6.get("q_amp", 0.0) == 0 and d1.get("q_amp", 0.0) == 0)

    lbbb_type = None
    if v5_v6_notched_r and v1_qs_rs and no_q_lateral and not v1_has_m_pattern:
        if qrs >= 0.12:
            lbbb_type = "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, V5-V6/DI/aVL sóng R đơn pha có móc (dạng chữ M) không có sóng Q, V1 dạng QS/rS"
            alerts.append("🚨 LBBB HOÀN TOÀN: CẦN ĐỐI CHIẾU TIÊU CHUẨN SGARBOSSA LOẠI TRỪ NMCT THÀNH TRƯỚC")
        else:
            lbbb_type = "Block nhánh trái không hoàn toàn (Incomplete LBBB): QRS < 0.12s"

    if lbbb_type:
        findings.append(("Block Dẫn Truyền Nội Thất", lbbb_type))

    if qrs > 0.11 and not rbbb_type and not lbbb_type:
        findings.append(("Block Dẫn Truyền Nội Thất", f"Chậm dẫn truyền nội thất không đặc hiệu (IVCD): QRS giãn rộng ({qrs:.3f}s > 0.11s)"))

    lafb = False
    if is_extreme_lad and qrs < 0.12:
        d1_qr = (d1.get("r_amp", 0.0) > 0 and d1.get("s_amp", 0.0) < d1.get("r_amp", 0.0))
        d3_rs = (d3.get("s_amp", 0.0) > d3.get("r_amp", 0.0))
        if d1_qr and d3_rs:
            lafb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch quá trái (-30° đến -90°); DI, aVL dạng qR; DII, DIII, aVF dạng rS (DI dương, aVF âm, DII âm)"))

    lpfb = False
    if is_extreme_rad and qrs < 0.12:
        d1_rs = (d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0))
        d3_qr = (d3.get("r_amp", 0.0) > d3.get("s_amp", 0.0))
        if d1_rs and d3_qr and v1.get("r_amp", 0.0) < 6.0:
            lpfb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch quá phải (≥ 120°); DI, aVL dạng rS; DII, DIII, aVF dạng qR"))

    return {"axis": axis_txt, "findings": findings, "alerts": alerts}

# =========================================================================
# 5. CHẨN ĐOÁN KÍCH THÍCH SỚM & NGOẠI TÂM THU (YDS 2026 CHƯƠNG 9 & 10)
# =========================================================================
def evaluate_preexcitation_and_ectopics_yds(data):
    leads = data["leads"]
    hr = data["hr"]
    qrs = data["qrs"]
    pr = data["pr"]
    all_prs = data.get("all_prs", [])
    rr_list = data["rr_list"]
    mean_rr = data["mean_rr"]

    findings = []
    alerts = []

    any_delta = any(l.get("has_delta", False) for l in leads.values())
    v1_r = leads.get("V1", {}).get("r_amp", 0.0)
    v1_s = leads.get("V1", {}).get("s_amp", 0.0)
    v2_r = leads.get("V2", {}).get("r_amp", 0.0)
    v2_s = leads.get("V2", {}).get("s_amp", 0.0)

    is_v1_v2_pos = (v1_r >= v1_s) and (v2_r >= v2_s)
    is_v1_v2_neg = (v1_s > v1_r) and (v2_s > v2_r)

    if pr < 0.12 and (qrs > 0.10 or any_delta):
        is_intermittent = any(p >= 0.13 for p in all_prs) and any(p < 0.12 for p in all_prs)
        if is_v1_v2_pos:
            wpw_type = "Hội chứng Wolff-Parkinson-White (WPW) Type A: PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta, QRS dương ở V1-V2 (Bó Kent bên trái)"
        elif is_v1_v2_neg:
            wpw_type = "Hội chứng Wolff-Parkinson-White (WPW) Type B: PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta, QRS âm ở V1-V2 (Bó Kent bên phải)"
        else:
            wpw_type = "Hội chứng Wolff-Parkinson-White (WPW): PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta"

        if is_intermittent:
            wpw_type += " [Dạng WPW từng lúc]"

        findings.append(("Hội chứng kích thích sớm", wpw_type))
        alerts.append("⚠️ HỘI CHỨNG WPW: TRÁNH DÙNG THUỐC CHẸN NÚT NHĨ THẤT (DIGOXIN, VERAPAMIL)")
    elif pr < 0.12 and qrs <= 0.10 and not any_delta:
        findings.append(("Hội chứng kích thích sớm", "Hội chứng Lown-Ganong-Levine (LGL): Khoảng PR ngắn (< 0.12s), QRS bình thường, không có sóng Delta (Bó James)"))

    early_indices = [i for i, r in enumerate(rr_list[:-1]) if r < 0.80 * mean_rr]
    if len(early_indices) > 0:
        pvc_count = 0
        pac_count = 0
        pjc_count = 0
        pvc_origins = []
        is_full_compensatory = False
        consecutive_pvcs = 0
        max_consecutive_pvcs = 0

        for idx in early_indices:
            r_early = rr_list[idx]
            r_next = rr_list[idx + 1] if idx + 1 < len(rr_list) else mean_rr
            cycle_pair = r_early + r_next
            is_full_compensatory = abs(cycle_pair - 2.0 * mean_rr) < 0.14 * mean_rr

            if qrs >= 0.12 or leads.get("V1", {}).get("qrs_w", 0.08) >= 0.12:
                pvc_count += 1
                consecutive_pvcs += 1
                max_consecutive_pvcs = max(max_consecutive_pvcs, consecutive_pvcs)
                pvc_origins.append("Thất trái" if v1_r >= v1_s else "Thất phải")
            else:
                consecutive_pvcs = 0
                if is_full_compensatory:
                    pjc_count += 1
                else:
                    pac_count += 1

        if pvc_count > 0:
            origin_str = f"xuất phát từ {max(set(pvc_origins), key=pvc_origins.count)}" if pvc_origins else ""
            if max_consecutive_pvcs >= 3:
                vt_rate = int(60.0 / (mean_rr * 0.65)) if mean_rr > 0 else 130
                findings.append(("Ngoại tâm thu thất", f"Cơn nhanh thất ngắn: Có {max_consecutive_pvcs} ngoại tâm thu thất liên tiếp, tần số {vt_rate} l/p"))
            elif max_consecutive_pvcs == 2:
                findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất cặp đôi (Couplet), {origin_str}"))
            elif len(rr_list) >= 4 and pvc_count >= len(rr_list) // 2:
                findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất nhịp đôi (Bigeminy), {origin_str}"))
            elif len(rr_list) >= 4 and pvc_count >= len(rr_list) // 3:
                findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất nhịp ba (Trigeminy), {origin_str}"))
            else:
                findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất (PVC), {origin_str}, nghỉ bù hoàn toàn"))

        if pac_count > 0:
            findings.append(("Ngoại tâm thu nhĩ", "Ngoại tâm thu nhĩ (PAC): Nhịp đến sớm với sóng P' biến dạng, QRS hẹp, nghỉ bù không hoàn toàn"))
        if pjc_count > 0:
            findings.append(("Ngoại tâm thu bộ nối", "Ngoại tâm thu bộ nối (PJC): Nhịp đến sớm, QRS hẹp, P' đảo hoặc lẫn QRS, nghỉ bù hoàn toàn"))

    return {"findings": findings, "alerts": alerts}

# =========================================================================
# 6. CHẨN ĐOÁN NHỊP NHANH SVT & VT (YDS 2026 CHƯƠNG 11, 12, 13)
# =========================================================================
def evaluate_tachycardias_yds(data):
    leads = data["leads"]
    hr = data["hr"]
    qrs = data["qrs"]
    pr = data["pr"]
    rr_cv = data["rr_cv"]
    p_ratio = data["p_ratio"]

    findings = []
    alerts = []

    if hr >= 220 and rr_cv > 0.30 and qrs >= 0.16:
        findings.append(("Rối loạn nhịp thất ác tính", "Rung thất (Ventricular Fibrillation - VF): Hoạt động điện thất hỗn loạn, vô tổ chức, mất hoàn toàn phức bộ QRS"))
        alerts.append("🚨 BÁO ĐỘNG ĐỎ: RUNG THẤT - BỆNH NHÂN NGƯNG TIM, KÍCH HOẠT CPR VÀ PHÁ RUNG NGAY")
        return {"findings": findings, "alerts": alerts}

    if 180 <= hr <= 300 and qrs >= 0.16 and rr_cv < 0.08:
        findings.append(("Rối loạn nhịp thất ác tính", "Cuồng thất (Ventricular Flutter): Các sóng hình sin đều đặn liên tục, tần số 150-300 lần/phút"))
        alerts.append("🚨 CẤP CỨU: CUỒNG THẤT - NGUY CƠ TIẾN TRIỂN THÀNH RUNG THẤT")
        return {"findings": findings, "alerts": alerts}

    v1_to_v6_amps = [leads.get(f"V{i}", {}).get("r_amp", 0.0) for i in range(1, 7)]
    if hr >= 150 and max(v1_to_v6_amps) - min(v1_to_v6_amps) > 12.0 and rr_cv > 0.20:
        findings.append(("Rối loạn nhịp thất ác tính", "Xoắn đỉnh (Torsades de Pointes): Nhịp nhanh thất đa dạng xoắn quanh đường đẳng điện"))
        alerts.append("🚨 CẤP CỨU: XOẮN ĐỈNH - TRUYỀN MAGNESIUM SULFATE")
        return {"findings": findings, "alerts": alerts}

    if hr > 100 and qrs >= 0.12:
        vt_criteria_met = []
        d1 = leads.get("I", {})
        avf = leads.get("aVF", {})
        v1 = leads.get("V1", {})

        net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
        net_avf = avf.get("r_amp", 0.0) - avf.get("s_amp", 0.0)
        if net_d1 < 0 and net_avf < 0:
            vt_criteria_met.append("Trục QRS vô định (DI âm, aVF âm)")
        if qrs > 0.16:
            vt_criteria_met.append(f"QRS rất rộng ({qrs*1000:.0f} ms > 160 ms)")

        chest_signs = [(leads.get(f"V{i}", {}).get("r_amp", 0.0) - leads.get(f"V{i}", {}).get("s_amp", 0.0)) for i in range(1, 7)]
        if all(s > 0 for s in chest_signs):
            vt_criteria_met.append("QRS đồng hướng dương ở V1-V6")
        elif all(s < 0 for s in chest_signs):
            vt_criteria_met.append("QRS đồng hướng âm ở V1-V6")

        if v1.get("left_rabbit_ear", False):
            vt_criteria_met.append("Dấu hiệu tai thỏ với tai trái lớn hơn ở V1 (R > R')")

        max_rs_time = max(l.get("r_to_s_time", 0.0) for l in leads.values())
        if max_rs_time > 0.10:
            vt_criteria_met.append(f"Dấu hiệu Brugada dương tính (R đến đáy S = {max_rs_time*1000:.0f} ms > 100 ms)")

        if v1.get("has_josephson", False) or leads.get("V2", {}).get("has_josephson", False):
            vt_criteria_met.append("Dấu hiệu Josephson dương tính (Khấc sườn xuống đáy sóng S tại V1/V2)")

        if len(vt_criteria_met) >= 2 or (qrs > 0.16 and len(vt_criteria_met) >= 1):
            findings.append(("Nhịp nhanh thất (VT)", f"Nhịp nhanh thất (Ventricular Tachycardia): {'; '.join(vt_criteria_met)}"))
            alerts.append("🚨 CẤP CỨU: NHỊP NHANH THẤT (VT) - ĐÁNH GIÁ NGAY HUYẾT ĐỘNG ĐỂ SỐC ĐIỆN")
            return {"findings": findings, "alerts": alerts}

    if hr > 100 and qrs >= 0.12:
        v1 = leads.get("V1", {})
        if v1.get("true_rsr", False) and not v1.get("left_rabbit_ear", False):
            findings.append(("Nhịp nhanh trên thất", "Nhịp nhanh trên thất dẫn truyền lệch hướng: QRS giãn rộng có dạng RBBB với tai thỏ bên phải cao hơn ở V1"))
            return {"findings": findings, "alerts": alerts}
        elif any(l.get("has_delta", False) for l in leads.values()):
            findings.append(("Nhịp nhanh trên thất", "Nhịp nhanh vào lại nhĩ thất dẫn truyền ngược dòng (Antidromic AVRT)"))
            return {"findings": findings, "alerts": alerts}

    if hr > 100 and qrs < 0.12:
        if rr_cv > 0.20 and p_ratio < 0.20:
            findings.append(("Rối loạn nhịp nhĩ", f"Rung nhĩ đáp ứng thất nhanh (AFib with RVR) - Tần số: {hr} l/p"))
            return {"findings": findings, "alerts": alerts}
        if 135 <= hr <= 165 and any(abs(l.get("st_shift", 0.0)) > 0 for l in leads.values()):
            findings.append(("Rối loạn nhịp nhĩ", f"Cuồng nhĩ (Atrial Flutter): Nghi ngờ cuồng nhĩ dẫn truyền 2:1 (Tần số thất {hr} l/p)"))
            return {"findings": findings, "alerts": alerts}
        if rr_cv > 0.15 and p_ratio >= 0.50:
            findings.append(("Rối loạn nhịp nhĩ", f"Nhịp nhanh nhĩ đa ổ (MAT) - Tần số: {hr} l/p"))
            return {"findings": findings, "alerts": alerts}

        rp_val = max(l.get("rp_interval", 0.0) for l in leads.values())
        pseudo_r = leads.get("V1", {}).get("pseudo_r_prime", False)
        p_inv = any(leads.get(ld, {}).get("p_inverted", False) for ld in ["II", "III", "aVF"])

        if pseudo_r or (0 < rp_val <= 0.09):
            findings.append(("Nhịp nhanh vào lại nút nhĩ thất (AVNRT)", f"AVNRT thể điển hình (Slow-Fast): Nhịp nhanh đều ({hr} l/p), QRS hẹp, RP ≤ 90 ms"))
        elif rp_val > 0.09 and rp_val < pr:
            findings.append(("Nhịp nhanh vào lại nhĩ thất (AVRT)", f"Orthodromic AVRT: Nhịp nhanh đều ({hr} l/p), QRS hẹp, RP dài > 90 ms"))
        elif p_inv and pr < 0.12:
            findings.append(("Nhịp nhanh bộ nối (JET)", f"Nhịp nhanh bộ nối (JET) - Tần số: {hr} l/p"))
        else:
            findings.append(("Nhịp nhanh trên thất", f"Nhịp nhanh kịch phát trên thất (SVT) - Tần số: {hr} l/p"))

    return {"findings": findings, "alerts": alerts}

# =========================================================================
# 7. CHẨN ĐOÁN HỘI CHỨNG VÀNH CẤP & MẠN (YDS 2026 CHƯƠNG 4 & 5)
# =========================================================================
def evaluate_yds_coronary_syndromes(leads, gender="Nam", age=55):
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

        if l_name in ["V2", "V3"]:
            cutoff = 1.5 if gender == "Nữ" else (2.0 if age >= 40 else 2.5)
        else:
            cutoff = 1.0

        if st_val >= cutoff:
            st_elevation_leads.append(l_name)

        if st_val <= -0.5:
            st_depression_leads.append((l_name, slope))

        is_chest = l_name.startswith("V")
        if (is_chest and t_val > 10.0) or (not is_chest and t_val > 5.0) or (r_val > 0 and t_val > 0.75 * r_val):
            hyperacute_t_leads.append(l_name)

        if t_val < -1.0 and (r_val > s_val or r_val > 5.0):
            inverted_t_leads.append(l_name)

        if -1.0 <= t_val <= 1.0 and r_val > 3.0:
            flat_t_leads.append(l_name)

        if t_morph == "biphasic_pos_neg":
            biphasic_t_leads.append(l_name)

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

    # De Winter
    avr_st = leads.get("aVR", {}).get("st_shift", 0.0)
    dewinter_candidates = [l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping" and l in hyperacute_t_leads]
    if len(dewinter_candidates) >= 2 or (avr_st >= 0.5 and len([l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping"]) >= 2):
        findings.append(("Hội chứng mạch vành cấp", "Hội chứng De Winter: Điểm J chênh xuống đi lên ở V1-V6 kèm sóng T cao đối xứng (Tương đương STEMI tắc đoạn gần LAD)"))
        alerts.append("🚨 CẤP CỨU: HỘI CHỨNG DE WINTER - CAN THIỆP MẠCH VÀNH KHẨN CẤP")

    # Wellens
    wellens_a = [l for l in biphasic_t_leads if l in ["V1", "V2", "V3"]]
    wellens_b = [l for l in inverted_t_leads if l in ["V1", "V2", "V3", "V4"]]
    if len(wellens_a) >= 2:
        findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type A: Sóng T hai pha (+/-) tại {', '.join(wellens_a)} (Gợi ý hẹp nặng đoạn gần LAD)"))
        alerts.append("⚠️ HỘI CHỨNG WELLENS TYPE A: NGUY CƠ TIẾN TRIỂN THÀNH NMCT DIỆN RỘNG")
    elif len(wellens_b) >= 2:
        findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type B: Sóng T âm sâu đối xứng tại {', '.join(wellens_b)} (Gợi ý hẹp nặng đoạn gần LAD)"))
        alerts.append("⚠️ HỘI CHỨNG WELLENS TYPE B: CHỈ ĐỊNH CHỤP MẠCH VÀNH SỚM")

    # LMCA / 3 nhánh
    if len(dep_leads_names) >= 6 and avr_st >= 1.0:
        findings.append(("Hội chứng mạch vành cấp", f"Gợi ý tổn thương Thân chung ĐM Vành Trái (LMCA) hoặc 3 nhánh: ST chênh xuống lan tỏa ({', '.join(dep_leads_names)}) kèm ST chênh lên tại aVR"))
        alerts.append("🚨 NGUY KỊCH: THEO DÕI HẸP NẶNG THÂN CHUNG LMCA / 3 NHÁNH")

    # STEMI
    stemi_regions = []
    culprit_artery = []
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

    inferior_leads = {"II", "III", "aVF"}.intersection(st_set)
    if len(inferior_leads) >= 2:
        recip = " (Soi gương ở aVL, DI, V1-V3)" if len({"aVL", "I", "V1", "V2"}.intersection(dep_set)) >= 1 else ""
        stemi_regions.append(f"Thành dưới (Inferior: {', '.join(sorted(list(inferior_leads)))}){recip}")
        culprit_artery.append("RCA (80%) hoặc LCx (20%)")

    v2_r = leads.get("V2", {}).get("r_amp", 0.0)
    v2_s = leads.get("V2", {}).get("s_amp", 0.0)
    v2_st = leads.get("V2", {}).get("st_shift", 0.0)
    v2_t = leads.get("V2", {}).get("t_amp", 0.0)
    if (v2_s > 0 and v2_r / v2_s > 1.0) and v2_st <= -0.5 and v2_t > 0:
        stemi_regions.append("Dấu hiệu gián tiếp NMCT Thành sau thực (R/S > 1, ST chênh xuống, T dương ở V2-V3 - Đề nghị đo V7-V9)")
        culprit_artery.append("RCA hoặc LCx")

    if stemi_regions:
        reg_txt = " + ".join(stemi_regions)
        art_txt = f" - ĐM thủ phạm dự đoán: {', '.join(set(culprit_artery))}" if culprit_artery else ""
        has_q = any(l in set(pathological_q_leads) for l in st_set)
        stage_str = "Bán cấp / Hoại tử (Đã có sóng Q)" if has_q else "Tối cấp / Cấp tính"
        findings.append(("Hội chứng mạch vành cấp (STEMI)", f"Nhồi máu cơ tim ST chênh lên - Vùng: {reg_txt} - Giai đoạn: {stage_str}{art_txt}"))
        alerts.append(f"🚨 CẤP CỨU: STEMI VÙNG {reg_txt.upper()} - KÍCH HOẠT PCI KHẨN CẤP")
    elif len(st_depression_leads) >= 2 or len(inverted_t_leads) >= 2 or len(hyperacute_t_leads) >= 2:
        ischemia_details = []
        spec_dep = [f"{l} (dạng {sl})" for (l, sl) in st_depression_leads if sl in ["horizontal", "downsloping"]]
        if len(spec_dep) >= 2:
            ischemia_details.append(f"ST chênh xuống đặc hiệu tại: {', '.join(spec_dep)}")
        elif len(dep_leads_names) >= 2:
            ischemia_details.append(f"ST chênh xuống tại: {', '.join(dep_leads_names)}")

        if len(inverted_t_leads) >= 2:
            ischemia_details.append(f"Sóng T âm sâu đảo ngược tại: {', '.join(inverted_t_leads)}")
        if len(hyperacute_t_leads) >= 2:
            ischemia_details.append(f"Sóng T tối cấp tại: {', '.join(hyperacute_t_leads)}")

        findings.append(("Thiếu máu cục bộ cơ tim (NSTE-ACS)", f"Biến đổi thiếu máu cơ tim cấp: {'; '.join(ischemia_details)}"))
        alerts.append("⚠️ CẢNH BÁO: THEO DÕI NSTE-ACS - ĐỊNH LƯỢNG TROPONIN HS")
    elif len(pathological_q_leads) >= 2:
        q_set = set(pathological_q_leads)
        old_mi_regions = []
        if len({"II", "III", "aVF"}.intersection(q_set)) >= 2:
            old_mi_regions.append("Thành dưới")
        if {"V1", "V2"}.issubset(q_set) or {"V2", "V3"}.issubset(q_set):
            old_mi_regions.append("Trước - Vách")
        if {"V4", "V5", "V6"}.intersection(q_set) and len({"V4", "V5", "V6"}.intersection(q_set)) >= 2:
            old_mi_regions.append("Thành trước - bên")
        if old_mi_regions:
            findings.append(("Hội chứng mạch vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI theo chuẩn ESC 2018) - Vùng: {', '.join(old_mi_regions)}"))

    return {"findings": findings, "alerts": alerts}

# =========================================================================
# 8. HÀM ĐIỀU PHỐI CHẨN ĐOÁN TOÀN DIỆN (CHÍNH XÁC THỨ TỰ GỌI)
# =========================================================================
def diagnose_ecg_comprehensive(data, gender="Nam", age=55):
    leads = data.get("leads", {})
    hr = data.get("hr", 72)

    findings = []
    alerts = []

    # 1. Nhiễu & Sai lệch điện cực
    tech_res = evaluate_artifacts_and_lead_reversals(leads)
    findings.extend(tech_res["findings"])
    alerts.extend(tech_res["alerts"])

    # 2. Block dẫn truyền & Nhịp chậm
    cond_res = evaluate_conduction_and_bradycardia_yds(data)
    axis_txt = cond_res["axis"]
    findings.extend(cond_res["findings"])
    alerts.extend(cond_res["alerts"])

    # 3. Kích thích sớm & Ngoại tâm thu
    pre_res = evaluate_preexcitation_and_ectopics_yds(data)
    findings.extend(pre_res["findings"])
    alerts.extend(pre_res["alerts"])

    # 4. Nhịp Nhanh SVT & VT
    tachy_res = evaluate_tachycardias_yds(data)
    findings.extend(tachy_res["findings"])
    alerts.extend(tachy_res["alerts"])

    # 5. Hội chứng Vành Cấp & Mạn
    coronary_res = evaluate_yds_coronary_syndromes(leads, gender=gender, age=age)
    findings.extend(coronary_res["findings"])
    alerts.extend(coronary_res["alerts"])

    # 6. Lớn nhĩ, Dày thất & Dày hai thất
    hyp_res = evaluate_atrial_ventricular_hypertrophy_yds(leads, gender=gender)
    findings.extend(hyp_res["findings"])
    alerts.extend(hyp_res["alerts"])

    if not any(k in f[0] for f in findings for k in ["Rối loạn nhịp", "Nhịp nhanh", "Block nhĩ thất", "Hội chứng kích thích sớm"]):
        if hr > 100:
            findings.append(("Nhịp học", f"Nhịp nhanh xoang: Tần số {hr} lần/phút"))
        elif hr < 60:
            findings.append(("Nhịp học", f"Nhịp chậm xoang: Tần số {hr} lần/phút"))
        else:
            findings.append(("Nhịp học", f"Nhịp xoang bình thường: Tần số {hr} lần/phút"))

    return {
        "axis": axis_txt,
        "sokolow": hyp_res["sokolow_lv"],
        "sokolow_rv": hyp_res["sokolow_rv"],
        "cornell": hyp_res["cornell"],
        "re_score": hyp_res["re_score"],
        "findings": findings,
        "alerts": alerts
    }

# =========================================================================
# 9. GIAO DIỆN HIỂN THỊ STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Bản Ghi ECG 12 Chuyển Đạo")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        gender_choice = st.radio("Giới tính bệnh nhân", ["Nam", "Nữ"], horizontal=True)
    with c_p2:
        age_choice = st.number_input("Tuổi", min_value=1, max_value=120, value=35)

    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG nạp vào hệ thống chẩn đoán YDS 2026", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên để phân tích.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Chuyên Khoa Toàn Diện")
    if uploaded:
        with st.spinner("Đang kiểm tra CAPS SCAN, đo đạc sóng P-QRS-ST-T, bóc tách lớn nhĩ, dày thất và loạn nhịp..."):
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
        m2.metric("Khoảng PR", f"{res['pr']:.2f} s", delta="Ngắn (<0.12s)" if res['pr'] < 0.12 else ("Kéo dài (>0.20s)" if res['pr'] > 0.20 else "Bình thường"), delta_color="inverse" if (res['pr'] < 0.12 or res['pr'] > 0.20) else "normal")
        m3.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m4.metric("Sokolow-Lyon (Trái)", f"{diag['sokolow']:.1f} mm")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}` | **Cornell:** `{diag['cornell']:.1f} mm` | **Điểm Romhilt-Estes:** `{diag['re_score']}/13`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán Phân Tầng")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if any(k in desc for k in ["STEMI", "độ III", "Rung thất", "Cuồng thất", "Xoắn đỉnh", "Nhịp nhanh thất", "cao độ", "LMCA", "ngưng xoang > 3 giây"]):
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif any(k in desc for k in ["WPW", "Wellens", "Ngoại tâm thu", "Block", "NSTE-ACS", "LGL", "AVNRT", "AVRT", "JET", "Dày", "Lớn", "Sai lệch"]):
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện nhiễu, sai lệch điện cực, phì đại buồng tim, loạn nhịp hoặc bệnh mạch vành.")

        st.markdown("---")
        with st.expander("🔍 Chi Tiết Đo Đạc Hình Thái 12 Chuyển Đạo (P, Q, R, S, ST, T, Delta, VAT)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                st_val = l_data.get('st_shift', 0.0)
                slope = l_data.get('st_slope', 'flat')
                t_val = l_data.get('t_amp', 0.0)

                eval_txt = "Đẳng điện"
                if st_val >= 1.0:
                    eval_txt = "🔴 ST Chênh Lên"
                elif st_val <= -0.5:
                    eval_txt = f"🔵 ST Chênh Xuống ({slope})"

                detail_list.append({
                    "Chuyển đạo": l_name,
                    "ST Chênh (mm)": f"{st_val:+.1f}",
                    "Đánh giá ST": eval_txt,
                    "Sóng T (mm)": f"{t_val:+.1f}",
                    "QRS (s)": f"{l_data.get('qrs_w', 0.08):.3f}",
                    "VAT (s)": f"{l_data.get('vat', 0.03):.3f}",
                    "Sóng R (mm)": f"{l_data.get('r_amp', 0.0):.1f}",
                    "Sóng S (mm)": f"{l_data.get('s_amp', 0.0):.1f}",
                    "PR (s)": f"{l_data.get('pr_interval', 0.16):.2f}"
                })
            st.dataframe(detail_list, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
