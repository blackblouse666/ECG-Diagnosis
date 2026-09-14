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
st.caption("Cập nhật chế độ nhập thủ công Block AV, Block nhánh, Block phân nhánh, Block xoang nhĩ, Trục điện tim & Bệnh mạch vành YDS 2026")

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
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

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
    roi_clean = roi.copy()
    label_h = int(h * 0.25)
    label_w = int(w * 0.25)
    roi_clean[:label_h, :label_w] = 0

    signal = []
    for col in range(w):
        pts = np.where(roi_clean[:, col] > 0)[0]
        if len(pts) > 0:
            signal.append(h - np.median(pts))
        else:
            signal.append(signal[-1] if len(signal) > 0 else h / 2.0)
    return np.array(signal)

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

        # 1. Sóng Q
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

        # 2. Sóng S & Điểm J
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

        # 3. ST chênh
        st_shift_mm = (sig[j_idx] / px_per_mv) * 10.0
        st_shifts.append(st_shift_mm)
        j80_idx = min(len(sig) - 1, j_idx + int(px_per_sec * 0.07))
        st_j80_shift = (sig[j80_idx] / px_per_mv) * 10.0
        slope_diff = st_j80_shift - st_shift_mm
        st_slopes.append("upsloping" if slope_diff > 0.4 else ("downsloping" if slope_diff < -0.4 else "horizontal"))

        # 4. Sóng T
        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.10)):min(len(sig), r + int(px_per_sec * 0.35))]
        if len(t_zone) > 6:
            t_max, t_min = np.max(t_zone), np.min(t_zone)
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

        # 5. Độ rộng QRS và VAT
        left_idx = r
        while left_idx > max(0, r - int(px_per_sec * 0.12)) and sig[left_idx] > 0.15 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.16)) and sig[right_idx] > 0.15 * sig[r]:
            right_idx += 1
        measured_qrs = (right_idx - left_idx) / px_per_sec
        qrs_widths.append(measured_qrs)
        vats.append((r - left_idx) / px_per_sec)

        # 6. Sóng Delta
        delta_zone = sig[left_idx:r]
        if len(delta_zone) > 5:
            first_half = delta_zone[:len(delta_zone)//2]
            second_half = delta_zone[len(delta_zone)//2:]
            s1 = np.mean(np.diff(first_half)) if len(first_half) > 1 else 0
            s2 = np.mean(np.diff(second_half)) if len(second_half) > 1 else 0
            if 0 < s1 < 0.55 * s2 and len(delta_zone) / px_per_sec >= 0.035:
                has_delta = True

        # 7. Sóng P và khoảng PR
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

                p_pos_part, p_neg_part = np.max(p_zone), np.min(p_zone)
                p_pos_amps.append(max(0.0, (p_pos_part / px_per_mv) * 10.0))
                if p_neg_part < 0:
                    p_neg_amps.append((abs(p_neg_part) / px_per_mv) * 10.0)
                    neg_pts = np.where(p_zone < -0.10 * (px_per_mv / 10.0))[0]
                    p_neg_durs.append(len(neg_pts) / px_per_sec)
                else:
                    p_neg_amps.append(0.0)
                    p_neg_durs.append(0.0)

        # 8. Sóng P retro & Khoảng RP
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

        # 9. Tai thỏ rsR' ở V1
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
    h_ecg = int(h_tot * 0.82)
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

    # THUẬT TOÁN ƯU TIÊN CHUYỂN ĐẠO DII TÍNH HR
    d2_rrs = [rr for rr in leads.get("II", {}).get("rr_intervals", []) if 0.20 <= rr <= 2.2]
    valid_rrs = [rr for rr in all_rr_intervals if 0.20 <= rr <= 2.2]

    if len(d2_rrs) >= 2:
        mean_rr = float(np.median(d2_rrs))
        rr_cv = float(np.std(d2_rrs)) / mean_rr if mean_rr > 0 else 0.0
    elif len(valid_rrs) > 0:
        mean_rr = float(np.median(valid_rrs))
        rr_cv = float(np.std(valid_rrs)) / mean_rr if mean_rr > 0 else 0.0
    else:
        mean_rr = 0.85
        rr_cv = 0.0

    hr = int(round(60.0 / mean_rr)) if mean_rr > 0 else 72

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
        "rr_list": d2_rrs if len(d2_rrs) >= 2 else valid_rrs,
        "qrs": qrs_final,
        "pr": pr_final,
        "all_prs": all_pr_intervals,
        "p_ratio": p_presence_ratio
    }

# =========================================================================
# 2. LỚP ĐỐI TƯỢNG PHÂN TÍCH LÂM SÀNG TOÀN DIỆN (YDS 2026)
# =========================================================================
class ECGClinicalAnalyzer:
    def __init__(self, data, gender="Nam", age=55, manual_override=None):
        self.data = data
        self.leads = data["leads"]
        self.hr = data["hr"]
        self.qrs = data["qrs"]
        self.pr = data["pr"]
        self.all_prs = data.get("all_prs", [])
        self.rr_list = data["rr_list"]
        self.mean_rr = data["mean_rr"]
        self.rr_cv = data["rr_cv"]
        self.p_ratio = data["p_ratio"]
        self.gender = gender
        self.age = age

        self.findings = []
        self.alerts = []
        self.axis_txt = "Trục trung gian sinh lý"
        self.calculated_axis_deg = None
        self.sokolow_lv = 0.0
        self.sokolow_rv = 0.0
        self.cornell = 0.0
        self.re_score = 0
        self.manual_override = manual_override or {}

    def evaluate_artifacts(self):
        d1, d2, d3 = self.leads.get("I", {}), self.leads.get("II", {}), self.leads.get("III", {})
        avr, v2 = self.leads.get("aVR", {}), self.leads.get("V2", {})

        limb_signals_clean = [self.leads.get(ld, {}).get("p_detected", False) for ld in ["I", "II", "III"]]
        has_notch = any(l.get("has_notch_sign", False) for l in self.leads.values())
        if any(limb_signals_clean) and has_notch:
            self.findings.append(("Kiểm tra kỹ thuật", "Dấu hiệu Notch / Sinus dương tính: Phát hiện phức bộ QRS hẹp diễu hành xuyên qua nhiễu dao động (Nhiễu do run cơ Parkinson hoặc chuyển động, không phải rung thất/nhanh thất)"))[cite: 5]

        d1_all_neg = (d1.get("r_amp", 0.0) < d1.get("s_amp", 0.0)) and (d1.get("t_amp", 0.0) < 0) and (d1.get("p_amp", 0.0) < 0 or not d1.get("p_detected", True))
        avr_pos = (avr.get("r_amp", 0.0) > avr.get("s_amp", 0.0)) and (avr.get("t_amp", 0.0) > 0)
        if d1_all_neg and avr_pos:
            self.findings.append(("Sai lệch điện cực", "Nghi ngờ Đảo ngược điện cực tay phải - tay trái (LA/RA): DI đảo ngược hoàn toàn (P, QRS, T âm), aVR dương tính"))[cite: 5]
            self.alerts.append("⚠️ CẢNH BÁO KỸ THUẬT: ĐẢO DÂY ĐIỆN CỰC TAY LA/RA - ĐỀ NGHỊ ĐO LẠI ECG TRƯỚC KHI ĐỌC KẾT QUẢ")

        if d1.get("p_amp", 0.0) > d2.get("p_amp", 0.0) and d3.get("t_amp", 0.0) < -0.5 and (d3.get("p_amp", 0.0) < 0):
            self.findings.append(("Sai lệch điện cực", "Gợi ý Đảo ngược điện cực tay trái - chân trái (LA/LL): Sóng P ở DI > P ở DII, P và T ở DIII đảo ngược"))[cite: 5]

        if d2.get("is_flatline", False):
            self.findings.append(("Sai lệch điện cực", "Lỗi đảo ngược điện cực RA/RL: Chuyển đạo DII là một đường thẳng đẳng điện"))[cite: 5]
            self.alerts.append("⚠️ LỖI KỸ THUẬT: CHUYỂN ĐẠO DII ĐẲNG ĐIỆN DO ĐẢO DÂY ĐẤT RA/RL")
        elif d3.get("is_flatline", False):
            self.findings.append(("Sai lệch điện cực", "Lỗi đảo ngược điện cực LA/RL: Chuyển đạo DIII là một đường thẳng đẳng điện"))[cite: 5]
            self.alerts.append("⚠️ LỖI KỸ THUẬT: CHUYỂN ĐẠO DIII ĐẲNG ĐIỆN DO ĐẢO DÂY ĐẤT LA/RL")

        if v2.get("p_amp", 0.0) < 0 or v2.get("t_morph") in ["biphasic_pos_neg", "biphasic_neg_pos"] and (v2.get("r_amp", 0.0) < 3.0):
            self.findings.append(("Sai lệch điện cực", "Gợi ý Đặt điện cực V1-V2 quá cao (KLS 2 hoặc 3): Sóng P ở V2 âm hoặc hai pha (Dễ gây giả sóng Q, giả RBBB hoặc giả Brugada)"))[cite: 5]

    def evaluate_conduction(self):
        d1, d2, d3, avf = self.leads.get("I", {}), self.leads.get("II", {}), self.leads.get("III", {}), self.leads.get("aVF", {})

        # TÍNH TOÁN TRỤC ĐIỆN TIM
        if self.manual_override.get("axis_calc_mode") == "Tính theo biên độ DI và aVF":
            net_d1 = self.manual_override.get("net_d1", 0.0)
            net_avf = self.manual_override.get("net_avf", 0.0)
            net_d2 = self.manual_override.get("net_d2", 0.0)
            net_d3 = self.manual_override.get("net_d3", 0.0)
            try:
                rad = math.atan2(2.0 * net_avf, math.sqrt(3.0) * net_d1)
                self.calculated_axis_deg = int(round(math.degrees(rad)))
            except:
                self.calculated_axis_deg = 0
        else:
            net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
            net_d2 = d2.get("r_amp", 0.0) - d2.get("s_amp", 0.0)
            net_d3 = d3.get("r_amp", 0.0) - d3.get("s_amp", 0.0)
            net_avf = avf.get("r_amp", 0.0) - avf.get("s_amp", 0.0)

        is_extreme_lad, is_extreme_rad = False, False

        if net_d1 > 0 and net_avf >= 0:
            deg_txt = f" ({self.calculated_axis_deg}°)" if self.calculated_axis_deg is not None else ""
            self.axis_txt = f"Bình thường: 0° đến +90°{deg_txt}"[cite: 5]
        elif net_d1 > 0 and net_avf < 0:
            if net_d2 < 0:
                is_extreme_lad = True
                deg_txt = f" ({self.calculated_axis_deg}°)" if self.calculated_axis_deg is not None else ""
                self.axis_txt = f"LAD (Trục lệch quá trái bệnh lý: -30° đến -90°){deg_txt}"[cite: 2]
            else:
                deg_txt = f" ({self.calculated_axis_deg}°)" if self.calculated_axis_deg is not None else ""
                self.axis_txt = f"Trục lệch trái sinh lý: 0° đến -30°{deg_txt}"[cite: 2]
        elif net_d1 <= 0 and net_avf > 0:
            if net_d3 > 0 and abs(net_d1) > 2.0:
                is_extreme_rad = True
                deg_txt = f" ({self.calculated_axis_deg}°)" if self.calculated_axis_deg is not None else ""
                self.axis_txt = f"RAD (Trục lệch quá phải bệnh lý: ≥ +120°){deg_txt}"[cite: 2]
            else:
                deg_txt = f" ({self.calculated_axis_deg}°)" if self.calculated_axis_deg is not None else ""
                self.axis_txt = f"Trục lệch phải: +90° đến +120°{deg_txt}"[cite: 5]
        else:
            self.axis_txt = "Trục vô định (Tây Bắc: -90° đến 180°)"[cite: 4]

        # ---------------- BLOCK NHĨ THẤT (ƯU TIÊN GHI ĐÈ THỦ CÔNG) ----------------
        m_av = self.manual_override.get("av_block_choice", "Tự động")
        av_diag = None

        if m_av == "Block AV độ I":
            av_diag = f"Block nhĩ thất độ I: Khoảng PR cố định và kéo dài ({self.pr:.2f}s > 0.20s), theo sau mỗi sóng P là một phức bộ QRS"[cite: 2]
        elif m_av == "Block AV độ II Mobitz 1 (Wenckebach)":
            av_diag = "Block nhĩ thất độ II Mobitz type 1 (Chu kỳ Wenckebach): Khoảng PR tăng dần cho đến khi có 1 sóng P không dẫn"[cite: 2]
        elif m_av == "Block AV độ II Mobitz 2":
            av_diag = "Block nhĩ thất độ II Mobitz type 2: Khoảng PR cố định, có sóng P không dẫn đột ngột (RR không dẫn = 2 x RR bình thường)"[cite: 2]
            self.alerts.append("⚠️ CẢNH BÁO: BLOCK NHĨ THẤT ĐỘ II MOBITZ 2 - NGUY CƠ TIẾN TRIỂN THÀNH BLOCK CAO ĐỘ")
        elif m_av == "Block AV 2:1":
            av_diag = "Block nhĩ thất 2:1: Sóng P dẫn và P không dẫn xen kẽ nhau (P/QRS = 2:1), khoảng RR đều nhau"[cite: 2]
        elif m_av == "Block AV cao độ (≥ 3:1)":
            av_diag = "Block nhĩ thất cao độ: Có ít nhất hai sóng P liên tiếp không dẫn truyền (tỉ lệ P/QRS ≥ 3:1), khoảng PR cố định"[cite: 2]
            self.alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT CAO ĐỘ (≥ 3:1) - NGUY CƠ NGẤT")
        elif m_av == "Block AV độ III (Phân ly nhĩ thất)":
            escape_type = self.manual_override.get("av3_escape_type", "Nhịp thoát bộ nối (QRS hẹp)")
            if "bộ nối" in escape_type:
                av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát bộ nối (QRS hẹp, tần số 40-60 l/p)"[cite: 2]
                self.alerts.append("🚨 BLOCK NHĨ THẤT ĐỘ 3 TẠI NÚT: CHỈ ĐỊNH NHẬP VIỆN HỒI SỨC TIM MẠCH")
            else:
                av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát thất (QRS rộng ≥ 0.12s, tần số ≤ 40 l/p)"[cite: 2]
                self.alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT ĐỘ 3 DƯỚI NÚT - CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP")
        elif m_av == "Tự động":
            long_rrs = [r for r in self.rr_list if r > 1.6 * self.mean_rr]
            if self.hr <= 40 and self.qrs >= 0.12 and self.rr_cv < 0.05:
                av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát thất (QRS rộng ≥ 0.12s, tần số ≤ 40 l/p)"[cite: 2]
                self.alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT ĐỘ 3 DƯỚI NÚT - CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP")
            elif self.hr < 60 and self.qrs < 0.12 and self.rr_cv < 0.05 and self.hr <= 45:
                av_diag = "Block nhĩ thất độ III: Phân ly nhĩ thất hoàn toàn, nhịp thoát bộ nối (QRS hẹp, tần số 40-60 l/p)"[cite: 2]
                self.alerts.append("🚨 BLOCK NHĨ THẤT ĐỘ 3 TẠI NÚT: CHỈ ĐỊNH NHẬP VIỆN HỒI SỨC TIM MẠCH")
            elif len(long_rrs) > 0:
                if len(self.all_prs) >= 3 and (max(self.all_prs) - min(self.all_prs) >= 0.05):
                    av_diag = "Block nhĩ thất độ II Mobitz type 1 (Chu kỳ Wenckebach): Khoảng PR tăng dần cho đến khi có 1 sóng P không dẫn"[cite: 2]
                elif any(abs(r - 2.0 * self.mean_rr) < 0.15 for r in long_rrs):
                    av_diag = "Block nhĩ thất độ II Mobitz type 2: Khoảng PR cố định, có sóng P không dẫn đột ngột (RR không dẫn = 2 x RR bình thường)"[cite: 2]
                    self.alerts.append("⚠️ CẢNH BÁO: BLOCK NHĨ THẤT ĐỘ II MOBITZ 2 - NGUY CƠ TIẾN TRIỂN THÀNH BLOCK CAO ĐỘ")
                elif any(r >= 3.0 * self.mean_rr for r in long_rrs):
                    av_diag = "Block nhĩ thất cao độ: Có ít nhất hai sóng P liên tiếp không dẫn truyền (tỉ lệ P/QRS ≥ 3:1), khoảng PR cố định"[cite: 2]
                    self.alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT CAO ĐỘ (≥ 3:1) - NGUY CƠ NGẤT")
            elif any(abs(r - 2.0 * self.mean_rr) < 0.12 for r in self.rr_list) and len(self.rr_list) >= 4 and self.rr_cv < 0.08 and self.hr < 55:
                av_diag = "Block nhĩ thất 2:1: Sóng P dẫn và P không dẫn xen kẽ nhau (P/QRS = 2:1), khoảng RR đều nhau"[cite: 2]
            elif self.pr > 0.20:
                av_diag = f"Block nhĩ thất độ I: Khoảng PR cố định và kéo dài ({self.pr:.2f}s > 0.20s), theo sau mỗi sóng P là một phức bộ QRS"[cite: 2]

        if av_diag:
            self.findings.append(("Block Nhĩ Thất", av_diag))

        # ---------------- BLOCK XOANG NHĨ ----------------
        m_sa = self.manual_override.get("sa_block_choice", "Tự động")
        if m_sa == "Block xoang nhĩ độ II Type 1":
            self.findings.append(("Block Xoang Nhĩ", "Block xoang nhĩ độ II type 1: Khoảng PP và RR ngắn dần cho đến khi có khoảng nghỉ mất hẳn sóng P (< 2 x PP)"))[cite: 2]
        elif m_sa == "Block xoang nhĩ độ II Type 2":
            self.findings.append(("Block Xoang Nhĩ", "Block xoang nhĩ độ II type 2: Khoảng PP cố định, có khoảng nghỉ đột ngột bằng bội số nguyên của PP"))[cite: 2]
        elif m_sa == "Ngưng xoang > 3 giây":
            self.findings.append(("Hội chứng suy nút xoang", "Khoảng ngưng xoang kéo dài > 3 giây - Nguy cơ ngất Adams-Stokes"))[cite: 2]
            self.alerts.append("🚨 CẢNH BÁO: NGƯNG XOANG > 3 GIÂY - CHỈ ĐỊNH TẠO NHỊP")

        # ---------------- BLOCK DẪN TRUYỀN NỘI THẤT & PHÂN NHÁNH ----------------
        m_bbb = self.manual_override.get("bbb_choice", "Tự động")
        m_fasc = self.manual_override.get("fascicular_choice", "Tự động")

        rbbb_type = None
        lbbb_type = None
        lafb = False
        lpfb = False

        if m_bbb == "Block nhánh phải hoàn toàn (Complete RBBB)":
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 có dạng chữ M (rsR'/rSR'), DI và V6 có sóng S rộng ≥ 0.04s"[cite: 2]
        elif m_bbb == "Block nhánh phải không hoàn toàn (Incomplete RBBB)":
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS < 0.12s, V1 có dạng chữ M (rsR') và S rộng ở DI/V6"[cite: 2]
        elif m_bbb == "Block nhánh trái hoàn toàn (Complete LBBB)":
            lbbb_type = "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, V5-V6/DI/aVL sóng R đơn pha có móc, V1 dạng QS/rS"[cite: 2]
            self.alerts.append("🚨 LBBB HOÀN TOÀN: CẦN ĐỐI CHIẾU TIÊU CHUẨN SGARBOSSA LOẠI TRỪ NMCT THÀNH TRƯỚC")
        elif m_bbb == "Block nhánh trái không hoàn toàn (Incomplete LBBB)":
            lbbb_type = "Block nhánh trái không hoàn toàn (Incomplete LBBB): QRS < 0.12s"[cite: 2]
        elif m_bbb == "Chậm dẫn truyền nội thất không đặc hiệu (IVCD)":
            self.findings.append(("Block Dẫn Truyền Nội Thất", f"Chậm dẫn truyền nội thất không đặc hiệu (IVCD): QRS giãn rộng ({self.qrs:.3f}s > 0.11s)"))[cite: 2]
        elif m_bbb == "Tự động":
            v1, v5, v6 = self.leads.get("V1", {}), self.leads.get("V5", {}), self.leads.get("V6", {})
            v1_has_m = v1.get("true_rsr", False) or (v1.get("r_amp", 0.0) > v1.get("s_amp", 0.0) and v1.get("r_amp", 0.0) > 4.0)
            lateral_broad_s = v6.get("slurred_s", False) or d1.get("slurred_s", False)

            if v1_has_m and lateral_broad_s:
                if self.qrs >= 0.12:
                    rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 có dạng chữ M (rsR'/rSR'), DI và V6 có sóng S rộng ≥ 0.04s"[cite: 2]
                else:
                    rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS < 0.12s, V1 có dạng chữ M (rsR') và S rộng ở DI/V6"[cite: 2]

            v5_v6_notched = v5.get("notched_r", False) or v6.get("notched_r", False) or d1.get("notched_r", False)
            v1_qs_rs = (v1.get("s_amp", 0.0) > 8.0 and v1.get("r_amp", 0.0) < 2.5)
            no_q_lateral = (v5.get("q_amp", 0.0) == 0 and v6.get("q_amp", 0.0) == 0 and d1.get("q_amp", 0.0) == 0)

            if v5_v6_notched and v1_qs_rs and no_q_lateral and not v1_has_m:
                if self.qrs >= 0.12:
                    lbbb_type = "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, V5-V6/DI/aVL sóng R đơn pha có móc, V1 dạng QS/rS"[cite: 2]
                    self.alerts.append("🚨 LBBB HOÀN TOÀN: CẦN ĐỐI CHIẾU TIÊU CHUẨN SGARBOSSA LOẠI TRỪ NMCT THÀNH TRƯỚC")
                else:
                    lbbb_type = "Block nhánh trái không hoàn toàn (Incomplete LBBB): QRS < 0.12s"[cite: 2]

            if self.qrs > 0.11 and not rbbb_type and not lbbb_type:
                self.findings.append(("Block Dẫn Truyền Nội Thất", f"Chậm dẫn truyền nội thất không đặc hiệu (IVCD): QRS giãn rộng ({self.qrs:.3f}s > 0.11s)"))[cite: 2]

        if rbbb_type:
            self.findings.append(("Block Dẫn Truyền Nội Thất", rbbb_type))
        if lbbb_type:
            self.findings.append(("Block Dẫn Truyền Nội Thất", lbbb_type))

        if m_fasc == "Block phân nhánh trái trước (LAFB)":
            lafb = True
            self.findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch quá trái (-30° đến -90°); DI, aVL dạng qR; DII, DIII, aVF dạng rS"))[cite: 2]
        elif m_fasc == "Block phân nhánh trái sau (LPFB)":
            lpfb = True
            self.findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch quá phải (≥ 120°); DI, aVL dạng rS; DII, DIII, aVF dạng qR"))[cite: 2]
        elif m_fasc == "Tự động":
            if is_extreme_lad and self.qrs < 0.12:
                d1_qr = (d1.get("r_amp", 0.0) > 0 and d1.get("s_amp", 0.0) < d1.get("r_amp", 0.0))
                d3_rs = (d3.get("s_amp", 0.0) > d3.get("r_amp", 0.0))
                if d1_qr and d3_rs:
                    lafb = True
                    self.findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch quá trái (-30° đến -90°); DI, aVL dạng qR; DII, DIII, aVF dạng rS"))[cite: 2]
            if is_extreme_rad and self.qrs < 0.12:
                d1_rs = (d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0))
                d3_qr = (d3.get("r_amp", 0.0) > d3.get("s_amp", 0.0))
                v1_r_val = self.leads.get("V1", {}).get("r_amp", 0.0)
                if d1_rs and d3_qr and v1_r_val < 6.0:
                    lpfb = True
                    self.findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch quá phải (≥ 120°); DI, aVL dạng rS; DII, DIII, aVF dạng qR"))[cite: 2]

        # Block 2 & 3 phân nhánh
        if rbbb_type and "Complete" in rbbb_type:
            if lafb:
                if av_diag and "độ I" in av_diag:
                    self.findings.append(("Block Ba Nhánh", "Block ba nhánh (Trifascicular Block): Block nhánh phải + Block phân nhánh trái trước + Block nhĩ thất độ 1"))[cite: 2]
                    self.alerts.append("⚠️ CẢNH BÁO: BLOCK BA NHÁNH - THEO DÕI NGUY CƠ TIẾN TRIỂN THÀNH BLOCK TIM HOÀN TOÀN")
                else:
                    self.findings.append(("Block Hai Nhánh", "Block hai nhánh (Bifascicular Block): Block nhánh phải kèm Block phân nhánh trái trước"))[cite: 2]
            elif lpfb:
                if av_diag and "độ I" in av_diag:
                    self.findings.append(("Block Ba Nhánh", "Block ba nhánh (Trifascicular Block): Block nhánh phải + Block phân nhánh trái sau + Block nhĩ thất độ 1"))[cite: 2]
                else:
                    self.findings.append(("Block Hai Nhánh", "Block hai nhánh (Bifascicular Block): Block nhánh phải kèm Block phân nhánh trái sau"))[cite: 2]

    def evaluate_preexcitation_and_ectopics(self):
        any_delta = any(l.get("has_delta", False) for l in self.leads.values())
        v1_r, v1_s = self.leads.get("V1", {}).get("r_amp", 0.0), self.leads.get("V1", {}).get("s_amp", 0.0)
        v2_r, v2_s = self.leads.get("V2", {}).get("r_amp", 0.0), self.leads.get("V2", {}).get("s_amp", 0.0)

        is_v1_v2_pos = (v1_r >= v1_s) and (v2_r >= v2_s)
        is_v1_v2_neg = (v1_s > v1_r) and (v2_s > v2_r)

        if self.pr < 0.12 and (self.qrs > 0.10 or any_delta):
            is_intermittent = any(p >= 0.13 for p in self.all_prs) and any(p < 0.12 for p in self.all_prs)
            if is_v1_v2_pos:
                wpw_type = "Hội chứng Wolff-Parkinson-White (WPW) Type A: PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta, QRS dương ở V1-V2 (Bó Kent bên trái)"[cite: 3]
            elif is_v1_v2_neg:
                wpw_type = "Hội chứng Wolff-Parkinson-White (WPW) Type B: PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta, QRS âm ở V1-V2 (Bó Kent bên phải)"[cite: 3]
            else:
                wpw_type = "Hội chứng Wolff-Parkinson-White (WPW): PR ngắn (< 0.12s), QRS giãn rộng có sóng Delta"[cite: 3]

            if is_intermittent:
                wpw_type += " [Dạng WPW từng lúc]"[cite: 3]

            self.findings.append(("Hội chứng kích thích sớm", wpw_type))
            self.alerts.append("⚠️ HỘI CHỨNG WPW: TRÁNH DÙNG THUỐC CHẸN NÚT NHĨ THẤT (DIGOXIN, VERAPAMIL)")
        elif self.pr < 0.12 and self.qrs <= 0.10 and not any_delta:
            self.findings.append(("Hội chứng kích thích sớm", "Hội chứng Lown-Ganong-Levine (LGL): Khoảng PR ngắn (< 0.12s), QRS bình thường, không có sóng Delta (Bó James)"))[cite: 3]

        early_indices = [i for i, r in enumerate(self.rr_list[:-1]) if r < 0.80 * self.mean_rr]
        if len(early_indices) > 0:
            pvc_count, pac_count = 0, 0
            pvc_origins = []
            consecutive_pvcs, max_consecutive_pvcs = 0, 0

            for idx in early_indices:
                r_early = self.rr_list[idx]
                r_next = self.rr_list[idx + 1] if idx + 1 < len(self.rr_list) else self.mean_rr
                cycle_pair = r_early + r_next
                is_full_compensatory = abs(cycle_pair - 2.0 * self.mean_rr) < 0.14 * self.mean_rr

                if self.qrs >= 0.12 or self.leads.get("V1", {}).get("qrs_w", 0.08) >= 0.12:
                    pvc_count += 1
                    consecutive_pvcs += 1
                    max_consecutive_pvcs = max(max_consecutive_pvcs, consecutive_pvcs)
                    pvc_origins.append("Thất trái" if v1_r >= v1_s else "Thất phải")[cite: 3]
                else:
                    consecutive_pvcs = 0
                    if not is_full_compensatory:
                        pac_count += 1

            if pvc_count > 0:
                origin_str = f"xuất phát từ {max(set(pvc_origins), key=pvc_origins.count)}" if pvc_origins else ""
                if max_consecutive_pvcs >= 3:
                    vt_rate = int(60.0 / (self.mean_rr * 0.65)) if self.mean_rr > 0 else 130
                    self.findings.append(("Ngoại tâm thu thất", f"Cơn nhanh thất ngắn: Có {max_consecutive_pvcs} ngoại tâm thu thất liên tiếp, tần số {vt_rate} l/p"))[cite: 3]
                elif max_consecutive_pvcs == 2:
                    self.findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất cặp đôi (Couplet), {origin_str}"))[cite: 3]
                else:
                    self.findings.append(("Ngoại tâm thu thất", f"Ngoại tâm thu thất (PVC), {origin_str}, nghỉ bù hoàn toàn"))[cite: 3]

            if pac_count > 0:
                self.findings.append(("Ngoại tâm thu nhĩ", "Ngoại tâm thu nhĩ (PAC): Nhịp đến sớm với sóng P' biến dạng, QRS hẹp, nghỉ bù không hoàn toàn"))[cite: 3]

    def evaluate_tachycardias(self):
        if self.hr >= 220 and self.rr_cv > 0.30 and self.qrs >= 0.16:
            self.findings.append(("Rối loạn nhịp thất ác tính", "Rung thất (Ventricular Fibrillation - VF): Hoạt động điện thất hỗn loạn, vô tổ chức, mất hoàn toàn phức bộ QRS"))[cite: 4]
            self.alerts.append("🚨 BÁO ĐỘNG ĐỎ: RUNG THẤT - BỆNH NHÂN NGƯNG TIM, KÍCH HOẠT CPR VÀ PHÁ RUNG NGAY")
            return

        if 180 <= self.hr <= 300 and self.qrs >= 0.16 and self.rr_cv < 0.08:
            self.findings.append(("Rối loạn nhịp thất ác tính", "Cuồng thất (Ventricular Flutter): Các sóng hình sin đều đặn liên tục, tần số 150-300 lần/phút"))[cite: 4]
            self.alerts.append("🚨 CẤP CỨU: CUỒNG THẤT - NGUY CƠ TIẾN TRIỂN THÀNH RUNG THẤT")
            return

        v1_to_v6_amps = [self.leads.get(f"V{i}", {}).get("r_amp", 0.0) for i in range(1, 7)]
        if self.hr >= 150 and max(v1_to_v6_amps) - min(v1_to_v6_amps) > 12.0 and self.rr_cv > 0.20:
            self.findings.append(("Rối loạn nhịp thất ác tính", "Xoắn đỉnh (Torsades de Pointes): Nhịp nhanh thất đa dạng xoắn quanh đường đẳng điện"))[cite: 4]
            self.alerts.append("🚨 CẤP CỨU: XOẮN ĐỈNH - TRUYỀN MAGNESIUM SULFATE")
            return

        if self.hr > 100 and self.qrs >= 0.12:
            vt_criteria_met = []
            d1, avf, v1 = self.leads.get("I", {}), self.leads.get("aVF", {}), self.leads.get("V1", {})

            net_d1 = d1.get("r_amp", 0.0) - d1.get("s_amp", 0.0)
            net_avf = avf.get("r_amp", 0.0) - avf.get("s_amp", 0.0)
            if net_d1 < 0 and net_avf < 0:
                vt_criteria_met.append("Trục QRS vô định (DI âm, aVF âm)")[cite: 4]
            if self.qrs > 0.16:
                vt_criteria_met.append(f"QRS rất rộng ({self.qrs*1000:.0f} ms > 160 ms)")[cite: 4]

            chest_signs = [(self.leads.get(f"V{i}", {}).get("r_amp", 0.0) - self.leads.get(f"V{i}", {}).get("s_amp", 0.0)) for i in range(1, 7)]
            if all(s > 0 for s in chest_signs):
                vt_criteria_met.append("QRS đồng hướng dương ở V1-V6")[cite: 4]
            elif all(s < 0 for s in chest_signs):
                vt_criteria_met.append("QRS đồng hướng âm ở V1-V6")[cite: 4]

            if v1.get("left_rabbit_ear", False):
                vt_criteria_met.append("Dấu hiệu tai thỏ với tai trái lớn hơn ở V1 (R > R')")[cite: 4]

            max_rs_time = max(l.get("r_to_s_time", 0.0) for l in self.leads.values())
            if max_rs_time > 0.10:
                vt_criteria_met.append(f"Dấu hiệu Brugada dương tính (R đến đáy S = {max_rs_time*1000:.0f} ms > 100 ms)")[cite: 4]

            if len(vt_criteria_met) >= 2 or (self.qrs > 0.16 and len(vt_criteria_met) >= 1):
                self.findings.append(("Nhịp nhanh thất (VT)", f"Nhịp nhanh thất (Ventricular Tachycardia): {'; '.join(vt_criteria_met)}"))[cite: 4]
                self.alerts.append("🚨 CẤP CỨU: NHỊP NHANH THẤT (VT) - ĐÁNH GIÁ NGAY HUYẾT ĐỘNG ĐỂ SỐC ĐIỆN")
                return

        if self.hr > 100 and self.qrs < 0.12:
            if self.rr_cv > 0.20 and self.p_ratio < 0.20:
                self.findings.append(("Rối loạn nhịp nhĩ", f"Rung nhĩ đáp ứng thất nhanh (AFib with RVR) - Tần số: {self.hr} l/p"))[cite: 4]
                return
            if 135 <= self.hr <= 165 and any(abs(l.get("st_shift", 0.0)) > 0 for l in self.leads.values()):
                self.findings.append(("Rối loạn nhịp nhĩ", f"Cuồng nhĩ (Atrial Flutter): Nghi ngờ cuồng nhĩ dẫn truyền 2:1 (Tần số thất {self.hr} l/p)"))[cite: 4]
                return

            rp_val = max(l.get("rp_interval", 0.0) for l in self.leads.values())
            pseudo_r = self.leads.get("V1", {}).get("pseudo_r_prime", False)

            if pseudo_r or (0 < rp_val <= 0.09):
                self.findings.append(("Nhịp nhanh vào lại nút nhĩ thất (AVNRT)", f"AVNRT thể điển hình (Slow-Fast): Nhịp nhanh đều ({self.hr} l/p), QRS hẹp, RP ≤ 90 ms"))[cite: 4]
            elif rp_val > 0.09 and rp_val < self.pr:
                self.findings.append(("Nhịp nhanh vào lại nhĩ thất (AVRT)", f"Orthodromic AVRT: Nhịp nhanh đều ({self.hr} l/p), QRS hẹp, RP dài > 90 ms"))[cite: 4]
            else:
                self.findings.append(("Nhịp nhanh trên thất", f"Nhịp nhanh kịch phát trên thất (SVT) - Tần số: {self.hr} l/p"))

    def evaluate_coronary_syndromes(self):
        st_elevation_leads, st_depression_leads = [], []
        hyperacute_t_leads, inverted_t_leads, flat_t_leads, biphasic_t_leads = [], [], [], []
        pathological_q_leads = []

        m_st_ant = self.manual_override.get("st_anterior", "Tự động")
        m_st_inf = self.manual_override.get("st_inferior", "Tự động")
        m_st_lat = self.manual_override.get("st_lateral", "Tự động")

        if m_st_ant == "ST chênh lên (STEMI)":
            for ld in ["V1", "V2", "V3", "V4"]:
                self.leads[ld]["st_shift"] = 2.5
        elif m_st_ant == "ST chênh xuống (Thiếu máu)":
            for ld in ["V1", "V2", "V3", "V4"]:
                self.leads[ld]["st_shift"] = -1.2
                self.leads[ld]["st_slope"] = "horizontal"

        if m_st_inf == "ST chênh lên (STEMI)":
            for ld in ["II", "III", "aVF"]:
                self.leads[ld]["st_shift"] = 2.0
        elif m_st_inf == "ST chênh xuống (Thiếu máu)":
            for ld in ["II", "III", "aVF"]:
                self.leads[ld]["st_shift"] = -1.0
                self.leads[ld]["st_slope"] = "downsloping"

        if m_st_lat == "ST chênh lên (STEMI)":
            for ld in ["I", "aVL", "V5", "V6"]:
                self.leads[ld]["st_shift"] = 1.8
        elif m_st_lat == "ST chênh xuống (Thiếu máu)":
            for ld in ["I", "aVL", "V5", "V6"]:
                self.leads[ld]["st_shift"] = -1.0
                self.leads[ld]["st_slope"] = "horizontal"

        for l_name, l_data in self.leads.items():
            if l_name == "aVR":
                continue

            st_val = l_data.get("st_shift", 0.0)
            t_val = l_data.get("t_amp", 0.0)
            t_morph = l_data.get("t_morph", "normal")
            r_val, s_val = l_data.get("r_amp", 0.0), l_data.get("s_amp", 0.0)
            q_dur, q_amp = l_data.get("q_dur", 0.0), l_data.get("q_amp", 0.0)
            slope = l_data.get("st_slope", "flat")

            cutoff = (1.5 if self.gender == "Nữ" else (2.0 if self.age >= 40 else 2.5)) if l_name in ["V2", "V3"] else 1.0[cite: 1]

            if st_val >= cutoff:
                st_elevation_leads.append(l_name)
            if st_val <= -0.5:
                st_depression_leads.append((l_name, slope))

            is_chest = l_name.startswith("V")
            if (is_chest and t_val > 10.0) or (not is_chest and t_val > 5.0) or (r_val > 0 and t_val > 0.75 * r_val):
                hyperacute_t_leads.append(l_name)[cite: 1]
            if t_val < -1.0 and (r_val > s_val or r_val > 5.0):
                inverted_t_leads.append(l_name)[cite: 1]
            if -1.0 <= t_val <= 1.0 and r_val > 3.0:
                flat_t_leads.append(l_name)[cite: 1]
            if t_morph == "biphasic_pos_neg":
                biphasic_t_leads.append(l_name)[cite: 1]

            if l_name in ["V2", "V3"]:
                if q_dur > 0.020 or (r_val == 0.0 and q_amp >= 2.0):
                    pathological_q_leads.append(l_name)[cite: 1]
            else:
                if (q_dur >= 0.038 and q_amp >= 1.0) or (r_val == 0.0 and q_amp >= 2.0):
                    pathological_q_leads.append(l_name)[cite: 1]

        st_set = set(st_elevation_leads)
        dep_leads_names = [item[0] for item in st_depression_leads]
        dep_set = set(dep_leads_names)

        avr_st = self.leads.get("aVR", {}).get("st_shift", 0.0)
        dewinter_candidates = [l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping" and l in hyperacute_t_leads]
        if len(dewinter_candidates) >= 2 or (avr_st >= 0.5 and len([l for (l, sl) in st_depression_leads if l.startswith("V") and sl == "upsloping"]) >= 2):
            self.findings.append(("Hội chứng mạch vành cấp", "Hội chứng De Winter: Điểm J chênh xuống đi lên ở V1-V6 kèm sóng T cao đối xứng (Tương đương STEMI tắc đoạn gần LAD)"))[cite: 1]
            self.alerts.append("🚨 CẤP CỨU: HỘI CHỨNG DE WINTER - CAN THIỆP MẠCH VÀNH KHẨN CẤP")

        wellens_a = [l for l in biphasic_t_leads if l in ["V1", "V2", "V3"]]
        wellens_b = [l for l in inverted_t_leads if l in ["V1", "V2", "V3", "V4"]]
        if len(wellens_a) >= 2:
            self.findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type A: Sóng T hai pha (+/-) tại {', '.join(wellens_a)} (Gợi ý hẹp nặng đoạn gần LAD)"))[cite: 1]
            self.alerts.append("⚠️ HỘI CHỨNG WELLENS TYPE A: NGUY CƠ TIẾN TRIỂN THÀNH NMCT DIỆN RỘNG")
        elif len(wellens_b) >= 2:
            self.findings.append(("Hội chứng mạch vành cấp", f"Hội chứng Wellens Type B: Sóng T âm sâu đối xứng tại {', '.join(wellens_b)} (Gợi ý hẹp nặng đoạn gần LAD)"))[cite: 1]
            self.alerts.append("⚠️ HỘI CHỨNG WELLENS TYPE B: CHỈ ĐỊNH CHỤP MẠCH VÀNH SỚM")

        stemi_regions, culprit_artery = [], []
        if {"V1", "V2", "V3", "V4", "V5", "V6"}.issubset(st_set) or ({"V1", "V2", "V3", "V4"}.issubset(st_set) and {"I", "aVL"}.intersection(st_set)):
            recip = " (Soi gương ở DII, DIII, aVF)" if len({"II", "III", "aVF"}.intersection(dep_set)) >= 1 else ""[cite: 1]
            stemi_regions.append(f"Thành trước rộng (Extensive Anterior: V1-V6, DI, aVL){recip}")[cite: 1]
            culprit_artery.append("Đoạn gần LAD (pLAD)")[cite: 1]
        elif {"V1", "V2", "V3", "V4"}.issubset(st_set):
            stemi_regions.append("Thành trước vách (Anteroseptal: V1-V4)")[cite: 1]
            culprit_artery.append("LAD (trước S1/D1)")[cite: 1]
        elif len({"V2", "V3", "V4", "V5"}.intersection(st_set)) >= 3:
            stemi_regions.append("Thành trước (Anterior: V2-V5)")[cite: 1]
            culprit_artery.append("LAD")[cite: 1]
        elif {"V1", "V2"}.issubset(st_set):
            stemi_regions.append("Vách liên thất (Septal: V1-V2)")[cite: 1]
            culprit_artery.append("Nhánh vách của LAD")[cite: 1]
        elif {"V3", "V4"}.issubset(st_set):
            stemi_regions.append("Thành trước (Anterior: V3-V4)")[cite: 1]
            culprit_artery.append("LAD đoạn giữa")[cite: 1]

        if {"I", "aVL"}.issubset(st_set) and not {"V5", "V6"}.intersection(st_set):
            recip = " (Soi gương ở DII, DIII, aVF)" if len({"II", "III", "aVF"}.intersection(dep_set)) >= 1 else ""[cite: 1]
            stemi_regions.append(f"Thành bên cao đơn thuần (High Lateral: DI, aVL){recip}")[cite: 1]
            culprit_artery.append("Nhánh D1 của LAD hoặc LCx")[cite: 1]
        elif {"V5", "V6"}.issubset(st_set) and not {"I", "aVL"}.intersection(st_set):
            stemi_regions.append("Thành bên thấp (Low Lateral: V5-V6)")[cite: 1]
            culprit_artery.append("Đoạn xa LAD (dLAD)")[cite: 1]
        elif {"V5", "V6", "I", "aVL"}.issubset(st_set):
            stemi_regions.append("Thành bên toàn bộ (Lateral: V5, V6, DI, aVL)")[cite: 1]
            culprit_artery.append("LCx hoặc nhánh D1 của LAD")[cite: 1]

        inferior_leads = {"II", "III", "aVF"}.intersection(st_set)
        if len(inferior_leads) >= 2:
            recip = " (Soi gương ở aVL, DI, V1-V3)" if len({"aVL", "I", "V1", "V2"}.intersection(dep_set)) >= 1 else ""[cite: 1]
            stemi_regions.append(f"Thành dưới (Inferior: {', '.join(sorted(list(inferior_leads)))}){recip}")[cite: 1]
            culprit_artery.append("RCA (80%) hoặc LCx (20%)")[cite: 1]

        if stemi_regions:
            reg_txt = " + ".join(stemi_regions)
            art_txt = f" - ĐM thủ phạm dự đoán: {', '.join(set(culprit_artery))}" if culprit_artery else ""
            has_q = any(l in set(pathological_q_leads) for l in st_set)
            stage_str = "Bán cấp / Hoại tử (Đã có sóng Q)" if has_q else "Tối cấp / Cấp tính"
            self.findings.append(("Hội chứng mạch vành cấp (STEMI)", f"Nhồi máu cơ tim ST chênh lên - Vùng: {reg_txt} - Giai đoạn: {stage_str}{art_txt}"))
            self.alerts.append(f"🚨 CẤP CỨU: STEMI VÙNG {reg_txt.upper()} - KÍCH HOẠT PCI KHẨN CẤP")
        elif len(st_depression_leads) >= 2 or len(inverted_t_leads) >= 2 or len(hyperacute_t_leads) >= 2:
            ischemia_details = []
            spec_dep = [f"{l} (dạng {sl})" for (l, sl) in st_depression_leads if sl in ["horizontal", "downsloping"]]
            if len(spec_dep) >= 2:
                ischemia_details.append(f"ST chênh xuống đặc hiệu tại: {', '.join(spec_dep)}")[cite: 1]
            elif len(dep_leads_names) >= 2:
                ischemia_details.append(f"ST chênh xuống tại: {', '.join(dep_leads_names)}")[cite: 1]

            if len(inverted_t_leads) >= 2:
                ischemia_details.append(f"Sóng T âm sâu đảo ngược tại: {', '.join(inverted_t_leads)}")[cite: 1]

            self.findings.append(("Thiếu máu cục bộ cơ tim (NSTE-ACS)", f"Biến đổi thiếu máu cơ tim cấp: {'; '.join(ischemia_details)}"))
            self.alerts.append("⚠️ CẢNH BÁO: THEO DÕI NSTE-ACS - ĐỊNH LƯỢNG TROPONIN HS")
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
                self.findings.append(("Hội chứng mạch vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI theo chuẩn ESC 2018) - Vùng: {', '.join(old_mi_regions)}"))[cite: 1]

    def evaluate_hypertrophy(self):
        d1, d2, d3, avf = self.leads.get("I", {}), self.leads.get("II", {}), self.leads.get("III", {}), self.leads.get("aVF", {})
        avl, avr, v1, v2 = self.leads.get("aVL", {}), self.leads.get("aVR", {}), self.leads.get("V1", {}), self.leads.get("V2", {})
        v3, v4, v5, v6 = self.leads.get("V3", {}), self.leads.get("V4", {}), self.leads.get("V5", {}), self.leads.get("V6", {})

        p_amp_d2 = d2.get("p_amp", 0.0)
        p_dur_d2 = d2.get("p_dur", 0.08)
        p_notched_d2 = d2.get("p_notched", False)
        v1_pos_p = v1.get("p_pos_amp", 0.0)
        v1_neg_p = v1.get("p_neg_amp", 0.0)
        v1_neg_dur = v1.get("p_neg_dur", 0.0)

        max_p_limb = max(p_amp_d2, d3.get("p_amp", 0.0), avf.get("p_amp", 0.0))
        is_rah = (max_p_limb >= 2.5) or (v1_pos_p > 1.5)[cite: 5]
        is_lah = (p_dur_d2 > 0.11 and p_notched_d2) or (v1_neg_p > 1.0 and v1_neg_dur > 0.04)[cite: 5]

        if is_rah and is_lah:
            self.findings.append(("Lớn buồng tim", "Lớn hai nhĩ: Sóng P vừa cao ≥ 2.5 mm vừa rộng ≥ 0.12s ở DII; tại V1 pha dương cao > 1.5 mm và pha âm sâu > 1 mm, rộng > 0.04s"))[cite: 5]
        elif is_rah:
            self.findings.append(("Lớn buồng tim", f"Lớn nhĩ phải (P phế): Sóng P cao {max_p_limb:.1f} mm (≥ 2.5 mm ở DII/DIII/aVF)"))[cite: 5]
        elif is_lah:
            self.findings.append(("Lớn buồng tim", "Lớn nhĩ trái (P nhĩ): Sóng P ở DII có dạng 2 đỉnh (lưng lạc đà) rộng > 0.11s; tại V1 pha âm rộng > 0.04s và sâu > 1 mm"))[cite: 5]

        s_v1 = self.manual_override["sv1"] if self.manual_override.get("sv1", 0.0) > 0 else v1.get("s_amp", 0.0)
        r_v5 = self.manual_override["rv5"] if self.manual_override.get("rv5", 0.0) > 0 else v5.get("r_amp", 0.0)
        r_v6 = self.manual_override["rv6"] if self.manual_override.get("rv6", 0.0) > 0 else v6.get("r_amp", 0.0)
        r_avl = self.manual_override["ravl"] if self.manual_override.get("ravl", 0.0) > 0 else avl.get("r_amp", 0.0)
        s_v3 = self.manual_override["sv3"] if self.manual_override.get("sv3", 0.0) > 0 else v3.get("s_amp", 0.0)
        r_v1 = self.manual_override["rv1"] if self.manual_override.get("rv1", 0.0) > 0 else v1.get("r_amp", 0.0)
        s_v5 = self.manual_override["sv5"] if self.manual_override.get("sv5", 0.0) > 0 else v5.get("s_amp", 0.0)

        self.sokolow_lv = s_v1 + max(r_v5, r_v6)
        self.sokolow_rv = r_v1 + max(s_v5, v6.get("s_amp", 0.0))
        self.cornell = r_avl + s_v3
        cornell_cutoff = 28.0 if self.gender == "Nam" else 20.0[cite: 5]

        lvh_voltage_met = []
        if (d1.get("r_amp", 0.0) + d3.get("s_amp", 0.0)) > 25.0:
            lvh_voltage_met.append("R(DI) + S(DIII) > 25 mm")[cite: 5]
        if r_avl > 11.0:
            lvh_voltage_met.append(f"RaVL = {r_avl:.1f} mm (> 11 mm)")[cite: 5]
        if self.sokolow_lv >= 35.0:
            lvh_voltage_met.append(f"Sokolow-Lyon = {self.sokolow_lv:.1f} mm (≥ 35 mm)")[cite: 5]
        if self.cornell > cornell_cutoff:
            lvh_voltage_met.append(f"Cornell = {self.cornell:.1f} mm (> {cornell_cutoff:.0f} mm ở {self.gender})")[cite: 5]

        has_lvh_strain = (v5.get("st_shift", 0.0) <= -0.5 and v5.get("t_amp", 0.0) < 0) or (v6.get("st_shift", 0.0) <= -0.5 and v6.get("t_amp", 0.0) < 0)[cite: 5]

        self.re_score = 0
        if (r_avl >= 11.0 or max(r_v5, r_v6) >= 30.0 or max(v1.get("s_amp", 0.0), v2.get("s_amp", 0.0)) >= 30.0):
            self.re_score += 3[cite: 5]
        if has_lvh_strain:
            self.re_score += 3[cite: 5]
        if is_lah:
            self.re_score += 3[cite: 5]
        if "LAD" in self.axis_txt or "lệch trái" in self.axis_txt.lower():
            self.re_score += 2[cite: 5]
        if self.qrs >= 0.09:
            self.re_score += 1[cite: 5]

        is_lvh_confirmed = (len(lvh_voltage_met) >= 1) or (self.re_score >= 5)[cite: 5]

        is_rad = "phải" in self.axis_txt.lower()
        rvh_signs = []
        if is_rad:
            rvh_signs.append("Trục lệch phải ≥ 110°")[cite: 5]
        if r_v1 >= 7.0:
            rvh_signs.append(f"Sóng R ưu thế ở V1 (R={r_v1:.1f} mm)")[cite: 5]
        if self.sokolow_rv >= 11.0:
            rvh_signs.append(f"Sokolow-Lyon phải = {self.sokolow_rv:.1f} mm (≥ 11 mm)")[cite: 5]

        is_rvh_confirmed = (len(rvh_signs) >= 2) and (r_v1 >= 6.0 or is_rad)

        katz_wachtel_val = max([(self.leads.get(f"V{i}", {}).get("r_amp", 0.0) + self.leads.get(f"V{i}", {}).get("s_amp", 0.0)) for i in range(2, 6)])
        if katz_wachtel_val > 50.0:
            self.findings.append(("Phì đại buồng tim", f"Dày hai thất theo tiêu chuẩn Katz-Wachtel: Tổng R + S ở V2-V5 = {katz_wachtel_val:.1f} mm (> 50 mm)"))[cite: 5]
        elif is_lvh_confirmed and is_rvh_confirmed:
            self.findings.append(("Phì đại buồng tim", "Dày hai thất: Thỏa mãn đồng thời tiêu chuẩn dày thất trái và dày thất phải"))
        elif is_lvh_confirmed:
            strain_txt = " kèm kiểu hình tăng gánh tâm thu thất trái (ST chênh xuống và T âm ở DI, aVL, V5, V6)" if has_lvh_strain else ""[cite: 5]
            self.findings.append(("Phì đại buồng tim", f"Dày thất trái (LVH): Thỏa {', '.join(lvh_voltage_met[:2])}; Điểm Romhilt-Estes = {self.re_score} điểm{strain_txt}"))[cite: 5]
        elif is_rvh_confirmed:
            self.findings.append(("Phì đại buồng tim", f"Dày thất phải (RVH): {'; '.join(rvh_signs[:2])}"))

    def analyze_all(self):
        self.evaluate_artifacts()
        self.evaluate_conduction()
        self.evaluate_preexcitation_and_ectopics()
        self.evaluate_tachycardias()
        self.evaluate_coronary_syndromes()
        self.evaluate_hypertrophy()

        if not any(k in f[0] for f in self.findings for k in ["Rối loạn nhịp", "Nhịp nhanh", "Block nhĩ thất", "Hội chứng kích thích sớm"]):
            if self.hr > 100:
                self.findings.append(("Nhịp học", f"Nhịp nhanh xoang: Tần số {self.hr} lần/phút (chuẩn DII)"))[cite: 4]
            elif self.hr < 60:
                self.findings.append(("Nhịp học", f"Nhịp chậm xoang: Tần số {self.hr} lần/phút (chuẩn DII)"))[cite: 2]
            else:
                self.findings.append(("Nhịp học", f"Nhịp xoang bình thường: Tần số {self.hr} lần/phút (chuẩn DII)"))

        return {
            "axis": self.axis_txt,
            "sokolow": self.sokolow_lv,
            "sokolow_rv": self.sokolow_rv,
            "cornell": self.cornell,
            "re_score": self.re_score,
            "findings": self.findings,
            "alerts": self.alerts
        }

# =========================================================================
# 4. GIAO DIỆN STREAMLIT
# =========================================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Bản Ghi ECG & Hỗ Trợ Lâm Sàng")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        gender_choice = st.radio("Giới tính bệnh nhân", ["Nam", "Nữ"], horizontal=True)
    with c_p2:
        age_choice = st.number_input("Tuổi", min_value=1, max_value=120, value=55)

    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Bản ghi ECG nạp vào hệ thống chẩn đoán YDS 2026", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên để phân tích.")

    st.markdown("---")
    use_manual_override = st.toggle("🛠️ Bật Chế độ Nhập / Tinh chỉnh thông số thủ công (Hỗ trợ AI)", value=False)
    
    manual_data = {}
    if use_manual_override:
        st.markdown("##### 📌 Tần số DII & Dẫn truyền nhĩ thất")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            manual_data["hr"] = st.number_input("Tần số tim DII (l/p)", min_value=20, max_value=300, value=75)
        with m_col2:
            manual_data["pr"] = st.number_input("Khoảng PR (giây)", min_value=0.04, max_value=0.50, value=0.16, step=0.01)
        with m_col3:
            manual_data["qrs"] = st.number_input("Độ rộng QRS (giây)", min_value=0.04, max_value=0.30, value=0.08, step=0.01)

        # MỤC MỚI BỔ SUNG: BLOCK NHĨ THẤT, BLOCK NHÁNH, BLOCK XOANG NHĨ
        st.markdown("##### ⚡ Phân tầng Block Dẫn truyền & Block Nhĩ Thất")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            manual_data["av_block_choice"] = st.selectbox(
                "Phân độ Block Nhĩ Thất (AV Block)", 
                ["Tự động", "Không", "Block AV độ I", "Block AV độ II Mobitz 1 (Wenckebach)", "Block AV độ II Mobitz 2", "Block AV 2:1", "Block AV cao độ (≥ 3:1)", "Block AV độ III (Phân ly nhĩ thất)"]
            )
            if manual_data["av_block_choice"] == "Block AV độ III (Phân ly nhĩ thất)":
                manual_data["av3_escape_type"] = st.radio("Loại nhịp thoát (Block độ 3)", ["Nhịp thoát bộ nối (QRS hẹp)", "Nhịp thoát thất (QRS rộng)"], horizontal=True)

            manual_data["sa_block_choice"] = st.selectbox(
                "Block Xoang Nhĩ & Ngưng Xoang",
                ["Tự động", "Không", "Block xoang nhĩ độ II Type 1", "Block xoang nhĩ độ II Type 2", "Ngưng xoang > 3 giây"]
            )

        with b_col2:
            manual_data["bbb_choice"] = st.selectbox(
                "Block Nhánh (Bundle Branch Block)",
                ["Tự động", "Không", "Block nhánh phải hoàn toàn (Complete RBBB)", "Block nhánh phải không hoàn toàn (Incomplete RBBB)", "Block nhánh trái hoàn toàn (Complete LBBB)", "Block nhánh trái không hoàn toàn (Incomplete LBBB)", "Chậm dẫn truyền nội thất không đặc hiệu (IVCD)"]
            )
            manual_data["fascicular_choice"] = st.selectbox(
                "Block Phân Nhánh (Fascicular Block)",
                ["Tự động", "Không", "Block phân nhánh trái trước (LAFB)", "Block phân nhánh trái sau (LPFB)"]
            )

        st.markdown("##### 📐 Tính toán Trục điện tim (Nhập biên độ đại số Net = R - S)")
        manual_data["axis_calc_mode"] = st.radio("Chế độ tính trục:", ["Tự động từ ảnh", "Tính theo biên độ DI và aVF"], horizontal=True)
        if manual_data["axis_calc_mode"] == "Tính theo biên độ DI và aVF":
            ax_c1, ax_c2, ax_c3, ax_c4 = st.columns(4)
            with ax_c1:
                manual_data["net_d1"] = st.number_input("Net DI (mm)", min_value=-50.0, max_value=50.0, value=8.0, step=0.5, help="Biên độ R trừ S ở DI")
            with ax_c2:
                manual_data["net_d2"] = st.number_input("Net DII (mm)", min_value=-50.0, max_value=50.0, value=6.0, step=0.5, help="Biên độ R trừ S ở DII")
            with ax_c3:
                manual_data["net_d3"] = st.number_input("Net DIII (mm)", min_value=-50.0, max_value=50.0, value=-2.0, step=0.5, help="Biên độ R trừ S ở DIII")
            with ax_c4:
                manual_data["net_avf"] = st.number_input("Net aVF (mm)", min_value=-50.0, max_value=50.0, value=3.0, step=0.5, help="Biên độ R trừ S ở aVF")

        st.markdown("##### 📏 Biên độ tính Dày thất (Sokolow-Lyon, Cornell)")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            manual_data["sv1"] = st.number_input("S ở V1 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            manual_data["rv5"] = st.number_input("R ở V5 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        with v_col2:
            manual_data["rv6"] = st.number_input("R ở V6 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            manual_data["ravl"] = st.number_input("R ở aVL (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        with v_col3:
            manual_data["sv3"] = st.number_input("S ở V3 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            manual_data["rv1"] = st.number_input("R ở V1 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
            manual_data["sv5"] = st.number_input("S ở V5 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

        st.markdown("##### 🫀 Hình thái đoạn ST theo vùng giải phẫu")
        st_c1, st_c2, st_c3 = st.columns(3)
        with st_c1:
            manual_data["st_anterior"] = st.selectbox("Thành trước (V1-V4)", ["Tự động", "Đẳng điện", "ST chênh lên (STEMI)", "ST chênh xuống (Thiếu máu)"])
        with st_c2:
            manual_data["st_inferior"] = st.selectbox("Thành dưới (DII, DIII, aVF)", ["Tự động", "Đẳng điện", "ST chênh lên (STEMI)", "ST chênh xuống (Thiếu máu)"])
        with st_c3:
            manual_data["st_lateral"] = st.selectbox("Thành bên (DI, aVL, V5, V6)", ["Tự động", "Đẳng điện", "ST chênh lên (STEMI)", "ST chênh xuống (Thiếu máu)"])

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Chuyên Khoa Toàn Diện")
    if uploaded:
        with st.spinner("Đang tính tần số chuẩn DII, đối chiếu tiêu chuẩn YDS 2026..."):
            res = process_ecg_dataset(img_pil)

            if use_manual_override:
                if manual_data["hr"] > 0:
                    res["hr"] = int(manual_data["hr"])
                    res["mean_rr"] = 60.0 / res["hr"]
                if manual_data["pr"] > 0:
                    res["pr"] = float(manual_data["pr"])
                if manual_data["qrs"] > 0:
                    res["qrs"] = float(manual_data["qrs"])

            analyzer = ECGClinicalAnalyzer(res, gender=gender_choice, age=age_choice, manual_override=manual_data if use_manual_override else None)
            diag = analyzer.analyze_all()

        if diag["alerts"]:
            for al in diag["alerts"]:
                if "🚨" in al:
                    st.error(al)
                else:
                    st.warning(al)

        st.markdown("#### Chỉ Số Đo Đạc Tự Động (Chuẩn hóa DII)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR - DII)", f"{res['hr']} bpm")
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
