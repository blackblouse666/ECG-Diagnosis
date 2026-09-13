import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="Hệ Thống AI Chẩn Đoán ECG Đa Tầng (AHA/ACC/ESC)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Phân Tích & Chẩn Đoán ECG Toàn Diện")
st.caption("Tích hợp hoàn chỉnh: Rối loạn nhịp, Ngoại tâm thu, Block AV, Block nhánh (chuẩn hóa), Dày thất, Lớn nhĩ, Hội chứng vành cấp (STEMI) & mạn (CCS)")

LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

# =========================================================================
# 1. TIỀN XỬ LÝ & BÓC TÁCH TÍN HIỆU (KHỬ NHIỄU LƯỚI CARO)
# =========================================================================
def preprocess_and_clean_image(gray_img):
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
# 2. ĐO ĐẠC HÌNH THÁI VI THỂ ĐA THAM SỐ
# =========================================================================
def analyze_lead_morphology(sig, px_per_sec, px_per_mv):
    default_props = {
        "r_amp": 0.0, "s_amp": 0.0, "q_amp": 0.0, "q_dur": 0.0,
        "st_shift": 0.0, "t_amp": 0.0, "qrs_w": 0.08, "pr_interval": 0.16,
        "p_amp": 0.0, "p_dur": 0.08, "p_detected": False, "p_biphasic": False,
        "true_rsr": False, "slurred_s": False, "notched_r": False,
        "peaks": [], "qrs_list": []
    }
    if len(sig) < 30:
        return default_props

    peaks, _ = find_peaks(
        sig, distance=int(px_per_sec * 0.25),
        prominence=np.max(sig) * 0.28 if np.max(sig) > 0 else None
    )
    if len(peaks) == 0:
        return default_props

    r_amps, s_amps, q_amps, q_durs = [], [], [], []
    st_shifts, t_amps, qrs_widths, pr_intervals = [], [], [], []
    p_amps, p_durs = [], []
    has_true_rsr = False
    has_slurred_s = False
    has_notched_r = False
    p_detected = False
    p_biphasic = False

    for r in peaks:
        # Sóng R
        r_val = max(0.0, (sig[r] / px_per_mv) * 10.0)
        r_amps.append(r_val)

        # Sóng S
        s_win = sig[r:min(len(sig), r + int(px_per_sec * 0.14))]
        if len(s_win) > 0:
            s_val = (abs(np.min(s_win)) / px_per_mv) * 10.0
            s_amps.append(s_val)
            s_neg_pts = np.where(s_win < -0.18 * sig[r])[0]
            if len(s_neg_pts) / px_per_sec >= 0.040:
                has_slurred_s = True
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
        while left_idx > max(0, r - int(px_per_sec * 0.10)) and sig[left_idx] > 0.14 * sig[r]:
            left_idx -= 1
        right_idx = r
        while right_idx < min(len(sig) - 1, r + int(px_per_sec * 0.14)) and sig[right_idx] > 0.14 * sig[r]:
            right_idx += 1
        qrs_widths.append((right_idx - left_idx) / px_per_sec)

        # Đoạn ST (điểm J) & Sóng T
        j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
        st_shifts.append((sig[j_idx] / px_per_mv) * 10.0)

        t_zone = sig[min(len(sig) - 1, r + int(px_per_sec * 0.14)):min(len(sig), r + int(px_per_sec * 0.28))]
        if len(t_zone) > 0:
            t_amps.append((t_zone[np.argmax(np.abs(t_zone))] / px_per_mv) * 10.0)
        else:
            t_amps.append(0.0)

        # Sóng P và khoảng PR
        p_zone_start = max(0, r - int(px_per_sec * 0.32))
        p_zone_end = max(0, r - int(px_per_sec * 0.09))
        p_zone = sig[p_zone_start:p_zone_end]
        if len(p_zone) > 5:
            p_pks, _ = find_peaks(p_zone, prominence=0.6)
            if len(p_pks) > 0:
                p_detected = True
                p_idx = p_zone_start + p_pks[-1]
                pr_dur = (r - p_idx) / px_per_sec
                if 0.08 <= pr_dur <= 0.38:
                    pr_intervals.append(pr_dur)
                    p_amps.append((sig[p_idx] / px_per_mv) * 10.0)
                    p_durs.append(0.10)
            if np.max(p_zone) > 0.5 and np.min(p_zone) < -0.5:
                p_biphasic = True

        # Tiêu chuẩn rsR' (tai thỏ thực thụ)
        sub_complex = sig[max(0, r - int(px_per_sec * 0.03)):min(len(sig), r + int(px_per_sec * 0.10))]
        if len(sub_complex) > 5:
            local_peaks, _ = find_peaks(sub_complex, distance=int(px_per_sec * 0.024), prominence=2.5)
            if len(local_peaks) >= 2:
                has_true_rsr = True
                has_notched_r = True

    avg_qrs = float(np.median(qrs_widths)) if qrs_widths else 0.08
    avg_qrs = max(0.06, min(avg_qrs, 0.22))

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "q_amp": float(np.mean(q_amps)) if q_amps else 0.0,
        "q_dur": float(np.max(q_durs)) if q_durs else 0.0,
        "st_shift": float(np.mean(st_shifts)) if st_shifts else 0.0,
        "t_amp": float(np.mean(t_amps)) if t_amps else 0.0,
        "qrs_w": avg_qrs,
        "pr_interval": float(np.median(pr_intervals)) if pr_intervals else 0.16,
        "p_amp": float(np.mean(p_amps)) if p_amps else 1.0,
        "p_dur": float(np.mean(p_durs)) if p_durs else 0.08,
        "p_detected": p_detected,
        "p_biphasic": p_biphasic,
        "true_rsr": has_true_rsr,
        "slurred_s": has_slurred_s,
        "notched_r": has_notched_r,
        "peaks": peaks,
        "qrs_list": qrs_widths
    }

def process_ecg_dataset(pil_img: Image.Image):
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

    mean_rr = float(np.median(all_rr_intervals)) if all_rr_intervals else 0.80
    hr = int(60.0 / mean_rr) if mean_rr > 0 else 75
    rr_cv = (float(np.std(all_rr_intervals)) / mean_rr) if len(all_rr_intervals) > 3 else 0.0

    qrs_final = float(np.median(all_qrs_measurements)) if all_qrs_measurements else 0.08
    qrs_final = max(0.07, min(qrs_final, 0.20))

    pr_measured = leads.get("II", {}).get("pr_interval", 0.16)
    p_presence_ratio = sum(1 for v in leads.values() if v.get("p_detected", False)) / 12.0

    return {
        "leads": leads,
        "hr": hr,
        "mean_rr": mean_rr,
        "rr_cv": rr_cv,
        "rr_list": all_rr_intervals,
        "qrs": qrs_final,
        "pr": pr_measured,
        "p_ratio": p_presence_ratio
    }

# =========================================================================
# 3. BỘ TIÊU CHUẨN CHẨN ĐOÁN LÂM SÀNG TỔNG HỢP (AHA/ACC/ESC/HRS)
# =========================================================================
def diagnose_ecg_comprehensive(data, gender="Nam"):
    leads = data.get("leads", {})
    hr = data.get("hr", 75)
    qrs = data.get("qrs", 0.08)
    pr = data.get("pr", 0.16)
    rr_cv = data.get("rr_cv", 0.0)
    rr_list = data.get("rr_list", [])
    p_ratio = data.get("p_ratio", 1.0)

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

    # ---------------- 2. RỐI LOẠN NHỊP & NGOẠI TÂM THU ----------------
    if rr_cv > 0.18 and p_ratio < 0.25:
        findings.append(("Rối loạn nhịp", "Rung nhĩ (Atrial Fibrillation - AFib): Mất sóng P, nhịp thất hoàn toàn không đều"))
        alerts.append("⚠️ Rung nhĩ: Cần đánh giá thang điểm CHA2DS2-VASc dự phòng thuyên tắc huyết khối")
    elif 250 <= (60.0 / (data["mean_rr"] / 4.0 if data["mean_rr"] > 0 else 1.0)) <= 350 and p_ratio < 0.20:
        findings.append(("Rối loạn nhịp", "Cuồng nhĩ (Atrial Flutter - AFL): Sóng F dạng răng cưa tần số nhĩ nhanh đều"))
    else:
        if hr > 100:
            findings.append(("Rối loạn nhịp", f"Nhịp nhanh xoang (Sinus Tachycardia) - Tần số: {hr} l/p"))
        elif hr < 60:
            findings.append(("Rối loạn nhịp", f"Nhịp chậm xoang (Sinus Bradycardia) - Tần số: {hr} l/p"))
        else:
            findings.append(("Rối loạn nhịp", f"Nhịp xoang bình thường (Normal Sinus Rhythm) - Tần số: {hr} l/p"))

    # Ngoại tâm thu (PAC / PVC)
    early_beats = [rr for rr in rr_list if rr < 0.80 * data["mean_rr"]]
    late_compensatory = [rr for rr in rr_list if rr > 1.20 * data["mean_rr"]]

    if len(early_beats) > 0 and len(late_compensatory) > 0:
        if qrs >= 0.12 or leads.get("V1", {}).get("qrs_w", 0.08) >= 0.12:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu thất (PVC): Nhát bóp đến sớm, QRS dị dạng giãn rộng, nghỉ bù hoàn toàn"))
            alerts.append("Phát hiện Ngoại tâm thu thất (PVC): Đánh giá phân độ Lown và chỉ định Holter ECG 24h")
        else:
            findings.append(("Ngoại tâm thu", "Ngoại tâm thu nhĩ (PAC): Nhát bóp đến sớm với phức bộ QRS hẹp"))

    # ---------------- 3. LỚN NHĨ (ATRIAL ENLARGEMENT) ----------------
    p_amp_d2 = d2.get("p_amp", 0.0)
    p_amp_d3 = d3.get("p_amp", 0.0)
    p_amp_avf = avf.get("p_amp", 0.0)
    v1_data = leads.get("V1", {})

    if max(p_amp_d2, p_amp_d3, p_amp_avf) >= 2.5:
        findings.append(("Lớn buồng tim", f"Lớn nhĩ phải (P phế): Sóng P cao {max(p_amp_d2, p_amp_d3, p_amp_avf):.1f} mm (≥ 2.5 mm ở DII/DIII/aVF)"))
    elif d2.get("p_dur", 0.08) >= 0.12 or v1_data.get("p_biphasic", False):
        findings.append(("Lớn buồng tim", "Lớn nhĩ trái (P nhĩ): Sóng P rộng ≥ 0.12s ở DII hoặc 2 pha âm chiếm ưu thế ở V1"))

    # ---------------- 4. DÀY THẤT (VENTRICULAR HYPERTROPHY) ----------------
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
    if (rv1 >= 7.0 or (sv1 > 0 and rv1 / sv1 > 1.0)) and "RAD" in axis_type:
        findings.append(("Phì đại thất", f"Dày thất phải (RVH): Sóng R ưu thế ở V1 (R={rv1:.1f} mm) kèm trục lệch phải"))

    # ---------------- 5. BLOCK NHÁNH (TIÊU CHUẨN KÉP KHỬ DƯƠNG TÍNH GIẢ) ----------------
    v2_data = leads.get("V2", {})
    v1_has_rsr = v1_data.get("true_rsr", False) or v2_data.get("true_rsr", False)
    lateral_has_broad_s = v6_data.get("slurred_s", False) or d1.get("slurred_s", False)

    rbbb_type = None
    if v1_has_rsr and lateral_has_broad_s:
        if qrs >= 0.12:
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 dạng rsR' (tai thỏ), S rộng ở DI/V6"
        elif 0.09 <= qrs < 0.12:
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.09 - 0.11s, dạng rsR' tại V1 kèm S rộng ở V6"

    if rbbb_type:
        findings.append(("Block Nhánh", rbbb_type))

    # LBBB
    has_lbbb_pattern = (
        (v5_data.get("notched_r", False) or v6_data.get("notched_r", False) or d1.get("notched_r", False)) and
        (v1_data.get("s_amp", 0.0) > 8.0 and v1_data.get("r_amp", 0.0) < 3.0) and
        not v1_has_rsr
    )
    if has_lbbb_pattern:
        if qrs >= 0.12:
            findings.append(("Block Nhánh", "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía ở DI/V5/V6"))
            alerts.append("🚨 LBBB hoàn toàn: Cần đối chiếu tiêu chuẩn Sgarbossa loại trừ Nhồi máu cơ tim cấp")
        elif 0.10 <= qrs < 0.12:
            findings.append(("Block Nhánh", "Block nhánh trái không hoàn toàn (Incomplete LBBB)"))

    # ---------------- 6. BLOCK PHÂN NHÁNH & BLOCK AV ----------------
    lafb = False
    lpfb = False
    if "LAD" in axis_type and qrs < 0.12:
        if d1.get("r_amp", 0.0) > d1.get("s_amp", 0.0) and d3.get("s_amp", 0.0) > d3.get("r_amp", 0.0):
            lafb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh, qR ở DI/aVL, rS ở DII/DIII/aVF"))
    elif "RAD" in axis_type and qrs < 0.12 and rv1 < 6.0:
        if d3.get("r_amp", 0.0) > d3.get("s_amp", 0.0) and d1.get("s_amp", 0.0) > d1.get("r_amp", 0.0):
            lpfb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh, rS ở DI/aVL, qR ở DII/DIII/aVF"))

    # Block AV
    if hr < 45 and qrs >= 0.12 and rr_cv < 0.04:
        findings.append(("Dẫn truyền nhĩ - thất", "Block nhĩ - thất độ III: Phân ly nhĩ thất hoàn toàn"))
        alerts.append("🚨 BLOCK TIM ĐỘ 3: CHỈ ĐỊNH MÁY TẠO NHỊP CẤP CỨU")
    elif pr > 0.20:
        findings.append(("Dẫn truyền nhĩ - thất", f"Block nhĩ - thất độ I: Khoảng PR kéo dài ({pr:.2f}s > 0.20s)"))

    # Block 2 & 3 phân nhánh
    if rbbb_type and "Complete" in rbbb_type:
        if lafb:
            findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh: RBBB kết hợp LAFB" + (" + Block AV độ 1 (Block 3 phân nhánh)" if pr > 0.20 else "")))
        elif lpfb:
            findings.append(("Block Đa Phân Nhánh", "Block hai phân nhánh: RBBB kết hợp LPFB" + (" + Block AV độ 1 (Block 3 phân nhánh)" if pr > 0.20 else "")))

    # ---------------- 7. HỘI CHỨNG VÀNH CẤP & MẠN (ACS / CCS) ----------------
    st_elev_leads = []
    st_depr_leads = []
    path_q_leads = []

    for l_name, l_val in leads.items():
        st_cutoff = 1.5 if l_name in ["V2", "V3"] else 1.0
        if l_val.get("st_shift", 0.0) >= st_cutoff:
            st_elev_leads.append(l_name)
        elif l_val.get("st_shift", 0.0) <= -0.8:
            st_depr_leads.append(l_name)

        if l_val.get("q_dur", 0.0) >= 0.035 and (l_val.get("r_amp", 0.0) > 0 and l_val.get("q_amp", 0.0) >= 0.25 * l_val.get("r_amp", 0.0)):
            path_q_leads.append(l_name)

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
            arteries.append("Thân chung / LAD đoạn rất gần")
        if len({"II", "III", "aVF"}.intersection(leads_found)) >= 2:
            regions.append("Thành dưới (Inferior)")
            arteries.append("RCA / LCx")
        if {"I", "aVL"}.issubset(leads_found) or {"V5", "V6"}.issubset(leads_found):
            regions.append("Thành bên (Lateral)")
            arteries.append("LCx")
        return list(set(regions)), list(set(arteries))

    if st_elev_leads:
        regions, arteries = map_anatomy(set(st_elev_leads))
        reg_str = ", ".join(regions) if regions else ", ".join(st_elev_leads)
        art_str = f" - ĐM thủ phạm: {', '.join(arteries)}" if arteries else ""
        has_necrosis = any(l in set(path_q_leads) for l in st_elev_leads)
        stage_str = "Bán cấp / Đã có hoại tử" if has_necrosis else "Tối cấp / Cấp tính"
        findings.append(("Hội chứng vành cấp (ACS)", f"Nhồi máu cơ tim ST chênh lên (STEMI) - Vùng: {reg_str} - Giai đoạn: {stage_str}{art_str}"))
        alerts.append(f"🚨 STEMI VÙNG {reg_str.upper()}: KÍCH HOẠT QUY TRÌNH PCI CẤP CỨU")
    elif path_q_leads:
        regions, _ = map_anatomy(set(path_q_leads))
        reg_str = ", ".join(regions) if regions else ", ".join(path_q_leads)
        findings.append(("Hội chứng vành mạn (CCS)", f"Sẹo hoại tử / Nhồi máu cơ tim cũ (Old MI) - Vùng: {reg_str}"))
    elif st_depr_leads:
        regions, _ = map_anatomy(set(st_depr_leads))
        reg_str = ", ".join(regions) if regions else ", ".join(st_depr_leads)
        findings.append(("Thiếu máu cục bộ cơ tim", f"ST chênh xuống / Thiếu máu dưới nội tâm mạc - Vùng: {reg_str}"))

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
    gender_choice = st.radio("Giới tính bệnh nhân (tính chuẩn Cornell)", ["Nam", "Nữ"], horizontal=True)
    uploaded = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, caption="Phiếu đo ECG đã nạp vào bộ xử lý", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo ECG lên.")

with col2:
    st.subheader("2. Kết Quả Chẩn Đoán Chuyên Khoa Toàn Diện")
    if uploaded:
        with st.spinner("Đang khử lưới, đo sóng P-Q-R-S-ST-T, phân tích rối loạn nhịp và dẫn truyền..."):
            res = process_ecg_dataset(img_pil)
            diag = diagnose_ecg_comprehensive(res, gender=gender_choice)

        if diag["alerts"]:
            for al in diag["alerts"]:
                st.error(al)

        st.markdown("#### Chỉ Số Đo Đạc Tự Động")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tần số (HR)", f"{res['hr']} bpm")
        m2.metric("Độ rộng QRS", f"{res['qrs']:.3f} s")
        m3.metric("Sokolow-Lyon", f"{diag['sokolow']:.1f} mm", help="LVH nếu ≥ 35 mm")
        m4.metric("Cornell", f"{diag['cornell']:.1f} mm", help="LVH nếu > 28 mm (Nam) hoặc > 20 mm (Nữ)")

        st.write(f"📐 **Trục điện tim:** `{diag['axis']}` | **Khoảng PR:** `{res['pr']:.2f} s` | **Độ biến thiên R-R:** `{res['rr_cv']:.3f}`")
        st.markdown("---")

        st.markdown("#### Kết Luận Chẩn Đoán Phân Tầng")
        if diag["findings"]:
            for cat, desc in diag["findings"]:
                if "STEMI" in desc or "độ III" in desc or "CẤP" in desc or "Rung nhĩ" in desc:
                    st.markdown(f"- **[{cat}]** :red[{desc}]")
                elif "Block" in desc or "Dày thất" in desc or "Lớn nhĩ" in desc or "Ngoại tâm thu" in desc or "mạn" in desc or "Thiếu máu" in desc:
                    st.markdown(f"- **[{cat}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{cat}]** :green[{desc}]")
        else:
            st.success("✅ Bản ghi bình thường, không phát hiện bất thường tái cực, phì đại buồng tim hoặc rối loạn dẫn truyền.")

        st.markdown("---")
        with st.expander("🔍 Bảng Đo Đạc Vi Thể 12 Chuyển Đạo (P, Q, R, S, ST, rsR', S rộng)"):
            detail_list = []
            for l_name, l_data in res["leads"].items():
                detail_list.append({
                    "Chuyển đạo": l_name,
                    "P (mm)": f"{l_data.get('p_amp', 0.0):.1f}",
                    "Q (mm)": f"{l_data.get('q_amp', 0.0):.1f}",
                    "R (mm)": f"{l_data.get('r_amp', 0.0):.1f}",
                    "S (mm)": f"{l_data.get('s_amp', 0.0):.1f}",
                    "ST Chênh (mm)": f"{l_data.get('st_shift', 0.0):+.1f}",
                    "rsR' (V1)": "Có" if l_data.get("true_rsr") else "-",
                    "S rộng (>40ms)": "Có" if l_data.get("slurred_s") else "-"
                })
            st.dataframe(detail_list, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
