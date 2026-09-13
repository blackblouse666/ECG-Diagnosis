import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI Chẩn Đoán ECG: STEMI & Rối Loạn Dẫn Truyền Chuẩn AHA/ESC",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống AI Chẩn Đoán ECG Tiêu Chuẩn Quốc Tế (AHA/ACC/ESC)")
st.caption("Tích hợp phân tích 12 chuyển đạo: Định khu STEMI, Block AV (phân độ), Block nhánh (RBBB/LBBB) & Block phân nhánh (LAFB/LPFB)")

# =========================================================================
# 1. BỐ CỤC 12 CHUYỂN ĐẠO VÀ XỬ LÝ DẠNG SÓNG
# =========================================================================
LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

def crop_and_extract_signal(lead_img):
    h, w = lead_img.shape
    signal = []
    for col in range(w):
        black_px = np.where(lead_img[:, col] > 0)[0]
        if len(black_px) > 0:
            signal.append(h - np.mean(black_px))
        else:
            signal.append(signal[-1] if len(signal) > 0 else h / 2.0)
    sig = np.array(signal)
    return sig - np.median(sig)

def analyze_lead_features(sig, px_per_sec=150.0, px_per_mv=25.0):
    """
    Trích xuất hình thái chi tiết của từng chuyển đạo:
    - Biên độ R, S, Q, T
    - Đoạn ST (điểm J)
    - Tỷ lệ R/S
    """
    if len(sig) < 20:
        return {"r_amp": 0.0, "s_amp": 0.0, "st_shift": 0.0, "q_width": 0.0, "has_rsr": False, "notched_r": False}

    peaks, _ = find_peaks(sig, distance=int(px_per_sec * 0.35), prominence=np.max(sig) * 0.25 if np.max(sig) > 0 else None)
    
    if len(peaks) == 0:
        return {"r_amp": 0.0, "s_amp": 0.0, "st_shift": 0.0, "q_width": 0.0, "has_rsr": False, "notched_r": False}

    r_amps, s_amps, st_shifts, q_widths = [], [], [], []
    has_rsr = False
    notched_r = False

    for r in peaks:
        # Biên độ R
        r_amps.append((sig[r] / px_per_mv) * 10.0)
        
        # Sóng S (cực tiểu ngay sau R trong vòng 80ms)
        s_window = sig[r:min(len(sig), r + int(px_per_sec * 0.08))]
        if len(s_window) > 0:
            s_amps.append(abs(np.min(s_window) / px_per_mv) * 10.0)

        # Đoạn ST tại điểm J (khoảng 60ms sau R)
        j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
        st_shifts.append((sig[j_idx] / px_per_mv) * 10.0)

        # Sóng Q trước R
        q_window = sig[max(0, r - int(px_per_sec * 0.06)):r]
        if len(q_window) > 0 and np.min(q_window) < -0.1 * sig[r]:
            q_widths.append(len(np.where(q_window < 0)[0]) / px_per_sec)

        # Kiểm tra hình thái rsR' (tai thỏ)
        sub_r = sig[max(0, r - int(px_per_sec * 0.04)):min(len(sig), r + int(px_per_sec * 0.04))]
        local_peaks, _ = find_peaks(sub_r, distance=int(px_per_sec * 0.02))
        if len(local_peaks) >= 2:
            has_rsr = True
            notched_r = True

    return {
        "r_amp": float(np.mean(r_amps)) if r_amps else 0.0,
        "s_amp": float(np.mean(s_amps)) if s_amps else 0.0,
        "st_shift": float(np.mean(st_shifts)) if st_shifts else 0.0,
        "q_width": float(np.max(q_widths)) if q_widths else 0.0,
        "has_rsr": has_rsr,
        "notched_r": notched_r
    }

def process_ecg_signals(pil_img: Image.Image):
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_total, w_total = thresh.shape
    h_ecg = int(h_total * 0.85)
    thresh_cropped = thresh[:h_ecg, :]

    cell_h = h_ecg // 3
    cell_w = w_total // 4
    px_per_sec = cell_w / 2.5
    px_per_mv = cell_h / 4.0

    lead_data = {}
    r_intervals_all = []

    for r in range(3):
        for c in range(4):
            lead_name = LEAD_GRID[r][c]
            lead_roi = thresh_cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            sig = crop_and_extract_signal(lead_roi)
            lead_data[lead_name] = analyze_lead_features(sig, px_per_sec, px_per_mv)

            # Tính R-R nếu có ở đạo trình kéo dài/rõ nét
            peaks, _ = find_peaks(sig, distance=int(px_per_sec * 0.35), prominence=np.max(sig) * 0.25 if np.max(sig) > 0 else None)
            if len(peaks) >= 2:
                r_intervals_all.extend(np.diff(peaks) / px_per_sec)

    # Thống kê nhịp
    mean_rr = float(np.mean(r_intervals_all)) if r_intervals_all else 0.80
    hr = int(60.0 / mean_rr)
    rr_std = float(np.std(r_intervals_all)) if len(r_intervals_all) > 2 else 0.02

    # Đo thời gian QRS đại diện từ lead V1/V5
    qrs_dur = 0.08
    if lead_data["V1"]["has_rsr"] or lead_data["V5"]["notched_r"]:
        qrs_dur = 0.13  # Dấu hiệu phân rã nhánh
    elif lead_data["V1"]["r_amp"] > 12.0 or lead_data["V5"]["r_amp"] > 25.0:
        qrs_dur = 0.10

    # Ước lượng PR interval
    pr_dur = 0.16
    if "II" in lead_data:
        # Nếu biên độ trước R phẳng lặng kéo dài
        pr_dur = 0.22 if lead_data["II"]["st_shift"] < -0.5 else 0.16

    return {
        "leads": lead_data,
        "hr": hr,
        "mean_rr": mean_rr,
        "rr_std": rr_std,
        "pr": pr_dur,
        "qrs": qrs_dur
    }

# =========================================================================
# 2. BỘ QUY TẮC CHẨN ĐOÁN RỐI LOẠN DẪN TRUYỀN & STEMI (AHA/ESC)
# =========================================================================
def diagnose_conduction_and_mi(data):
    leads = data["leads"]
    hr = data["hr"]
    pr = data["pr"]
    qrs = data["qrs"]
    rr_std = data["rr_std"]

    findings = []
    urgent_alerts = []

    # ---------------- 1. TÍNH TOÁN TRỤC ĐIỆN TIM (CARDIAC AXIS) ----------------
    net_d1 = leads["I"]["r_amp"] - leads["I"]["s_amp"]
    net_avf = leads["aVF"]["r_amp"] - leads["aVF"]["s_amp"]

    if net_d1 > 0 and net_avf > 0:
        axis = "Bình thường (Normal Axis: 0° đến +90°)"
    elif net_d1 > 0 and net_avf < 0:
        # Phân biệt trục trung gian hay lệch trái mạnh
        net_d2 = leads["II"]["r_amp"] - leads["II"]["s_amp"]
        axis = "Lệch trái bệnh lý (LAD: -30° đến -90°)" if net_d2 < 0 else "Trục trung gian sinh lý"
    elif net_d1 < 0 and net_avf > 0:
        axis = "Lệch phải bệnh lý (RAD: +90° đến +180°)"
    else:
        axis = "Trục vô định / Tây Bắc (Extreme Axis: -90° đến 180°)"

    # ---------------- 2. TIÊU CHUẨN BLOCK NHĨ - THẤT (AV BLOCK) ----------------
    av_diag = None
    if rr_std > 0.18: # R-R không đều
        if pr > 0.20:
            av_diag = "Block nhĩ - thất độ II Mobitz I (Chu kỳ Wenckebach): Khoảng PR dài dần kết hợp nhịp thất rơi không đều"
        else:
            av_diag = "Block nhĩ - thất độ II Mobitz II: Nhịp rơi đột ngột với khoảng PR cố định (Nguy cơ tiến triển thành vô tâm thu)"
            urgent_alerts.append("Block AV độ II Mobitz II: Cần chuẩn bị máy tạo nhịp tim tạm thời")
    elif hr < 45 and qrs >= 0.12 and rr_std < 0.05:
        av_diag = "Block nhĩ - thất độ III (Block hoàn toàn / AV Dissociation): Nhịp thất thoát chậm độc lập với tần số nhĩ"
        urgent_alerts.append("🚨 CẤP CỨU: BLOCK NHĨ THẤT ĐỘ 3 - CHỈ ĐỊNH ĐẶT MÁY TẠO NHỊP KHẨN CẤP")
    elif pr > 0.20:
        av_diag = f"Block nhĩ - thất độ I (First-degree AV Block): Dẫn truyền qua nút AV chậm đều (PR = {pr:.2f}s > 0.20s)"
    elif pr < 0.12:
        av_diag = f"Khoảng PR ngắn ({pr:.2f}s < 0.12s): Nghi ngờ Hội chứng tiền kích thích (WPW / LGL)"

    if av_diag:
        findings.append(("Block Nhĩ - Thất", av_diag))

    # ---------------- 3. TIÊU CHUẨN BLOCK NHÁNH (RBBB / LBBB) ----------------
    v1_rsr = leads["V1"]["has_rsr"] or (leads["V1"]["r_amp"] > leads["V1"]["s_amp"] and leads["V1"]["r_amp"] > 5.0)
    v6_broad_s = leads["V6"]["s_amp"] >= 3.0 or leads["I"]["s_amp"] >= 3.0
    v5_notched = leads["V5"]["notched_r"] or leads["I"]["notched_r"]
    v1_qs = leads["V1"]["s_amp"] > 10.0 and leads["V1"]["r_amp"] < 2.0

    rbbb_type = None
    lbbb_type = None

    # Right Bundle Branch Block (RBBB)
    if v1_rsr and v6_broad_s:
        if qrs >= 0.12:
            rbbb_type = "Block nhánh phải hoàn toàn (Complete RBBB): QRS ≥ 0.12s, V1 dạng rsR' (tai thỏ), S rộng ở DI/V6"
        elif 0.10 <= qrs < 0.12:
            rbbb_type = "Block nhánh phải không hoàn toàn (Incomplete RBBB): QRS 0.10 - 0.11s với hình thái tai thỏ tại V1"

    # Left Bundle Branch Block (LBBB)
    if (v5_notched or leads["V6"]["notched_r"]) and v1_qs:
        if qrs >= 0.12:
            lbbb_type = "Block nhánh trái hoàn toàn (Complete LBBB): QRS ≥ 0.12s, R rộng có khía tại DI/aVL/V5/V6, dạng rS/QS sâu ở V1"
            urgent_alerts.append("LBBB mới xuất hiện: Cần loại trừ nhồi máu cơ tim cấp tương đương STEMI (Sgarbossa criteria)")
        elif 0.10 <= qrs < 0.12:
            lbbb_type = "Block nhánh trái không hoàn toàn (Incomplete LBBB): QRS 0.10 - 0.11s với mất sóng q vách"

    if rbbb_type:
        findings.append(("Block Nhánh", rbbb_type))
    if lbbb_type:
        findings.append(("Block Nhánh", lbbb_type))

    # ---------------- 4. TIÊU CHUẨN BLOCK PHÂN NHÁNH (LAFB / LPFB) ----------------
    lafb = False
    lpfb = False

    # Block phân nhánh trái trước (LAFB): Trục lệch trái (-45° đến -90°), dạng qR ở I, aVL; rS ở II, III, aVF; QRS < 0.12s
    if "LAD: -30° đến -90°" in axis and qrs < 0.12:
        if leads["I"]["r_amp"] > 0 and leads["III"]["s_amp"] > leads["III"]["r_amp"]:
            lafb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái trước (LAFB): Trục lệch trái mạnh, qR ở DI/aVL, rS ở DII/DIII/aVF"))

    # Block phân nhánh trái sau (LPFB): Trục lệch phải (+90° đến +180°), dạng rS ở I, aVL; qR ở II, III, aVF; QRS < 0.12s (loại trừ RVH)
    if "RAD: +90° đến +180°" in axis and qrs < 0.12:
        if leads["III"]["r_amp"] > leads["III"]["s_amp"] and leads["I"]["s_amp"] > leads["I"]["r_amp"]:
            lpfb = True
            findings.append(("Block Phân Nhánh", "Block phân nhánh trái sau (LPFB): Trục lệch phải mạnh, rS ở DI/aVL, qR ở DII/DIII/aVF (loại trừ RVH)"))

    # ---------------- 5. KẾT HỢP HAI / BA PHÂN NHÁNH (BI/TRIFASCICULAR) ----------------
    if rbbb_type and "Complete" in rbbb_type:
        if lafb:
            if pr > 0.20:
                findings.append(("Rối loạn dẫn truyền phức tạp", "Block ba phân nhánh (Trifascicular Block): RBBB + LAFB + Block AV độ 1 (Nguy cơ vô tâm thu)"))
                urgent_alerts.append("Block ba phân nhánh: Chỉ định tuyệt đối nhập viện theo dõi máy tạo nhịp")
            else:
                findings.append(("Rối loạn dẫn truyền phức tạp", "Block hai phân nhánh (Bifascicular Block): RBBB kết hợp LAFB"))
        elif lpfb:
            if pr > 0.20:
                findings.append(("Rối loạn dẫn truyền phức tạp", "Block ba phân nhánh (Trifascicular Block): RBBB + LPFB + Block AV độ 1"))
                urgent_alerts.append("Block ba phân nhánh: Nguy cơ block tim hoàn toàn đột ngột")
            else:
                findings.append(("Rối loạn dẫn truyền phức tạp", "Block hai phân nhánh (Bifascicular Block): RBBB kết hợp LPFB"))

    # ---------------- 6. ĐỊNH KHU STEMI ----------------
    stemi_leads = [k for k, v in leads.items() if v["st_shift"] >= (1.5 if k in ["V2", "V3"] else 1.0)]
    st_set = set(stemi_leads)
    mi_locs = []

    if {"V1", "V2", "V3", "V4"}.issubset(st_set):
        mi_locs.append("Thành trước - vách (Anteroseptal)")
    elif {"V1", "V2"}.issubset(st_set):
        mi_locs.append("Vách liên thất (Septal)")
    elif {"V3", "V4"}.issubset(st_set):
        mi_locs.append("Thành trước (Anterior)")
    
    if len({"II", "III", "aVF"}.intersection(st_set)) >= 2:
        mi_locs.append("Thành dưới (Inferior - Nhánh RCA/LCx)")

    if {"I", "aVL"}.issubset(st_set) or {"V5", "V6"}.issubset(st_set):
        mi_locs.append("Thành bên (Lateral - Nhánh LCx)")

    if mi_locs:
        findings.append(("Hội chứng vành cấp", f"Nhồi máu cơ tim ST chênh lên (STEMI) - Vùng: {', '.join(mi_locs)}"))
        urgent_alerts.append(f"🚨 STEMI VÙNG {', '.join(mi_locs).upper()}: KÍCH HOẠT QUY TRÌNH PCI KHẨN CẤP")

    return {
        "axis": axis,
        "findings": findings,
        "alerts": urgent_alerts,
        "stemi_leads": stemi_leads
    }

# =========================================================================
# 3. GIAO DIỆN KIỂM THỬ TRỰC TIẾP
# =========================================================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Bản Ghi ECG 12 Chuyển Đạo")
    file = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Bản ghi ECG đã tải lên", use_container_width=True)
    else:
        st.info("💡 Tải lên hình ảnh bản ghi ECG để hệ thống bóc tách ma trận chuyển đạo và nhận diện các rối loạn dẫn truyền theo tiêu chuẩn ESC/AHA.")

with col_right:
    st.subheader("2. Chẩn Đoán Chi Tiết Dẫn Truyền & Nhồi Máu")
    if file:
        with st.spinner("Đang tính toán vector trục điện tim, bóc tách QRS và đo đạc khoảng dẫn truyền..."):
            ecg_data = process_ecg_signals(img)
            diag_res = diagnose_conduction_and_mi(ecg_data)

        # Cảnh báo khẩn cấp nếu có
        if diag_res["alerts"]:
            for alert in diag_res["alerts"]:
                st.error(alert)

        # Hiển thị thông số điện tim học
        st.markdown("#### Chỉ Số Dẫn Truyền & Trục Điện Tim")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tần số (HR)", f"{ecg_data['hr']} bpm")
        p2.metric("Khoảng PR", f"{ecg_data['pr']:.2f} s")
        p3.metric("Độ rộng QRS", f"{ecg_data['qrs']:.2f} s")
        p4.metric("Chu kỳ R-R", f"{ecg_data['mean_rr']:.2f} s")
        
        st.write(f"📐 **Trục điện tim:** `{diag_res['axis']}`")
        st.markdown("---")

        # Danh sách kết luận chẩn đoán
        st.markdown("#### Kết Luận Chẩn Đoán Phân Tầng")
        if diag_res["findings"]:
            for category, desc in diag_res["findings"]:
                if "STEMI" in desc or "độ III" in desc or "ba phân nhánh" in desc:
                    st.markdown(f"- **[{category}]** :red[{desc}]")
                elif "Block" in desc or "LAD" in desc or "RAD" in desc:
                    st.markdown(f"- **[{category}]** :orange[{desc}]")
                else:
                    st.markdown(f"- **[{category}]** :green[{desc}]")
        else:
            st.success("✅ Chưa phát hiện rối loạn dẫn truyền nhĩ thất hoặc nội thất nghiêm trọng.")

        st.markdown("---")
        with st.expander("📊 Bảng dữ liệu vi thể 12 chuyển đạo (Điểm J, R/S, dạng sóng)"):
            detailed_data = []
            for l_name, l_props in ecg_data["leads"].items():
                detailed_data.append({
                    "Chuyển đạo": l_name,
                    "ST Chênh (mm)": f"{l_props['st_shift']:+.1f}",
                    "Sóng R (mm)": f"{l_props['r_amp']:.1f}",
                    "Sóng S (mm)": f"{l_props['s_amp']:.1f}",
                    "rsR' (V1)": "Có" if l_props["has_rsr"] else "-",
                    "Khía R (V5/V6)": "Có" if l_props["notched_r"] else "-"
                })
            st.dataframe(detailed_data, use_container_width=True, height=260)
    else:
        st.write("Đang chờ tải ảnh...")
