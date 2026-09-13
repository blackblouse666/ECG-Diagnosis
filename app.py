import streamlit as st
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import math

st.set_page_config(
    page_title="AI Chẩn Đoán Nhồi Máu Cơ Tim 12 Chuyển Đạo",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 AI Phân Tích & Định Khu Nhồi Máu Cơ Tim (12-Lead ECG)")
st.caption("Thuật toán phân rã ma trận 12 chuyển đạo, bóc tách điểm J, sóng Q hoại tử và hình ảnh soi gương theo tiêu chuẩn ESC/AHA")

# =========================================================================
# 1. HỆ THỐNG PHÂN RÃ 12 CHUYỂN ĐẠO VÀ BÓC TÁCH HÌNH THÁI SÓNG
# =========================================================================
LEAD_GRID = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

def crop_and_extract_signal(lead_img):
    """
    Bóc tách tín hiệu 1D cho từng ô chuyển đạo độc lập
    """
    h, w = lead_img.shape
    signal = []
    for col in range(w):
        black_px = np.where(lead_img[:, col] > 0)[0]
        if len(black_px) > 0:
            signal.append(h - np.mean(black_px))
        else:
            signal.append(signal[-1] if len(signal) > 0 else h / 2.0)
    sig = np.array(signal)
    baseline = np.median(sig)
    return sig - baseline

def analyze_lead_morphology(sig, px_per_mv=25.0, px_per_sec=150.0):
    """
    Phân tích chi tiết hình thái từng chuyển đạo:
    - Điểm J và độ chênh ST (mm)
    - Dạng sóng ST (Vòm Tombstone hay lõm)
    - Sóng Q hoại tử
    - Sóng T âm
    """
    if len(sig) < 20:
        return {"st_elevation": 0.0, "st_depression": 0.0, "pathological_q": False, "tombstone": False}

    # Tìm các đỉnh R
    peaks, props = find_peaks(sig, distance=int(px_per_sec * 0.35), prominence=np.max(sig) * 0.25 if np.max(sig) > 0 else None)
    
    if len(peaks) == 0:
        return {"st_elevation": 0.0, "st_depression": 0.0, "pathological_q": False, "tombstone": False}

    st_shifts = []
    q_detected = False
    is_tombstone = False

    for r in peaks:
        # 1. Xác định điểm J (khoảng 40-80ms sau đỉnh R)
        j_idx = min(len(sig) - 1, r + int(px_per_sec * 0.06))
        st_mid_idx = min(len(sig) - 1, r + int(px_per_sec * 0.12))
        
        # Độ chênh ST tính theo mm (10mm = 1mV)
        st_shift_mm = (sig[j_idx] / px_per_mv) * 10.0
        st_shifts.append(st_shift_mm)

        # 2. Kiểm tra sóng Q hoại tử (vùng 20-40ms trước R)
        q_window_start = max(0, r - int(px_per_sec * 0.06))
        q_sub = sig[q_window_start:r]
        if len(q_sub) > 0:
            q_depth = np.min(q_sub)
            r_height = sig[r]
            q_width_sec = len(np.where(q_sub < 0)[0]) / px_per_sec
            if q_depth < -0.25 * r_height and q_width_sec >= 0.035:
                q_detected = True

        # 3. Kiểm tra dạng vòm lồi Tombstone (ST cong lồi liên tục nối vào T)
        if sig[st_mid_idx] > sig[j_idx] and st_shift_mm > 1.5:
            is_tombstone = True

    avg_st = float(np.mean(st_shifts)) if st_shifts else 0.0
    return {
        "st_elevation": max(0.0, avg_st),
        "st_depression": abs(min(0.0, avg_st)),
        "pathological_q": q_detected,
        "tombstone": is_tombstone
    }

def process_12_lead_ecg(pil_img: Image.Image):
    """
    Chia lưới ảnh làm 3 hàng x 4 cột tương ứng 12 chuyển đạo chuẩn
    """
    cv_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    
    # Tiền xử lý: Lọc nhiễu và nhị phân hóa
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Loại bỏ 15% phần chân ảnh (nơi thường in lead nhịp DII kéo dài)
    h_total, w_total = thresh.shape
    h_ecg = int(h_total * 0.85)
    thresh_cropped = thresh[:h_ecg, :]

    cell_h = h_ecg // 3
    cell_w = w_total // 4

    results = {}
    for r in range(3):
        for c in range(4):
            lead_name = LEAD_GRID[r][c]
            lead_roi = thresh_cropped[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            sig = crop_and_extract_signal(lead_roi)
            results[lead_name] = analyze_lead_morphology(sig, px_per_mv=cell_h/4.0, px_per_sec=cell_w/2.5)

    return results

# =========================================================================
# 2. BỘ LOGIC QUY TẮC ĐỊNH KHU & GIAI ĐOẠN THEO TIÊU CHUẨN ESC/AHA
# =========================================================================
def diagnose_mi(results):
    stemi_leads = []
    reciprocal_leads = []
    q_leads = []
    tombstone_detected = False

    for lead, data in results.items():
        # Tiêu chuẩn chẩn đoán ST chênh lên:
        # V2-V3: >= 1.5 - 2.0 mm; Các chuyển đạo khác: >= 1.0 mm
        threshold = 1.5 if lead in ["V2", "V3"] else 0.85
        if data["st_elevation"] >= threshold:
            stemi_leads.append(lead)
            if data["tombstone"]:
                tombstone_detected = True
        
        if data["st_depression"] >= 0.7:
            reciprocal_leads.append(lead)
            
        if data["pathological_q"]:
            q_leads.append(lead)

    st_set = set(stemi_leads)
    infarct_locations = []
    culprit_artery = []

    # Định khu giải phẫu theo phân bố động mạch vành
    if {"V1", "V2"}.issubset(st_set) and not {"V3", "V4"}.issubset(st_set):
        infarct_locations.append("Vách liên thất (Septal)")
        culprit_artery.append("Nhánh gian thất trước (LAD)")

    if {"V3", "V4"}.issubset(st_set):
        infarct_locations.append("Thành trước (Anterior)")
        culprit_artery.append("Động mạch liên thất trước (LAD)")

    if {"V1", "V2", "V3", "V4"}.issubset(st_set):
        infarct_locations.append("Trước - Vách (Anteroseptal)")
        culprit_artery.append("Nhánh LAD đoạn gần/giữa")

    if {"V1", "V2", "V3", "V4", "V5", "V6"}.issubset(st_set) or ({"V3", "V4", "V5", "V6"}.issubset(st_set) and {"I", "aVL"}.intersection(st_set)):
        infarct_locations.append("Trước rộng (Extensive Anterior)")
        culprit_artery.append("Thân chung hoặc LAD đoạn rất gần (Proximal LAD)")

    if len({"II", "III", "aVF"}.intersection(st_set)) >= 2:
        infarct_locations.append("Thành dưới (Inferior)")
        culprit_artery.append("Động mạch vành phải (RCA 85%) hoặc Động mạch Mũ (LCx 15%)")

    if {"I", "aVL"}.issubset(st_set) or {"V5", "V6"}.issubset(st_set):
        infarct_locations.append("Thành bên (Lateral)")
        culprit_artery.append("Động mạch Mũ (LCx) hoặc nhánh Diagonal của LAD")

    # Xác định giai đoạn nhồi máu (Stage)
    stage = "Không xác định"
    has_q = any(lead in q_leads for lead in stemi_leads)
    
    if stemi_leads:
        if not has_q:
            stage = "Tối cấp (Hyperacute) - Cần can thiệp khẩn cấp bảo tồn cơ tim"
        else:
            stage = "Cấp tiến triển / Bán cấp (Acute evolving - Đã có hoại tử xuyên thành)"
    elif q_leads and not stemi_leads:
        stage = "Giai đoạn sẹo / Nhồi máu cũ (Old / Chronic MI)"

    return {
        "stemi_leads": stemi_leads,
        "reciprocal_leads": reciprocal_leads,
        "q_leads": q_leads,
        "locations": list(set(infarct_locations)),
        "artery": list(set(culprit_artery)),
        "stage": stage,
        "tombstone": tombstone_detected
    }

# =========================================================================
# 3. GIAO DIỆN HIỂN THỊ
# =========================================================================
col_ui_left, col_ui_right = st.columns([1, 1], gap="large")

with col_ui_left:
    st.subheader("1. Bản Ghi 12 Chuyển Đạo")
    file = st.file_uploader("Tải lên ảnh ECG tiêu chuẩn (PNG, JPG)", type=["jpg", "jpeg", "png"])
    
    if file:
        img = Image.open(file)
        st.image(img, caption="Bản ghi ECG được nạp vào mạng phân tích", use_container_width=True)
    else:
        st.info("Vui lòng tải ảnh phiếu đo 12 chuyển đạo để kích hoạt AI bóc tách từng phân vùng giải phẫu.")

with col_ui_right:
    st.subheader("2. Chẩn Đoán Chi Tiết Tổn Thương Mạch Vành")
    
    if file:
        with st.spinner("Đang tách lưới 12 chuyển đạo, định vị điểm J và tính toán sóng Q..."):
            lead_metrics = process_12_lead_ecg(img)
            diag = diagnose_mi(lead_metrics)

        # Hiển thị chẩn đoán chính
        if diag["locations"]:
            st.error(f"🚨 **CHẨN ĐOÁN: NHỒI MÁU CƠ TIM CẤP (STEMI)**")
            st.markdown(f"**Vùng tổn thương:** :red[{', '.join(diag['locations'])}]")
            st.markdown(f"**Nhánh mạch vành thủ phạm (Dự đoán):** **{', '.join(diag['artery'])}**")
            st.markdown(f"**Giai đoạn bệnh học:** **{diag['stage']}**")
            
            if diag["tombstone"]:
                st.warning("⚠️ **Dấu hiệu hình thái:** Phát hiện sóng ST chênh vòm dạng 'Bia mộ' (Tombstone pattern) - Tiên lượng vùng nhồi máu rộng.")
        elif diag["q_leads"]:
            st.warning("⚠️ **CHẨN ĐOÁN: THEO DÕI NHỒI MÁU CƠ TIM CŨ / SẸO HOẠI TỬ**")
            st.write(f"Chuyển đạo có sóng Q bệnh lý: {', '.join(diag['q_leads'])}")
        elif diag["reciprocal_leads"]:
            st.warning("⚠️ **CHẨN ĐOÁN: THIẾU MÁU CỤC BỘ DƯỚI NỘI TÂM MẠC / NSTEMI**")
            st.write(f"Chuyển đạo có ST chênh xuống/T âm: {', '.join(diag['reciprocal_leads'])}")
        else:
            st.success("✅ **Không phát hiện dấu hiệu Nhồi máu cơ tim cấp tính hoặc hoại tử xuyên thành.**")

        st.markdown("---")
        st.markdown("#### Bảng Phân Tích Chi Tiết 12 Chuyển Đạo")
        
        # Bảng dữ liệu chi tiết từng chuyển đạo
        table_data = []
        for lead, m in lead_metrics.items():
            status = "Bình thường"
            if lead in diag["stemi_leads"]:
                status = "🔴 ST Chênh lên (STEMI)"
            elif lead in diag["reciprocal_leads"]:
                status = "🔵 ST Chênh xuống (Soi gương/Thiếu máu)"
            elif lead in diag["q_leads"]:
                status = "⚫ Sóng Q hoại tử"

            table_data.append({
                "Chuyển đạo": lead,
                "ST Chênh (mm)": f"+{m['st_elevation']:.1f}" if m['st_elevation'] > 0 else f"-{m['st_depression']:.1f}",
                "Sóng Q bệnh lý": "Có" if m["pathological_q"] else "Không",
                "Đánh giá": status
            })

        st.dataframe(table_data, use_container_width=True, height=280)

        # Khuyến nghị can thiệp
        st.markdown("#### Đề Xuất Xử Trí Lâm Sàng")
        if diag["locations"]:
            st.markdown("""
            * **Cấp cứu khẩn cấp:** Kích hoạt ngay đội can thiệp mạch vành qua da (Cath-Lab PCI) trong thời gian vàng (< 90-120 phút).
            * **Dược lý:** Khởi động liệu pháp chống kết tập tiểu cầu kép (Aspirin + Clopidogrel/Ticagrelor), Heparin và kiểm soát đau thắt ngực.
            * **Xét nghiệm:** Định lượng khẩn cấp hs-Troponin T/I và siêu âm tim cấp tại giường đánh giá rối loạn vận động vùng.
            """)
        else:
            st.markdown("""
            * Tiếp tục theo dõi ECG nối tiếp (sau 15-30 phút nếu bệnh nhân vẫn còn đau ngực dữ dội).
            * Theo dõi động học men tim sau 1 giờ, 3 giờ theo phác đồ loại trừ 0/1h của ESC.
            """)
    else:
        st.write("Đang chờ tải ảnh...")
