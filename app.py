import streamlit as st
import numpy as np
from PIL import Image
import math

st.set_page_config(
    page_title="Hệ Thống Chẩn Đoán ECG Tiêu Chuẩn Quốc Tế (AHA/ACC/ESC)",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Hệ Thống Phân Tích & Chẩn Đoán ECG Chuyên Khoa")
st.caption("Tiêu chuẩn chẩn đoán định lượng dựa trên khuyến cáo Hội Tim Mạch Hoa Kỳ (AHA/ACC) và Châu Âu (ESC)")

tab_input, tab_report = st.tabs(["📝 Nhập Liệu Lâm Sàng & ECG", "📋 Báo Cáo Chẩn Đoán Chi Tiết"])

with tab_input:
    col_demo, col_intervals = st.columns([1, 1], gap="large")

    with col_demo:
        st.subheader("1. Thông tin Bệnh Nhân & Bản Ghi")
        patient_name = st.text_input("Họ và tên bệnh nhân", value="Trần Thị B")
        c_age, c_gen = st.columns(2)
        with c_age:
            age = st.number_input("Tuổi", min_value=1, max_value=120, value=65)
        with c_gen:
            gender = st.selectbox("Giới tính", ["Nam", "Nữ"], index=1)

        uploaded_file = st.file_uploader("Tải lên bản ghi 12 chuyển đạo (Ảnh)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Ảnh bản ghi ECG", use_container_width=True)

        st.subheader("2. Hình Thái Sóng P & Nhĩ")
        p_morph = st.selectbox(
            "Đặc điểm sóng P",
            [
                "Bình thường (Đồng dạng, dương ở DII, aVF, âm ở aVR)",
                "P cao ≥ 2.5 mm ở DII/DIII/aVF (P phế)",
                "P rộng ≥ 0.12s, có khía ở DII hoặc 2 pha âm chiếm ưu thế ở V1 (P nhĩ)",
                "Không có sóng P, thay bằng sóng f lăn tăn",
                "Không có sóng P, thay bằng sóng F răng cưa tần số 250-350 l/p",
                "Có khoảng ngưng xoang (Không có P-QRS-T) bội số của PP"
            ]
        )

        st.subheader("3. Ngoại Tâm Thu (Ectopic Beats)")
        ectopic = st.selectbox(
            "Loại nhịp ngoại vị phát hiện",
            [
                "Không có",
                "Ngoại tâm thu nhĩ (PAC - P' đến sớm, QRS hẹp)",
                "Ngoại tâm thu bộ nối (PJC - Không có P hoặc P âm sau QRS, QRS hẹp)",
                "Ngoại tâm thu thất đơn ổ (PVC - QRS rộng, biến dạng, nghỉ bù hoàn toàn)",
                "Ngoại tâm thu thất đa ổ / Chuỗi nhịp đôi / Nhịp ba"
            ]
        )

    with col_intervals:
        st.subheader("4. Các Khoảng Thời Gian & Đo Đoạn Điện Học")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            hr = st.number_input("Tần số thất (nhịp/phút)", 20, 260, 75)
            pr_interval = st.number_input("Khoảng PR (giây)", 0.06, 0.50, 0.16, step=0.01)
            qrs_dur = st.number_input("Thời gian QRS (giây)", 0.04, 0.25, 0.08, step=0.01)
        with col_i2:
            qt_interval = st.number_input("Khoảng QT đo được (giây)", 0.15, 0.80, 0.38, step=0.01)
            axis = st.selectbox("Trục điện tim", ["Bình thường", "Lệch trái (-30° đến -90°)", "Lệch phải (+90° đến +180°)", "Vô định"])
            r_progression = st.selectbox("Đặc điểm QRS tại V1/V6", [
                "Bình thường",
                "Dạng rsR' hoặc 'tai thỏ' ở V1, S sâu rộng ở V6",
                "Sóng R rộng có khía ở V5-V6, D1, aVL; QS ở V1",
                "V1: Dạng vòm (Coved-type) ST chênh lên ≥ 2mm tiếp nối sóng T âm",
                "V1/V2: Dạng yên ngựa (Saddleback) ST chênh lên ≥ 2mm"
            ])

        st.subheader("5. Biên Độ Điện Thế (Tiêu chuẩn Dày Thất)")
        c_v1, c_v5 = st.columns(2)
        with c_v1:
            sv1 = st.number_input("Biên độ sóng S tại V1 (mm)", 0.0, 45.0, 10.0, step=0.5)
            rv1 = st.number_input("Biên độ sóng R tại V1 (mm)", 0.0, 30.0, 2.0, step=0.5)
        with c_v5:
            rv5 = st.number_input("Biên độ sóng R tại V5/V6 (mm)", 0.0, 50.0, 14.0, step=0.5)
            ravl = st.number_input("Biên độ sóng R tại aVL (mm)", 0.0, 30.0, 6.0, step=0.5)

        st.subheader("6. Định Khu ST & Sóng T (Nhồi Máu / Thiếu Máu)")
        st_elevation_leads = st.multiselect(
            "Chuyển đạo có ST chênh lên (≥ 1mm ở ngoại vi, ≥ 1.5-2mm ở trước tim)",
            ["D1", "aVL", "D2", "D3", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V3R", "V4R"]
        )
        st_depression_leads = st.multiselect(
            "Chuyển đạo có ST chênh xuống / Sóng T âm sâu (≥ 0.5mm)",
            ["D1", "aVL", "D2", "D3", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        )
        q_wave = st.checkbox("Có sóng Q bệnh lý hoại tử (rộng ≥ 0.04s, sâu ≥ 25% sóng R)")

# ==================== LOGIC ĐÁNH GIÁ CHẨN ĐOÁN ====================
diagnoses = []
critical_alerts = []

# 1. Nhịp cơ bản & Rối loạn nhịp xoang
if "sóng f lăn tăn" in p_morph:
    diagnoses.append(("Rối loạn nhịp", "Rung nhĩ (Atrial Fibrillation - AFib)"))
elif "sóng F răng cưa" in p_morph:
    diagnoses.append(("Rối loạn nhịp", "Cuồng nhĩ (Atrial Flutter - AFL)"))
elif "khoảng ngưng xoang" in p_morph:
    diagnoses.append(("Rối loạn nhịp xoang", "Block xoang nhĩ (Sinoatrial Exit Block) / Ngưng xoang"))
    critical_alerts.append("Block xoang nhĩ / Ngưng xoang: Nguy cơ ngất hoặc vô tâm thu")
else:
    if hr < 60:
        diagnoses.append(("Rối loạn nhịp xoang", f"Nhịp chậm xoang (Sinus Bradycardia) - Tần số: {hr} l/p"))
    elif hr > 100:
        diagnoses.append(("Rối loạn nhịp xoang", f"Nhịp nhanh xoang (Sinus Tachycardia) - Tần số: {hr} l/p"))
    else:
        diagnoses.append(("Nhịp tim", f"Nhịp xoang bình thường (Normal Sinus Rhythm) - Tần số: {hr} l/p"))

# 2. Ngoại tâm thu
if ectopic != "Không có":
    diagnoses.append(("Rối loạn tính tự động", ectopic))
    if "đa ổ" in ectopic:
        critical_alerts.append("Ngoại tâm thu thất phức tạp: Nguy cơ khởi phát nhanh thất/rung thất")

# 3. Block dẫn truyền nhĩ - thất (AV Block)
if pr_interval > 0.20:
    diagnoses.append(("Dẫn truyền nhĩ thất", f"Block nhĩ thất độ I (First-degree AV Block) - PR = {pr_interval:.2f}s"))
elif pr_interval < 0.12 and "sóng f" not in p_morph:
    diagnoses.append(("Hội chứng kích thích sớm", "Khoảng PR ngắn (< 0.12s): Nghi ngờ Hội chứng Wolff-Parkinson-White (WPW)"))

# 4. Block dẫn truyền nội thất (Bundle Branch Block)
if qrs_dur >= 0.12:
    if "tai thỏ" in r_progression:
        diagnoses.append(("Dẫn truyền nội thất", "Block nhánh phải hoàn toàn (Complete RBBB)"))
    elif "Sóng R rộng có khía" in r_progression:
        diagnoses.append(("Dẫn truyền nội thất", "Block nhánh trái hoàn toàn (Complete LBBB)"))
        critical_alerts.append("LBBB mới xuất hiện có thể là tương đương nhồi máu cơ tim cấp (STEMI equivalent)")
    else:
        diagnoses.append(("Dẫn truyền nội thất", f"Chậm dẫn truyền nội thất không đặc hiệu (IVCD) - QRS: {qrs_dur:.2f}s"))
elif 0.10 <= qrs_dur < 0.12:
    if "tai thỏ" in r_progression:
        diagnoses.append(("Dẫn truyền nội thất", "Block nhánh phải không hoàn toàn (Incomplete RBBB)"))
    elif "Sóng R rộng có khía" in r_progression:
        diagnoses.append(("Dẫn truyền nội thất", "Block nhánh trái không hoàn toàn (Incomplete LBBB)"))

# Block phân nhánh
if axis == "Lệch trái (-30° đến -90°)" and qrs_dur < 0.12:
    diagnoses.append(("Dẫn truyền nội thất", "Block phân nhánh trái trước (Left Anterior Fascicular Block - LAFB)"))
elif axis == "Lệch phải (+90° đến +180°)" and qrs_dur < 0.12:
    diagnoses.append(("Dẫn truyền nội thất", "Block phân nhánh trái sau (Left Posterior Fascicular Block - LPFB)"))

# 5. Phì đại / Lớn buồng tim
# Lớn nhĩ
if "P cao ≥ 2.5 mm" in p_morph:
    diagnoses.append(("Phì đại buồng tim", "Lớn nhĩ phải (Right Atrial Enlargement / P phế)"))
elif "P rộng ≥ 0.12s" in p_morph:
    diagnoses.append(("Phì đại buồng tim", "Lớn nhĩ trái (Left Atrial Enlargement / P nhĩ)"))

# Dày thất (Tiêu chuẩn Sokolow-Lyon & Cornell)
sokolow_lv = sv1 + rv5
sokolow_rv = rv1 + (sv1 * 0.5)
cornell = ravl + sv1

is_lvh = False
if sokolow_lv >= 35.0:
    diagnoses.append(("Phì đại buồng tim", f"Dày thất trái (LVH) theo Sokolow-Lyon (SV1 + RV5 = {sokolow_lv:.1f} mm ≥ 35 mm)"))
    is_lvh = True
if (gender == "Nam" and cornell > 28.0) or (gender == "Nữ" and cornell > 20.0):
    diagnoses.append(("Phì đại buồng tim", f"Dày thất trái (LVH) theo Cornell (RaVL + SV1 = {cornell:.1f} mm)"))
    is_lvh = True

if rv1 > 7.0 or (rv1 > sv1 and axis == "Lệch phải (+90° đến +180°)"):
    diagnoses.append(("Phì đại buồng tim", f"Dày thất phải (RVH) - RV1 = {rv1:.1f} mm, trục lệch phải"))

# 6. Khoảng QTc
rr_sec = 60.0 / hr
qtc_ms = (qt_interval / math.sqrt(rr_sec)) * 1000
qtc_cutoff = 460 if gender == "Nữ" else 450

if qtc_ms > qtc_cutoff:
    diagnoses.append(("Khoảng tái cực", f"Khoảng QTc kéo dài ({qtc_ms:.0f} ms > {qtc_cutoff} ms)"))
    critical_alerts.append(f"QTc kéo dài ({qtc_ms:.0f} ms): Nguy cơ cao khởi phát Xoắn đỉnh (Torsades de Pointes)")

# 7. Hội chứng Brugada
if "Dạng vòm" in r_progression:
    diagnoses.append(("Bệnh kênh ion (Channelopathies)", "Hội chứng Brugada Type 1 (ST chênh lên dạng vòm ≥ 2mm tại V1-V2)"))
    critical_alerts.append("Brugada Type 1: Nguy cơ đột tử do loạn nhịp thất, chỉ định theo dõi chuyên khoa tim mạch")
elif "Dạng yên ngựa" in r_progression:
    diagnoses.append(("Bệnh kênh ion (Channelopathies)", "Gợi ý hình thái Brugada Type 2 (ST chênh dạng yên ngựa tại V1-V2)"))

# 8. Nhồi máu cơ tim & Thiếu máu cục bộ theo phân vùng
st_set = set(st_elevation_leads)

# Xác định vùng giải phẫu nhồi máu ST chênh lên (STEMI)
stemi_regions = []
if {"V1", "V2"}.issubset(st_set) and not {"V3", "V4"}.issubset(st_set):
    stemi_regions.append("Vách liên thất (Septal - V1, V2)")
if {"V3", "V4"}.issubset(st_set):
    stemi_regions.append("Thành trước (Anterior - V3, V4)")
if {"V1", "V2", "V3", "V4"}.issubset(st_set):
    stemi_regions.append("Trước - Vách (Anteroseptal - V1-V4)")
if {"V1", "V2", "V3", "V4", "V5", "V6"}.issubset(st_set):
    stemi_regions.append("Trước rộng (Extensive Anterior - V1-V6, D1, aVL)")
if {"D2", "D3", "aVF"}.intersection(st_set) and len({"D2", "D3", "aVF"}.intersection(st_set)) >= 2:
    stemi_regions.append("Thành dưới (Inferior - DII, DIII, aVF)")
if {"D1", "aVL"}.issubset(st_set) or {"V5", "V6"}.issubset(st_set):
    stemi_regions.append("Thành bên (Lateral - DI, aVL, V5, V6)")
if {"V7", "V8", "V9"}.intersection(st_set):
    stemi_regions.append("Thành sau thực thụ (Posterior - V7-V9)")
if {"V3R", "V4R"}.intersection(st_set):
    stemi_regions.append("Thất phải (Right Ventricular - V3R, V4R)")

if stemi_regions:
    regions_str = ", ".join(stemi_regions)
    stage = "Bán cấp / Có hoại tử" if q_wave else "Cấp tính (Tối cấp/Cấp)"
    diagnoses.append(("Hội chứng vành cấp", f"Nhồi máu cơ tim ST chênh lên (STEMI) - Vùng: {regions_str} - Giai đoạn: {stage}"))
    critical_alerts.append(f"🚨 STEMI VÙNG {regions_str.upper()}: KÍCH HOẠT QUY TRÌNH CAN THIỆP MẠCH VÀNH CẤP CỨU (PCI)")

# Thiếu máu cục bộ cơ tim (Ischemia / Non-STEMI)
dep_set = set(st_depression_leads)
ischemia_regions = []
if {"D2", "D3", "aVF"}.intersection(dep_set):
    ischemia_regions.append("Thành dưới (DII, DIII, aVF)")
if {"V4", "V5", "V6"}.intersection(dep_set):
    ischemia_regions.append("Thành trước - bên (V4-V6)")
if {"D1", "aVL"}.intersection(dep_set):
    ischemia_regions.append("Thành bên cao (DI, aVL)")

if ischemia_regions and not stemi_regions:
    regions_str = ", ".join(ischemia_regions)
    diagnoses.append(("Thiếu máu cục bộ cơ tim", f"Thiếu máu cục bộ cơ tim / NSTEMI - Vùng: {regions_str}"))

# ==================== HIỂN THỊ BÁO CÁO ====================
with tab_report:
    st.header("KẾT QUẢ ĐỌC ĐIỆN TÂM ĐỒ TOÀN DIỆN")
    st.markdown(f"**Bệnh nhân:** {patient_name} | **Tuổi:** {age} | **Giới:** {gender}")
    st.markdown("---")

    # Hiển thị cảnh báo đỏ khẩn cấp
    if critical_alerts:
        for alert in critical_alerts:
            st.error(f"🚨 **CẢNH BÁO LÂM SÀNG NGUY CẤP:** {alert}")

    # Bảng chỉ số tóm tắt
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Tần số tim", f"{hr} bpm")
    col_m2.metric("PR", f"{pr_interval:.2f} s")
    col_m3.metric("QRS", f"{qrs_dur:.2f} s")
    col_m4.metric("QTc (Bazett)", f"{qtc_ms:.0f} ms")
    col_m5.metric("Sokolow-Lyon (Thất trái)", f"{sokolow_lv:.1f} mm")

    st.markdown("### Kết Luận Chẩn Đoán Chi Tiết")
    
    # Hiển thị từng nhóm bệnh
    for category, text in diagnoses:
        if "Nhồi máu cơ tim" in text or "Brugada" in text or "Nguy cơ" in text:
            st.markdown(f"- **[{category}]** :red[{text}]")
        elif "Block" in text or "Dày" in text or "Thiếu máu" in text or "Lớn" in text:
            st.markdown(f"- **[{category}]** :orange[{text}]")
        else:
            st.markdown(f"- **[{category}]** :green[{text}]")

    st.markdown("---")
    st.markdown("### Đề Xuất Xử Trí Lâm Sàng")
    if critical_alerts:
        st.write("1. **Cấp cứu mạch vành/Hồi sức:** Khẩn trương làm xét nghiệm Men tim (Troponin I/T siêu nhạy), điện giải đồ toàn phần.")
        st.write("2. **Theo dõi liên tục:** Gắn monitor theo dõi sát nhịp tim và dấu hiệu sinh tồn, chuẩn bị máy sốc điện tại giường.")
        st.write("3. **Hội chẩn:** Kích hoạt cath-lab can thiệp mạch vành nếu chẩn đoán STEMI.")
    elif is_lvh or "Block" in str(diagnoses):
        st.write("1. Chỉ định **Siêu âm tim qua thành ngực (TTE)** để đánh giá chức năng tâm thu thất trái (LVEF) và kích thước buồng tim.")
        st.write("2. Đánh giá huyết áp động mạch và rà soát các yếu tố nguy cơ tim mạch nền.")
    else:
        st.write("1. Hiện tại không phát hiện tổn thương cơ tim cấp hoặc rối loạn dẫn truyền ác tính.")
        st.write("2. Đề nghị kết hợp chặt chẽ với triệu chứng cơ năng (đau ngực, khó thở, ngất) để theo dõi ngoại trú định kỳ.")

    st.caption("Báo cáo được khởi tạo tự động dựa trên thuật toán tham chiếu AHA/ESC. Cần bác sĩ chuyên khoa tim mạch ký duyệt trước khi ra y lệnh.")
