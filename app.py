import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from polynomial import format_polynomial, derivative_coeffs, evaluate_poly
from bisection import solve_bisection
from false_position import solve_false_position
from newton_raphson import solve_newton_raphson

st.set_page_config(page_title="Polynomial Root Finder", layout="wide")
st.title("โปรแกรมหารากของสมการพหุนาม (Polynomial Root Finder)")

# --- แถบด้านข้าง: โจทย์ตัวอย่างสำหรับกดทดสอบ T1 - T8 ได้ทันที ---
st.sidebar.header("โหลดโจทย์ทดสอบ (T1 - T8)")
presets = {
    "กำหนดเอง": None,
    "T1: x² - 2 (ราก ≈ 1.414214)": (2, [1.0, 0.0, -2.0], 0.0, 2.0, 1.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T2: x³ + 3x² - 1 (ราก ≈ 0.532089)": (3, [1.0, 3.0, 0.0, -1.0], 0.0, 1.0, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T3: x² - 3x + 1 (ราก ≈ 2.618034)": (2, [1.0, -3.0, 1.0], 2.0, 3.0, 2.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T4: x³ - 2x² - x - 3 (ราก ≈ 2.757279)": (3, [1.0, -2.0, -1.0, -3.0], 2.0, 3.0, 2.0, 0.001, "|x_i+1 - x_i| / |x_i+1| * 100 < ε"),
    "T5: 5x³ - 4x + 1 (ราก ≈ -1.000000)": (3, [5.0, 0.0, -4.0, 1.0], -2.0, -0.5, -2.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T6: x³ - 6x² + 11x - 6 (ช่วง [0.5, 1.5])": (3, [1.0, -6.0, 11.0, -6.0], 0.5, 1.5, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T7: x² + 1 (แจ้งเตือนไม่คร่อมราก)": (2, [1.0, 0.0, 1.0], -1.0, 1.0, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T8: x³ - 2x + 2 (x₀=0 วนวัฏจักร / x₀=-2 ได้คำตอบ)": (3, [1.0, 0.0, -2.0, 2.0], -2.0, -1.0, 0.0, 0.0001, "|x_i+1 - x_i| < ε"),
}
selected_preset = st.sidebar.selectbox("เลือกโจทย์ทดสอบเพื่อเติมค่าอัตโนมัติ:", list(presets.keys()))
preset_data = presets[selected_preset]

# --- 1. ส่วนกรอกสมการพหุนาม (กรอกสัมประสิทธิ์ทีละช่อง) ---
st.subheader("1. กำหนดสมการพหุนาม f(x)")
default_deg = preset_data[0] if preset_data else 3
degree = st.number_input("ระบุดีกรีสูงสุดของพหุนาม (n):", min_value=1, max_value=10, value=default_deg, step=1)

cols = st.columns(degree + 1)
coeffs = []
for i in range(degree + 1):
    power = degree - i
    default_val = preset_data[1][i] if (preset_data and i < len(preset_data[1])) else (1.0 if i == 0 else 0.0)
    label = f"สัมประสิทธิ์ x^{power}" if power > 1 else ("สัมประสิทธิ์ x" if power == 1 else "ค่าคงที่ (a₀)")
    with cols[i]:
        val = st.number_input(label, value=float(default_val), format="%g", key=f"coeff_{i}")
        coeffs.append(val)

# ตรวจสอบกรณีสัมประสิทธิ์เป็น 0 ทั้งหมด
if all(c == 0 for c in coeffs):
    st.error("สมการไม่ถูกต้อง: สัมประสิทธิ์ทุกตัวเป็น 0 กรุณากรอกค่าอย่างน้อย 1 พจน์")
    st.stop()

# แสดงสมการ f(x) และอนุพันธ์ f'(x) อัตโนมัติ
d_coeffs = derivative_coeffs(coeffs)
col_eq1, col_eq2 = st.columns(2)
col_eq1.info(f"**สมการที่รับเข้ามา:** $f(x) = {format_polynomial(coeffs)}$")
col_eq2.success(f"**อนุพันธ์อัตโนมัติ:** $f'(x) = {format_polynomial(d_coeffs)}$")

# --- 2. ส่วนตั้งค่าพารามิเตอร์และเลือกระเบียบวิธี ---
st.subheader("2. ตั้งค่าการคำนวณและเลือกระเบียบวิธี")
mode = st.radio(
    "เลือกโหมดการทำงาน:",
    ["แบ่งครึ่งช่วง (Bisection)", "วางผิดตำแหน่ง (False Position)", "นิวตัน-ราฟสัน (Newton-Raphson)", "เปรียบเทียบทั้ง 3 วิธีพร้อมกัน"],
    horizontal=True
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    default_tol = preset_data[5] if preset_data else 0.0001
    tol = st.number_input("ค่าคลาดเคลื่อนที่ยอมรับได้ (ε):", value=default_tol, min_value=1e-12, format="%.6f")
with c2:
    max_iter = st.number_input("จำนวนรอบสูงสุด:", value=100, min_value=1, max_value=1000, step=1)
with c3:
    criteria_list = ["|x_i+1 - x_i| < ε", "|f(x)| < ε", "|x_i+1 - x_i| / |x_i+1| * 100 < ε"]
    default_crit_idx = criteria_list.index(preset_data[6]) if preset_data else 0
    stop_criterion = st.selectbox("เกณฑ์การหยุด:", criteria_list, index=default_crit_idx)
with c4:
    decimals = st.slider("จำนวนตำแหน่งทศนิยมในตาราง:", min_value=2, max_value=10, value=6)

# ช่องกรอกค่าเริ่มต้นที่เปลี่ยนตามโหมด
if mode in ["แบ่งครึ่งช่วง (Bisection)", "วางผิดตำแหน่ง (False Position)"]:
    ic1, ic2 = st.columns(2)
    xl = ic1.number_input("ขอบซ้าย (XL):", value=preset_data[2] if preset_data else 0.0, format="%g")
    xr = ic2.number_input("ขอบขวา (XR):", value=preset_data[3] if preset_data else 2.0, format="%g")
elif mode == "นิวตัน-ราฟสัน (Newton-Raphson)":
    x0 = st.number_input("จุดเริ่มต้น (x₀):", value=preset_data[4] if preset_data else 1.0, format="%g")
else:  # โหมดเปรียบเทียบทั้ง 3 วิธี
    ic1, ic2, ic3 = st.columns(3)
    xl = ic1.number_input("ขอบซ้าย (XL) สำหรับ Bisection & False Position:", value=preset_data[2] if preset_data else 0.0, format="%g")
    xr = ic2.number_input("ขอบขวา (XR) สำหรับ Bisection & False Position:", value=preset_data[3] if preset_data else 2.0, format="%g")
    x0 = ic3.number_input("จุดเริ่มต้น (x₀) สำหรับ Newton-Raphson:", value=preset_data[4] if preset_data else 1.0, format="%g")

# ฟังก์ชันช่วยจัดรูปแบบทศนิยมใน DataFrame
def format_df(df, dec):
    fmt = {col: f"{{:.{dec}f}}" for col in df.columns if col != "รอบที่"}
    return df.style.format(fmt)

# ฟังก์ชันวาดกราฟ f(x) พร้อมจุดราก
def plot_function(coeffs, root, x_range):
    x_min, x_max = min(x_range) - 1, max(x_range) + 1
    if root is not None:
        x_min, x_max = min(x_min, root - 1), max(x_max, root + 1)
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = [evaluate_poly(coeffs, x) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)", line=dict(color="#2563eb", width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    if root is not None:
        fig.add_trace(go.Scatter(
            x=[root], y=[evaluate_poly(coeffs, root)],
            mode="markers+text", name="ค่ารากที่พบ",
            text=[f"x ≈ {root:.{decimals}f}"], textposition="top center",
            marker=dict(color="#dc2626", size=12, symbol="circle")
        ))
    fig.update_layout(title="กราฟของฟังก์ชัน f(x) และตำแหน่งรากที่พบ", xaxis_title="x", yaxis_title="f(x)", height=400)
    return fig

# --- 3. ปุ่มคำนวณและแสดงผลลัพธ์ ---
if st.button("คำนวณหารากสมการ", type="primary", use_container_width=True):
    st.divider()

    if mode != "เปรียบเทียบทั้ง 3 วิธีพร้อมกัน":
        if mode == "แบ่งครึ่งช่วง (Bisection)":
            root, df, msg = solve_bisection(coeffs, xl, xr, tol, max_iter, stop_criterion)
            plot_range = [xl, xr]
        elif mode == "วางผิดตำแหน่ง (False Position)":
            root, df, msg = solve_false_position(coeffs, xl, xr, tol, max_iter, stop_criterion)
            plot_range = [xl, xr]
        else:
            root, df, msg = solve_newton_raphson(coeffs, x0, tol, max_iter, stop_criterion)
            plot_range = [x0 - 2, x0 + 2]

        if root is None or "เตือน" in msg:
            st.warning(msg)
        else:
            st.success(f"**ผลลัพธ์:** {msg} | **ค่ารากของสมการ $x \\approx$ `{root:.{decimals}f}`**")

        if df is not None and not df.empty:
            st.subheader("ตารางการวนซ้ำ")
            st.dataframe(format_df(df, decimals), use_container_width=True)

        st.plotly_chart(plot_function(coeffs, root, plot_range), use_container_width=True)

    else:
        # โหมดเปรียบเทียบทั้ง 3 วิธีพร้อมกัน
        res_bi = solve_bisection(coeffs, xl, xr, tol, max_iter, stop_criterion)
        res_fp = solve_false_position(coeffs, xl, xr, tol, max_iter, stop_criterion)
        res_nr = solve_newton_raphson(coeffs, x0, tol, max_iter, stop_criterion)

        summary_data = []
        for name, (r, df, msg) in [
            ("แบ่งครึ่งช่วง (Bisection)", res_bi),
            ("วางผิดตำแหน่ง (False Position)", res_fp),
            ("นิวตัน-ราฟสัน (Newton-Raphson)", res_nr)
        ]:
            iters = len(df) if (df is not None and not df.empty) else 0
            root_str = f"{r:.{decimals}f}" if r is not None else "-"
            summary_data.append({"ระเบียบวิธี": name, "ค่ารากที่พบ (x)": root_str, "จำนวนรอบ": iters, "สถานะ": msg})

        st.subheader("ตารางสรุปเปรียบเทียบทั้ง 3 วิธี")
        st.table(pd.DataFrame(summary_data))

        # กราฟเปรียบเทียบค่าคลาดเคลื่อนต่อรอบ
        fig_err = go.Figure()
        for name, (_, df, _) in [("Bisection", res_bi), ("False Position", res_fp), ("Newton-Raphson", res_nr)]:
            if df is not None and not df.empty:
                fig_err.add_trace(go.Scatter(x=df["รอบที่"], y=df["ε"], mode="lines+markers", name=name))
        fig_err.update_layout(title="กราฟเปรียบเทียบอัตราการลู่เข้า (ค่าคลาดเคลื่อน ε ต่อรอบ)", xaxis_title="รอบที่", yaxis_title="ค่าคลาดเคลื่อน (ε)", yaxis_type="log", height=400)
        st.plotly_chart(fig_err, use_container_width=True)

        # แสดงตารางแยก 3 แท็บ
        tab1, tab2, tab3 = st.tabs(["ตาราง Bisection", "ตาราง False Position", "ตาราง Newton-Raphson"])
        with tab1:
            if res_bi[1] is not None and not res_bi[1].empty:
                st.dataframe(format_df(res_bi[1], decimals), use_container_width=True)
            else:
                st.warning(res_bi[2])
        with tab2:
            if res_fp[1] is not None and not res_fp[1].empty:
                st.dataframe(format_df(res_fp[1], decimals), use_container_width=True)
            else:
                st.warning(res_fp[2])
        with tab3:
            if res_nr[1] is not None and not res_nr[1].empty:
                st.dataframe(format_df(res_nr[1], decimals), use_container_width=True)
            else:
                st.warning(res_nr[2])