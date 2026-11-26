import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from polynomial import format_polynomial, derivative_coeffs, evaluate_poly
from bisection import solve_bisection
from false_position import solve_false_position
from newton_raphson import solve_newton_raphson

st.set_page_config(page_title="RootFinder Studio", layout="wide")

# ซ่อนเมนูและตกแต่งสไตล์ Minimal ด้วย CSS
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}
    div[data-testid="stMetric"] {
        background-color: #FAFAFA;
        border: 1px solid #E4E4E7;
        padding: 14px 18px;
        border-radius: 10px;
    }
    .eq-box {
        background-color: #F4F4F5;
        border-left: 3px solid #18181B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- ส่วนหัวแบบเรียบง่าย ---
st.markdown("### RootFinder Studio")
st.caption("เครื่องมือคำนวณและวิเคราะห์การลู่เข้าของรากสมการพหุนาม")
st.divider()

# ชุดข้อมูลทดสอบมาตรฐาน T1 - T8
presets = {
    "กำหนดสมการเอง": None,
    "T1 · x² - 2": (2, [1.0, 0.0, -2.0], 0.0, 2.0, 1.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T2 · x³ + 3x² - 1": (3, [1.0, 3.0, 0.0, -1.0], 0.0, 1.0, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T3 · x² - 3x + 1": (2, [1.0, -3.0, 1.0], 2.0, 3.0, 2.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T4 · x³ - 2x² - x - 3 (ε สัมพัทธ์)": (3, [1.0, -2.0, -1.0, -3.0], 2.0, 3.0, 2.0, 0.001, "|x_i+1 - x_i| / |x_i+1| * 100 < ε"),
    "T5 · 5x³ - 4x + 1": (3, [5.0, 0.0, -4.0, 1.0], -2.0, -0.5, -2.0, 0.0001, "|x_i+1 - x_i| < ε"),
    "T6 · x³ - 6x² + 11x - 6": (3, [1.0, -6.0, 11.0, -6.0], 0.5, 1.5, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T7 · x² + 1 (ไม่คร่อมราก)": (2, [1.0, 0.0, 1.0], -1.0, 1.0, 0.5, 0.0001, "|x_i+1 - x_i| < ε"),
    "T8 · x³ - 2x + 2 (วัฏจักรเมื่อ x₀=0)": (3, [1.0, 0.0, -2.0, 2.0], -2.0, -1.0, 0.0, 0.0001, "|x_i+1 - x_i| < ε"),
}

# แบ่งหน้าจอ 2 คอลัมน์: ซ้าย (แผงควบคุม) | ขวา (ผลลัพธ์)
left_col, right_col = st.columns([1.15, 2.0], gap="large")

with left_col:
    selected_preset = st.selectbox("ชุดโจทย์ทดสอบ", list(presets.keys()))
    preset_data = presets[selected_preset]

    st.markdown("**1. สมการพหุนาม**")
    default_deg = preset_data[0] if preset_data else 3
    degree = st.number_input("ดีกรีสูงสุด (n)", min_value=1, max_value=10, value=default_deg, step=1)

    # จัดช่องกรอกสัมประสิทธิ์แถวละ 3 ช่อง ไม่ให้บีบแคบเกินไป
    coeffs = []
    for row_start in range(0, degree + 1, 3):
        row_cols = st.columns(3)
        for offset in range(3):
            idx = row_start + offset
            if idx <= degree:
                power = degree - idx
                default_val = preset_data[1][idx] if (preset_data and idx < len(preset_data[1])) else (1.0 if idx == 0 else 0.0)
                lbl = f"x^{power}" if power > 1 else ("x" if power == 1 else "a₀")
                with row_cols[offset]:
                    val = st.number_input(lbl, value=float(default_val), format="%g", key=f"c_{idx}")
                    coeffs.append(val)

    if all(c == 0 for c in coeffs):
        st.error("สัมประสิทธิ์ทุกตัวเป็น 0 กรุณากรอกค่าอย่างน้อย 1 พจน์")
        st.stop()

    d_coeffs = derivative_coeffs(coeffs)
    st.markdown(
        f'<div class="eq-box">'
        f'<b>f(x)</b> = {format_polynomial(coeffs)}<br>'
        f'<span style="color:#52525B;"><b>f\'(x)</b> = {format_polynomial(d_coeffs)}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("**2. ระเบียบวิธีและเงื่อนไข**")
    mode = st.selectbox(
        "ระเบียบวิธี",
        ["Bisection (แบ่งครึ่งช่วง)", "False Position (วางผิดตำแหน่ง)", "Newton-Raphson (นิวตัน-ราฟสัน)", "เปรียบเทียบทั้ง 3 วิธี"]
    )

    if mode in ["Bisection (แบ่งครึ่งช่วง)", "False Position (วางผิดตำแหน่ง)"]:
        c_in1, c_in2 = st.columns(2)
        xl = c_in1.number_input("ขอบซ้าย (XL)", value=preset_data[2] if preset_data else 0.0, format="%g")
        xr = c_in2.number_input("ขอบขวา (XR)", value=preset_data[3] if preset_data else 2.0, format="%g")
    elif mode == "Newton-Raphson (นิวตัน-ราฟสัน)":
        x0 = st.number_input("จุดเริ่มต้น (x₀)", value=preset_data[4] if preset_data else 1.0, format="%g")
    else:
        c_in1, c_in2, c_in3 = st.columns(3)
        xl = c_in1.number_input("XL", value=preset_data[2] if preset_data else 0.0, format="%g")
        xr = c_in2.number_input("XR", value=preset_data[3] if preset_data else 2.0, format="%g")
        x0 = c_in3.number_input("x₀", value=preset_data[4] if preset_data else 1.0, format="%g")

    with st.expander("ตั้งค่าขั้นสูง (ε, รอบสูงสุด, เกณฑ์หยุด, ทศนิยม)", expanded=True):
        p1, p2 = st.columns(2)
        default_tol = preset_data[5] if preset_data else 0.0001
        tol = p1.number_input("ค่าคลาดเคลื่อน (ε)", value=default_tol, min_value=1e-12, format="%.6f")
        max_iter = p2.number_input("รอบสูงสุด", value=100, min_value=1, max_value=1000, step=1)

        criteria_list = ["|x_i+1 - x_i| < ε", "|f(x)| < ε", "|x_i+1 - x_i| / |x_i+1| * 100 < ε"]
        default_crit_idx = criteria_list.index(preset_data[6]) if preset_data else 0
        stop_criterion = st.selectbox("เกณฑ์การหยุด", criteria_list, index=default_crit_idx)
        decimals = st.slider("ทศนิยมในตาราง", min_value=2, max_value=10, value=6)

def format_df(df: pd.DataFrame, dec: int):
    fmt = {col: f"{{:.{dec}f}}" for col in df.columns if col != "รอบที่"}
    return df.style.format(fmt)

def build_minimal_plot(coeffs: list[float], root: float | None, x_range: list[float], dec: int):
    x_min, x_max = min(x_range) - 1, max(x_range) + 1
    if root is not None:
        x_min, x_max = min(x_min, root - 1), max(x_max, root + 1)
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = [evaluate_poly(coeffs, x) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x)", line=dict(color="#18181B", width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="#A1A1AA")
    if root is not None:
        fig.add_trace(go.Scatter(
            x=[root], y=[evaluate_poly(coeffs, root)],
            mode="markers+text", name="Root",
            text=[f"  x ≈ {root:.{dec}f}"], textposition="top right",
            marker=dict(color="#2563EB", size=10)
        ))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        height=340,
        xaxis_title="x", yaxis_title="f(x)",
        showlegend=False
    )
    return fig

# --- พื้นที่แสดงผลฝั่งขวา ---
with right_col:
    if mode != "เปรียบเทียบทั้ง 3 วิธี":
        if mode == "Bisection (แบ่งครึ่งช่วง)":
            res = solve_bisection(coeffs, xl, xr, tol, max_iter, stop_criterion)
            plot_range = [xl, xr]
        elif mode == "False Position (วางผิดตำแหน่ง)":
            res = solve_false_position(coeffs, xl, xr, tol, max_iter, stop_criterion)
            plot_range = [xl, xr]
        else:
            res = solve_newton_raphson(coeffs, x0, tol, max_iter, stop_criterion)
            plot_range = [x0 - 2, x0 + 2]

        # การ์ดสรุปผล 3 ช่องด้านบน
        m1, m2, m3 = st.columns(3)
        m1.metric("ค่ารากที่พบ (x)", f"{res.root:.{decimals}f}" if res.root is not None else "ไม่พบ")
        m2.metric("จำนวนรอบ", f"{res.iterations} รอบ")
        m3.metric("True Error (เทียบ numpy)", f"{res.true_error:.2e}" if res.true_error is not None else "-")

        if not res.is_success:
            st.warning(f"สถานะ: {res.status}")

        tab_graph, tab_table = st.tabs(["กราฟฟังก์ชัน f(x)", "ตารางการวนซ้ำ"])
        with tab_graph:
            st.plotly_chart(build_minimal_plot(coeffs, res.root, plot_range, decimals), use_container_width=True)
        with tab_table:
            if not res.table.empty:
                st.dataframe(format_df(res.table, decimals), use_container_width=True, height=340)
                csv_data = res.table.to_csv(index=False).encode("utf-8-sig")
                st.download_button("ดาวน์โหลดตาราง (CSV)", data=csv_data, file_name=f"{res.method_name}_result.csv", mime="text/csv")
            else:
                st.info("ไม่มีข้อมูลการวนซ้ำ")

    else:
        res_bi = solve_bisection(coeffs, xl, xr, tol, max_iter, stop_criterion)
        res_fp = solve_false_position(coeffs, xl, xr, tol, max_iter, stop_criterion)
        res_nr = solve_newton_raphson(coeffs, x0, tol, max_iter, stop_criterion)

        summary_rows = []
        for r in [res_bi, res_fp, res_nr]:
            summary_rows.append({
                "ระเบียบวิธี": r.method_name,
                "ค่าราก (x)": f"{r.root:.{decimals}f}" if r.root is not None else "-",
                "จำนวนรอบ": r.iterations,
                "True Error": f"{r.true_error:.2e}" if r.true_error is not None else "-",
                "สถานะ": r.status
            })

        st.markdown("**สรุปเปรียบเทียบทั้ง 3 ระเบียบวิธี**")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        tab_conv, tab_t1, tab_t2, tab_t3 = st.tabs(["กราฟอัตราการลู่เข้า", "Bisection", "False Position", "Newton-Raphson"])
        with tab_conv:
            fig_err = go.Figure()
            colors = {"Bisection": "#71717A", "False Position": "#2563EB", "Newton-Raphson": "#18181B"}
            for r in [res_bi, res_fp, res_nr]:
                if not r.table.empty:
                    fig_err.add_trace(go.Scatter(
                        x=r.table["รอบที่"], y=r.table["ε"],
                        mode="lines+markers", name=r.method_name,
                        line=dict(color=colors[r.method_name], width=2)
                    ))
            fig_err.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                xaxis_title="รอบที่", yaxis_title="ค่าคลาดเคลื่อน ε (Log)", yaxis_type="log"
            )
            st.plotly_chart(fig_err, use_container_width=True)

        for tab, r in zip([tab_t1, tab_t2, tab_t3], [res_bi, res_fp, res_nr]):
            with tab:
                if not r.table.empty:
                    st.dataframe(format_df(r.table, decimals), use_container_width=True, height=300)
                else:
                    st.warning(r.status)