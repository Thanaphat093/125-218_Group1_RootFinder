import pandas as pd
from polynomial import evaluate_poly

def solve_bisection(coeffs, xl, xr, tol, max_iter, stop_criterion):
    """
    ระเบียบวิธีแบ่งครึ่งช่วง (Bisection Method)
    """
    fl = evaluate_poly(coeffs, xl)
    fr = evaluate_poly(coeffs, xr)

    # ตรวจสอบเงื่อนไขช่วงคร่อมราก
    if fl * fr > 0:
        return None, None, "ข้อผิดพลาด: ช่วงเริ่มต้นไม่คร่อมราก (f(XL) · f(XR) > 0) กรุณาเปลี่ยนช่วง [XL, XR] ใหม่"
    if fl == 0:
        return xl, pd.DataFrame(), "พบค่ารากที่ขอบซ้าย XL ทันที"
    if fr == 0:
        return xr, pd.DataFrame(), "พบค่ารากที่ขอบขวา XR ทันที"

    rows = []
    xm_old = xl

    for i in range(1, max_iter + 1):
        xm = (xl + xr) / 2.0
        fm = evaluate_poly(coeffs, xm)

        # คำนวณค่าความคลาดเคลื่อนตามเกณฑ์หยุดที่เลือก
        if stop_criterion == "|f(x)| < ε":
            error = abs(fm)
        elif stop_criterion == "|x_i+1 - x_i| < ε":
            error = abs(xm - xm_old) if i > 1 else abs(xr - xl) / 2.0
        else:  # "|x_i+1 - x_i| / |x_i+1| * 100 < ε"
            error = (abs(xm - xm_old) / abs(xm) * 100.0) if (i > 1 and xm != 0) else 100.0

        rows.append({
            "รอบที่": i,
            "XL": xl,
            "XR": xr,
            "XM": xm,
            "f(XL)": fl,
            "f(XR)": fr,
            "f(XM)": fm,
            "ε": error
        })

        # ตรวจสอบเกณฑ์หยุด
        if fm == 0 or (i > 1 and error < tol) or (stop_criterion == "|f(x)| < ε" and error < tol):
            return xm, pd.DataFrame(rows), f"ลู่เข้าสู่คำตอบใน {i} รอบ"

        # ปรับช่วงใหม่
        if fl * fm < 0:
            xr = xm
            fr = fm
        else:
            xl = xm
            fl = fm

        xm_old = xm

    return xm, pd.DataFrame(rows), f"เตือน: ไม่ลู่เข้าภายในจำนวนรอบสูงสุด ({max_iter} รอบ)"