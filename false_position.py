import pandas as pd
from polynomial import evaluate_poly
from models import MethodResult, compute_true_error

def solve_false_position(coeffs: list[float], xl: float, xr: float, tol: float, max_iter: int, stop_criterion: str) -> MethodResult:
    fl = evaluate_poly(coeffs, xl)
    fr = evaluate_poly(coeffs, xr)

    if fl * fr > 0:
        return MethodResult("False Position", None, 0, pd.DataFrame(), "ช่วงไม่คร่อมราก (f(XL)·f(XR) > 0)", False)
    if fl == 0:
        return MethodResult("False Position", xl, 0, pd.DataFrame(), "พบค่ารากที่ขอบซ้าย XL ทันที", True, 0.0)
    if fr == 0:
        return MethodResult("False Position", xr, 0, pd.DataFrame(), "พบค่ารากที่ขอบขวา XR ทันที", True, 0.0)

    rows = []
    xm_old = xl

    for i in range(1, max_iter + 1):
        if (fl - fr) == 0:
            return MethodResult("False Position", None, i, pd.DataFrame(rows), "ตัวหาร f(XL) - f(XR) เป็นศูนย์", False)

        xm = xr - (fr * (xl - xr)) / (fl - fr)
        fm = evaluate_poly(coeffs, xm)

        if stop_criterion == "|f(x)| < ε":
            error = abs(fm)
        elif stop_criterion == "|x_i+1 - x_i| < ε":
            error = abs(xm - xm_old) if i > 1 else abs(xm - xl)
        else:
            error = (abs(xm - xm_old) / abs(xm) * 100.0) if (i > 1 and xm != 0) else 100.0

        rows.append({
            "รอบที่": i, "XL": xl, "XR": xr, "XM": xm,
            "f(XL)": fl, "f(XR)": fr, "f(XM)": fm, "ε": error
        })

        if fm == 0 or (i > 1 and error < tol) or (stop_criterion == "|f(x)| < ε" and error < tol):
            df = pd.DataFrame(rows)
            return MethodResult("False Position", xm, i, df, f"ลู่เข้าใน {i} รอบ", True, compute_true_error(coeffs, xm))

        if fl * fm < 0:
            xr, fr = xm, fm
        else:
            xl, fl = xm, fm
        xm_old = xm

    df = pd.DataFrame(rows)
    return MethodResult("False Position", xm, max_iter, df, f"ไม่ลู่เข้าภายใน {max_iter} รอบ", False, compute_true_error(coeffs, xm))