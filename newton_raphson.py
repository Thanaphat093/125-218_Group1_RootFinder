import pandas as pd
from polynomial import evaluate_poly, derivative_coeffs

def solve_newton_raphson(coeffs, x0, tol, max_iter, stop_criterion):
    """
    ระเบียบวิธีนิวตัน-ราฟสัน (Newton-Raphson Method)
    """
    d_coeffs = derivative_coeffs(coeffs)
    xi = float(x0)
    rows = []

    for i in range(1, max_iter + 1):
        f_xi = evaluate_poly(coeffs, xi)
        df_xi = evaluate_poly(d_coeffs, xi)

        # ดักจับกรณีอนุพันธ์เป็นศูนย์
        if abs(df_xi) < 1e-14:
            return None, pd.DataFrame(rows), f"ข้อผิดพลาด: อนุพันธ์ f'(x) = 0 ที่รอบที่ {i} (x = {xi}) ไม่สามารถหารต่อได้"

        x_next = xi - (f_xi / df_xi)
        f_next = evaluate_poly(coeffs, x_next)

        if stop_criterion == "|f(x)| < ε":
            error = abs(f_next)
        elif stop_criterion == "|x_i+1 - x_i| < ε":
            error = abs(x_next - xi)
        else:  # "|x_i+1 - x_i| / |x_i+1| * 100 < ε"
            error = (abs(x_next - xi) / abs(x_next) * 100.0) if x_next != 0 else 100.0

        rows.append({
            "รอบที่": i,
            "x_i": xi,
            "f(x_i)": f_xi,
            "f'(x_i)": df_xi,
            "x_i+1": x_next,
            "ε": error
        })

        if f_next == 0 or error < tol:
            return x_next, pd.DataFrame(rows), f"ลู่เข้าสู่คำตอบใน {i} รอบ"

        xi = x_next

    return xi, pd.DataFrame(rows), f"เตือน: ไม่ลู่เข้าภายในจำนวนรอบสูงสุด ({max_iter} รอบ) (อาจเกิดการวนเป็นวัฏจักรหรือลู่ออก)"
