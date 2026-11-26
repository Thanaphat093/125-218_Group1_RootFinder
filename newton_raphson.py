import pandas as pd
from polynomial import evaluate_poly, derivative_coeffs
from models import MethodResult, compute_true_error

def solve_newton_raphson(coeffs: list[float], x0: float, tol: float, max_iter: int, stop_criterion: str) -> MethodResult:
    d_coeffs = derivative_coeffs(coeffs)
    xi = float(x0)
    rows = []

    for i in range(1, max_iter + 1):
        f_xi = evaluate_poly(coeffs, xi)
        df_xi = evaluate_poly(d_coeffs, xi)

        if abs(df_xi) < 1e-14:
            return MethodResult("Newton-Raphson", None, i, pd.DataFrame(rows), f"อนุพันธ์ f'(x) = 0 ที่รอบที่ {i} (x = {xi})", False)

        x_next = xi - (f_xi / df_xi)
        f_next = evaluate_poly(coeffs, x_next)

        if stop_criterion == "|f(x)| < ε":
            error = abs(f_next)
        elif stop_criterion == "|x_i+1 - x_i| < ε":
            error = abs(x_next - xi)
        else:
            error = (abs(x_next - xi) / abs(x_next) * 100.0) if x_next != 0 else 100.0

        rows.append({
            "รอบที่": i, "x_i": xi, "f(x_i)": f_xi,
            "f'(x_i)": df_xi, "x_i+1": x_next, "ε": error
        })

        if f_next == 0 or error < tol:
            df = pd.DataFrame(rows)
            return MethodResult("Newton-Raphson", x_next, i, df, f"ลู่เข้าใน {i} รอบ", True, compute_true_error(coeffs, x_next))

        xi = x_next

    df = pd.DataFrame(rows)
    return MethodResult("Newton-Raphson", xi, max_iter, df, f"ไม่ลู่เข้าภายใน {max_iter} รอบ (อาจวนเป็นวัฏจักร)", False, compute_true_error(coeffs, xi))