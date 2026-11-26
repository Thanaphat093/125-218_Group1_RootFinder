def format_polynomial(coeffs: list[float]) -> str:
    degree = len(coeffs) - 1
    terms = []
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = degree - i
        sign = "-" if c < 0 else "+"
        abs_c = abs(c)

        c_str = f"{int(abs_c)}" if abs_c.is_integer() else f"{abs_c:g}"
        if c_str == "1" and power > 0:
            c_str = ""

        var_str = "" if power == 0 else ("x" if power == 1 else f"x{str(power).translate(superscripts)}")
        term = f"{c_str}{var_str}"

        if not terms:
            terms.append(f"-{term}" if sign == "-" else term)
        else:
            terms.append(f" {sign} {term}")

    return "".join(terms) if terms else "0"


def evaluate_poly(coeffs: list[float], x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def derivative_coeffs(coeffs: list[float]) -> list[float]:
    degree = len(coeffs) - 1
    if degree <= 0:
        return [0.0]
    return [coeffs[i] * (degree - i) for i in range(degree)]