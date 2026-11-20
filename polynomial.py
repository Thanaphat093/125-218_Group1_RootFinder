def format_polynomial(coeffs):
    """
    แปลงลิสต์สัมประสิทธิ์ [a_n, a_{n-1}, ..., a_0] ให้เป็นข้อความสมการที่อ่านง่าย
    เช่น [1, -2, -1, -3] -> "x³ - 2x² - x - 3"
    """
    degree = len(coeffs) - 1
    terms = []
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        power = degree - i
        sign = "-" if c < 0 else "+"
        abs_c = abs(c)

        # จัดรูปแบบตัวเลขสัมประสิทธิ์ (ตัด .0 ทิ้งถ้าเป็นจำนวนเต็ม)
        c_str = f"{int(abs_c)}" if abs_c.is_integer() else f"{abs_c:g}"
        if c_str == "1" and power > 0:
            c_str = ""

        # จัดรูปแบบตัวแปร x และเลขชี้กำลัง
        if power == 0:
            var_str = ""
        elif power == 1:
            var_str = "x"
        else:
            var_str = f"x{str(power).translate(superscripts)}"

        term = f"{c_str}{var_str}"
        if not terms:
            terms.append(f"-{term}" if sign == "-" else term)
        else:
            terms.append(f" {sign} {term}")

    return "".join(terms) if terms else "0"


def evaluate_poly(coeffs, x):
    """คำนวณค่า f(x) ด้วยวิธี Horner's Method (แม่นยำและรวดเร็ว)"""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def derivative_coeffs(coeffs):
    """
    หาสัมประสิทธิ์ของอนุพันธ์ f'(x) ตามกฎ d/dx (a_k * x^k) = k * a_k * x^(k-1)
    """
    degree = len(coeffs) - 1
    if degree <= 0:
        return [0.0]
    return [coeffs[i] * (degree - i) for i in range(degree)]