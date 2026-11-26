from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class MethodResult:
    method_name: str
    root: Optional[float]
    iterations: int
    table: pd.DataFrame
    status: str
    is_success: bool
    true_error: Optional[float] = None

def compute_true_error(coeffs: list[float], approx_root: Optional[float]) -> Optional[float]:
    """เทียบค่ารากที่หาได้กับรากอ้างอิงจาก numpy.roots เพื่อหาค่าคลาดเคลื่อนที่แท้จริง"""
    if approx_root is None:
        return None
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        return None
    closest_root = real_roots[np.argmin(np.abs(real_roots - approx_root))]
    return float(abs(closest_root - approx_root))