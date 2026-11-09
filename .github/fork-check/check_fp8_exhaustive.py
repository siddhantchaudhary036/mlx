"""Exhaustive check of mx.to_fp8 / mx.from_fp8 against an exact reference.

The reference rounds with exact rational arithmetic (fractions.Fraction):
nearest E4M3FN value, ties to even code, saturate to 448 (0x7E), NaN -> 0x7F.

Test points: every fp8 value, every midpoint between neighbours, and the
nearest representable numbers just above and below each midpoint in the
input dtype. Those are the points where double rounding goes wrong.

Usage:
    python check_fp8_exhaustive.py            # checks float32 and float64
    python check_fp8_exhaustive.py float64
"""

import math
import sys
from fractions import Fraction

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

# Decode the finite non-negative codes 0x00..0x7E. float32 decoding is exact.
CODES = list(range(0x7F))
VALUES = mx.from_fp8(mx.array(CODES, dtype=mx.uint8), dtype=mx.float32).tolist()
EXACT = [Fraction(v) for v in VALUES]
assert VALUES[0x7E] == 448.0 and VALUES[0x38] == 1.0, "unexpected fp8 table"


def reference(x: float) -> int:
    if math.isnan(x):
        return 0x7F
    sign = 0x80 if math.copysign(1.0, x) < 0 else 0
    a = Fraction(abs(x)) if math.isfinite(x) else None
    if a is None or a >= EXACT[-1]:
        return sign | 0x7E
    best = min(CODES, key=lambda c: (abs(a - EXACT[c]), c & 1))
    return sign | best


def test_points(np_dtype):
    pts = set()
    for i, v in enumerate(VALUES):
        pts.add(v)
        if i + 1 < len(VALUES):
            mid = (VALUES[i] + VALUES[i + 1]) / 2
            m = np_dtype(mid)
            pts.add(float(m))
            pts.add(float(np.nextafter(m, np_dtype(np.inf))))
            pts.add(float(np.nextafter(m, np_dtype(0))))
    pts.update([500.0, 1e6, float("inf")])
    pos = sorted(p for p in pts if float(np_dtype(p)) == p)
    return pos + [-p for p in pos] + [float("nan")]


def check(mx_dtype, np_dtype):
    pts = test_points(np_dtype)
    got = mx.to_fp8(mx.array(pts, dtype=mx_dtype)).tolist()
    bad = [(p, g, reference(p)) for p, g in zip(pts, got) if g != reference(p)]

    decoded = mx.from_fp8(mx.array(list(range(256)), dtype=mx.uint8), dtype=mx_dtype)
    want = mx.from_fp8(mx.array(list(range(256)), dtype=mx.uint8), dtype=mx.float32)
    dec_ok = mx.array_equal(decoded.astype(mx.float32), want, equal_nan=True).item()

    print(f"{mx_dtype}: to_fp8 {len(pts) - len(bad)}/{len(pts)} correct, "
          f"from_fp8 {'OK' if dec_ok else 'WRONG'}")
    for p, g, r in bad[:10]:
        print(f"    x={p!r}: got {g:#04x}, expected {r:#04x}")
    return not bad and dec_ok


if __name__ == "__main__":
    which = sys.argv[1:] or ["float32", "float64"]
    table = {"float32": (mx.float32, np.float32), "float64": (mx.float64, np.float64)}
    ok = all(check(*table[w]) for w in which)
    sys.exit(0 if ok else 1)
