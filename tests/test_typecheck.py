"""Test that type mismatches in buffer arguments are properly rejected."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np
import c2ImageD11

# The wrapper returns C integers as int on Python 3 and long on Python 2.
try:
    INT_TYPES = (int, long)  # noqa: F821 (Python 2)
except NameError:
    INT_TYPES = (int,)


def test_score_f64_only():
    """score() with f64 gv works."""
    ubi = np.eye(3, dtype=np.float64)
    gv = np.random.randn(10, 3).astype(np.float64)
    n = c2ImageD11.score(ubi, gv, 0.5)
    assert type(n) in INT_TYPES
    assert n >= 0


def test_score_f32_works():
    """score() with f32 gv works (via score_f32 baseline)."""
    ubi = np.eye(3, dtype=np.float64)
    gv = np.random.randn(10, 3).astype(np.float32)
    n = c2ImageD11.score(ubi, gv, 0.5)
    assert type(n) in INT_TYPES
    assert n >= 0


def test_score_and_refine_f64_works():
    """score_and_refine() with f64 gv works."""
    ubi = np.random.randn(3, 3).astype(np.float64)
    gv = np.random.randn(10, 3).astype(np.float64)
    n, s = c2ImageD11.score_and_refine(ubi, gv, 0.5)
    assert type(n) in INT_TYPES


def test_score_and_refine_f32_works():
    """score_and_refine() with f32 gv works."""
    ubi = np.random.randn(3, 3).astype(np.float64)
    gv = np.random.randn(10, 3).astype(np.float32)
    n, s = c2ImageD11.score_and_refine(ubi, gv, 0.5)
    assert type(n) in INT_TYPES


def test_score_rejects_int_gv():
    """score() given int gv raises an error (no matching overload)."""
    ubi = np.eye(3, dtype=np.float64)
    gv = np.random.randint(0, 10, size=(10, 3))
    with pytest.raises((ValueError, TypeError)):
        c2ImageD11.score(ubi, gv, 0.5)


def test_score_rejects_1d_ubi():
    """score() rejects 1D ubi (needs 3x3)."""
    gv = np.random.randn(10, 3).astype(np.float64)
    with pytest.raises(ValueError):
        c2ImageD11.score(np.ones(9), gv, 0.5)


def test_localmaxlabel_rejects_float64_with_arg_info():
    """localmaxlabel() with float64 data raises TypeError naming the arg,
    its 1-based position, and the received PEP 3118 format."""
    img = np.random.poisson(3, (21, 21)).astype(np.float64)
    labels = np.zeros(img.shape, np.int32)
    wrk = np.zeros(img.shape, np.uint8)
    with pytest.raises(TypeError) as exc:
        c2ImageD11.localmaxlabel(img, labels, wrk)
    msg = str(exc.value)
    assert "argument 1" in msg
    assert "(data)" in msg
    assert "has unsupported format 'd'" in msg


def test_score_and_refine_soa_via_T():
    """score_and_refine() accepts (3,N) SoA via .T.copy()."""
    ubi = np.random.randn(3, 3).astype(np.float64)
    gv = np.random.randn(10, 3).astype(np.float64)
    n, s = c2ImageD11.score_and_refine(ubi, gv.T.copy(), 0.5)
    assert type(n) in INT_TYPES
