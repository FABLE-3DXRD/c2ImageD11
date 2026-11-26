"""localmaxlabel must give the same labels with many threads as with one.

Port of the test from ImageD11 PR #689 (thread race on relabelled paths).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import multiprocessing
import os

import numpy as np
import pytest

import c2ImageD11 as ci


def _ncores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def _box_smooth(a, w):
    """Separable box filter of width 2*w+1 (edges clipped), numpy only."""
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(w + 1, w) if ax == axis else (0, 0)
                                 for ax in (0, 1)], mode="edge"), axis=axis)
        if axis == 0:
            a = (c[2 * w + 1:] - c[:-2 * w - 1]) / (2 * w + 1)
        else:
            a = (c[:, 2 * w + 1:] - c[:, :-2 * w - 1]) / (2 * w + 1)
    return a


@pytest.mark.skipif(_ncores() < 16,
                    reason="the race shows only with many threads (before "
                           "the fix, 3-5/500 runs wrong at 64-96 threads, "
                           "0/500 at 32 or fewer)")
def test_threads_match_one_thread():
    # smoothed noise gives many basins with long uphill paths
    rs = np.random.RandomState(0)
    im = rs.random_sample((2048, 2048))
    for _ in range(3):  # three box passes ~ gaussian
        im = _box_smooth(im, 4)
    im = im.astype(np.float32)
    im -= np.median(im)
    im[im < 0] = 0
    lab = np.empty(im.shape, np.int32)
    wrk = np.empty(im.shape, np.uint8)
    nmax = ci.cimaged11_omp_get_max_threads()
    ci.cimaged11_omp_set_num_threads(1)
    try:
        n1 = ci.localmaxlabel(im, lab, wrk)
        ref = lab.copy()
        ci.cimaged11_omp_set_num_threads(_ncores())
        nbad = 0
        for _ in range(500):
            lab[:] = rs.randint(1, 10 ** 6, lab.shape)  # stale labels
            n = ci.localmaxlabel(im, lab, wrk)
            assert n == n1
            if not np.array_equal(lab, ref):
                nbad += 1
        assert nbad == 0, "{} of 500 threaded runs differ".format(nbad)
    finally:
        ci.cimaged11_omp_set_num_threads(nmax)
