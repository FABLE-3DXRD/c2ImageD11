#ifndef LOCALMAXLABEL_HPP
#define LOCALMAXLABEL_HPP

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "cImageD11.h"

/* Threads read l[] of pixels that another thread is relabelling. A thread
 * that reads l[q] == 0 then reads lout[q], so lout[q] must be written before
 * l[q] is zeroed and the reader must see them in that order.
 * (ImageD11 PR #689.)
 */
#if defined(__GNUC__) || defined(__clang__)
#define LOAD_ACQ(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define STORE_REL(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#else /* msvc: volatile accesses have acquire / release semantics on x86 */
#define LOAD_ACQ(p) (*(volatile uint8_t *)(p))
#define STORE_REL(p, v) (*(volatile uint8_t *)(p) = (v))
#endif

#define pick(A, B, I, J)                                                       \
    if ((A) > (B)) {                                                           \
        (B) = (A);                                                             \
        (I) = (J);                                                             \
    }

template<typename TPixel>
static int neighbormax_impl(const TPixel *restrict im, int32_t *restrict lout,
                             uint8_t *restrict l, intptr_t dim0, intptr_t dim1) {
    intptr_t i, j; int p, k0, k1, k2, npks;
    TPixel mx0, mx1, mx2;
    npks = 0;

    for (i = 0; i < dim1; i++) {
        lout[i] = 0;
        l[i] = 0;
        lout[dim1 * (dim0 - 1) + i] = 0;
        l[dim1 * (dim0 - 1) + i] = 0;
    }

#pragma omp parallel for private(j, p, k0, k1, k2, mx0, mx1, mx2) reduction(+ : npks)
    for (i = dim1; i < (dim0 - 1) * dim1; i = i + dim1) {
        lout[i] = 0;
        l[i] = 0;
        p = i + 1;
        mx0 = im[p - 1 - dim1];
        k0 = 1;
        pick(im[p - 1], mx0, k0, 2);
        pick(im[p - 1 + dim1], mx0, k0, 3);
        k1 = 4;
        mx1 = im[p - dim1];
        pick(im[p], mx1, k1, 5);
        pick(im[p + dim1], mx1, k1, 6);
        for (j = 1; j < dim1 - 1; j++) {
            p = i + j;
            mx2 = im[p + 1 - dim1];
            k2 = 7;
            pick(im[p + 1], mx2, k2, 8);
            pick(im[p + 1 + dim1], mx2, k2, 9);
            pick(mx1, mx0, k0, k1);
            pick(mx2, mx0, k0, k2);
            l[p] = k0;
            if (k0 == 5) {
                lout[i]++;
            }
            mx0 = mx1;
            k0 = k1 - 3;
            mx1 = mx2;
            k1 = k2 - 3;
        }
        lout[i + dim1 - 1] = 0;
        l[i + dim1 - 1] = 0;
        npks += lout[i];
    }
    return npks;
}

template<typename TPixel>
static int localmaxlabel_impl(const TPixel *restrict im, int32_t *restrict lout,
                               uint8_t *restrict l, intptr_t dim0, intptr_t dim1) {
    intptr_t i, j, p, q, nt, lo, hi; int k, npk, t, tid, npks;
    uint8_t v;
#define noisy 0
    int o[10] = {0,
                 (int)(-1 - dim1), -1,          (int)(-1 + dim1), (int)(-dim1), 0,
                 (int)(+dim1),     (int)(+1 - dim1), +1,          (int)(+1 + dim1)};

    if (noisy)
        printf("Not using intrinsics\n");
    npks = neighbormax_impl<TPixel>(im, lout, l, dim0, dim1);
    if (noisy) {
        printf("    neighbormax %d peaks\n", npks);
    }
    npk = 0;
    for (i = 0; i < dim0 * dim1; i = i + dim1) {
        t = npk;
        npk += lout[i];
        lout[i] = t;
    }
    if (noisy) {
        printf("    cumsum %d\n", npk);
    }
#pragma omp parallel for private(nt, t, j, p) schedule(dynamic)
    for (i = 0; i < (dim0 - 1); i++) {
        t = lout[i * dim1];
        nt = lout[(i + 1) * dim1];
        if (t == nt)
            continue;
        for (j = 1; j < (dim1 - 1); j++) {
            p = dim1 * i + j;
            if (l[p] == 5) {
                t++;
                lout[p] = t;
                l[p] = 0;
                if (t == nt)
                    break;
            }
        }
    }
    for (i = 0; i < dim0 * dim1; i = i + dim1) {
        lout[i] = 0;
    }
    if (noisy) {
        printf("    relabel\n");
    }
#pragma omp parallel private(q, i, tid, nt, k, lo, hi, v)
    {
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nt = omp_get_num_threads();
#else
        tid = 0;
        nt = 1;
#endif
        lo = dim0 * dim1 * tid / nt;
        hi = dim0 * dim1 * (tid + 1) / nt;
        for (i = lo; i < hi; i++) {
            v = LOAD_ACQ(&l[i]);
            if (v == 0)
                continue;
            k = 0;
            q = i + o[v];
            v = LOAD_ACQ(&l[q]);
            while (v) {
                q = q + o[v];
                v = LOAD_ACQ(&l[q]);
                k++;
            }
            lout[i] = lout[q];
            if (k > 0) {
                q = i + o[LOAD_ACQ(&l[i])];
                v = LOAD_ACQ(&l[q]);
                while (v) {
                    if ((q >= lo) && (q < hi)) {
                        lout[q] = lout[i]; // label first, then mark done
                        STORE_REL(&l[q], 0);
                    }
                    q = q + o[v];
                    v = LOAD_ACQ(&l[q]);
                }
            }
            STORE_REL(&l[i], 0);
        }
    }
    if (noisy) {
        printf("    write\n");
    }
    return npk;
}

#endif
