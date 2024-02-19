# distutils: language = c++
# distutils: extra_compile_args=-fopenmp -std=c++11 -std=gnu++11
# distutils: extra_link_args=-fopenmp

from cython.parallel cimport prange
cimport cython
from libc.math cimport sqrt, floor
import numpy as np
cimport openmp
from libcpp.vector cimport vector

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double dist_parallel(double[::1] x, double[::1] y):
#     cdef double total = 0
#     cdef Py_ssize_t i
#     for i in prange(x.shape[0], nogil=True):
#         total += x[i]*y[i]
#     total = sqrt(total)
#     return total

@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef double l2_dist(double[::1] x, double[::1] y) nogil:
    cdef double total = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t num = x.shape[0]
    cdef double tmp
    for i in range(num):
        tmp = x[i]-y[i]
        total += tmp*tmp
    return sqrt(total)

@cython.boundscheck(False)
@cython.wraparound(False)
def update_dists(double[::1] dists,double[:,::1] elements, double[::1] item):
    cdef Py_ssize_t num = elements.shape[0]
    cdef Py_ssize_t i
    cdef double tmp
    for i in prange(num, nogil=True):
        tmp = l2_dist(elements[i],item)
        dists[i] = min(dists[i],tmp)

@cython.boundscheck(False)
@cython.wraparound(False)
def update_dists_sequential(double[::1] dists,double[:,::1] elements, double[::1] item):
    cdef Py_ssize_t num = elements.shape[0]
    cdef Py_ssize_t i
    cdef double tmp
    for i in range(num):
        tmp = l2_dist(elements[i],item)
        dists[i] = min(dists[i],tmp)

@cython.boundscheck(False)
@cython.wraparound(False)
def pdist(double[:,::1] features):
    cdef Py_ssize_t N = features.shape[0]
    cdef Py_ssize_t M = N*(N-1)//2
    results = np.zeros(M)
    cdef Py_ssize_t i,j,idx
    cdef double[::1] results_view = results
    cdef double tmp = 0.0
    for idx in prange(M,nogil=True):
        i = N - 2 - <Py_ssize_t>(sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5)
        j = idx + i + 1 - M + (N-i)*((N-i)-1)//2
        tmp = l2_dist(features[i],features[j])
        # idx = N * i + j - ((i + 2) * (i + 1))//2
        results_view[idx]=tmp
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
def cdist(double[:,::1] x, double[:,::1] y ):
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t M = y.shape[0]
    results = np.zeros(N*M)
    cdef double[::1] results_view = results
    cdef Py_ssize_t i,j,idx
    cdef double tmp = 0.0
    for idx in prange(N*M,nogil=True):
        i = idx//M
        j = idx%M
        tmp = l2_dist(x[i],y[j])
        # idx = N * i + j - ((i + 2) * (i + 1))//2
        results_view[idx]=tmp
    return results

@cython.boundscheck(False)
@cython.wraparound(False)
def edges(double[:,::1] features,double diversity_threshold):
    """
    Returns the edges between items which have distance below a threshold.
    """
    cdef Py_ssize_t N = features.shape[0]
    cdef Py_ssize_t M = N*(N-1)//2
    cdef Py_ssize_t i,j,idx
    cdef vector[vector[Py_ssize_t]] us,vs
    cdef vector[vector[double]] dists
    cdef vector[Py_ssize_t] u,v
    cdef vector[double] dist
    cdef double tmp = 0.0
    cdef int tid,num_threads
    num_threads = openmp.omp_get_max_threads()
    us.resize(num_threads)
    vs.resize(num_threads)
    dists.resize(num_threads)
    for idx in prange(M,nogil=True):
        tid = openmp.omp_get_thread_num()
        # https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        i = N - 2 - <Py_ssize_t>(sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5)
        j = idx + i + 1 - M + (N-i)*((N-i)-1)//2
        tmp = l2_dist(features[i],features[j])
        # idx = N * i + j - ((i + 2) * (i + 1))//2
        if tmp < diversity_threshold:
            us[tid].push_back(i)
            vs[tid].push_back(j)
            dists[tid].push_back(tmp)
    
    # unroll all thread arrays
    for tid in range(num_threads):
        for _u in us[tid]:
            u.push_back(_u)
        for _v in vs[tid]:
            v.push_back(_v)
        for _dist in dists[tid]:
            dist.push_back(_dist)
    return u,v,dist