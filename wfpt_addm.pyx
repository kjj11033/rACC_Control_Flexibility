# cython: embedsignature=True
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# distutils: language = c++
#
# Cython version of the Navarro & Fuss, 2009 DDM PDF. Based on the following code by Navarro & Fuss:
# http://www.psychocmath.logy.adelaide.edu.au/personalpages/staff/danielnavarro/resources/wfpt.m
#
# This implementation is about 170 times faster than the matlab
# reference version.
#
# Copyleft Thomas Wiecki (thomas_wiecki[at]brown.edu) & Imri Sofer, 2011
# GPLv3

import hddm
include "integrate.pxi"

import scipy.integrate as integrate
from copy import copy
import numpy as np

cimport numpy as np
cimport cython

from cython.parallel import *

def wiener_like_addm(np.ndarray[double, ndim=1] x,
                      np.ndarray[longlong, ndim=1] conflict,
                      np.ndarray[longlong, ndim=1] starting_trial,    
                      double alpha, double om, double v, 
                      double sv, double a, double z, double sz, double t,
                      double st, double err, int n_st=10, int n_sz=10, bint use_adaptive=1, double simps_err=1e-8,
                      double p_outlier=0, double w_outlier=0):
    # q here represents conflict probability
    # Here, conflict with parameter alpha will be used to update conflict probability
    
    # parameter list
    # 1. DDM parameter (v,a,z,t,sv,sz,st)
    # 2. alpha
    # 3. om
    
    # input list
    # 1. x: RT
    # 2. Conf: Conflict
    # 3. q_init=0.5 fix
    
    # variable list
    # q: conflict probability
    
    cdef int N = x.shape[0]
    cdef Py_ssize_t j
    cdef Py_ssize_t s_size
    cdef double p
    cdef double sum_logp = 0
    cdef double wp_outlier = w_outlier * p_outlier
    cdef double alfa
    cdef double qs
    cdef double qs_pre
    cdef double xs    
    cdef double conflicts
    cdef double starting_trial_idx    

    cdef extern from "math.h":
        double log(double x) nogil

    #N = x.shape[0]
    sum_logp = 0
    wp_outlier = w_outlier * p_outlier    
    
    if not ((p_outlier >= 0) & (p_outlier <= 1)):
        return -np.inf
    alfa = (2.718281828459**alpha) / (1 + 2.718281828459**alpha)
    q_init_t = 0.5
    om_trans = (2.718281828459**om) / (1 + 2.718281828459**om)
    
    for j in range(N):
        conflicts = conflict[j]
        starting_trial_idx = starting_trial[j]
        xs = x[j]
        if ((j==0) or (starting_trial_idx==1)):
            qs_pre = q_init_t
            
        qs = qs_pre + alfa*(conflicts-qs_pre)
        if conflicts==1:
            p = full_pdf(xs, v*((1-om_trans)*qs_pre+om_trans), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        elif conflicts==0:            
            p = full_pdf(xs, v*((1-om_trans)*(1-qs_pre)+om_trans), sv, a, z,
                         sz, t, st, err, n_st, n_sz, use_adaptive, simps_err)
        
        qs_pre = qs           
            
        # If one probability = 0, the log sum will be -Inf
        p = p * (1 - p_outlier) + wp_outlier
        if p == 0:
            return -np.inf        
        sum_logp += np.log(p)        

        
    return sum_logp     