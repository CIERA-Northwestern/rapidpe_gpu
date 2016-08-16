# Copyright (C) 2013  Evan Ochsner, R. O'Shaughnessy
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Code to compute the log likelihood of parameters of a gravitational waveform. Precomputes terms that depend only on intrinsic parameters and computes the log likelihood for given values of extrinsic parameters

Requires python SWIG bindings of the LIGO Algorithms Library (LAL)
"""
import lal
import lalsimulation as lalsim
from lalinference.rapid_pe import lalsimutils as lsu
import numpy as np
from scipy import interpolate, integrate
from scipy import special
from itertools import product
from common_cl import distRef

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, R. O'Shaughnessy"

#
# Main driver functions
#
def precompute_likelihood_terms(event_time_geo, t_window, P, data_dict,
        psd_dict, Lmax, fMax, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0., remove_zero=True, verbose=True):
    """
    Compute < h_lm(t) | d > and < h_lm | h_l'm' >

    Returns:
        - Dictionary of interpolating functions, keyed on detector, then (l,m)
          e.g. rholms_intp['H1'][(2,2)]
        - Dictionary of "cross terms" <h_lm | h_l'm' > keyed on (l,m),(l',m')
          e.g. crossTerms[((2,2),(2,1))]
        - Dictionary of discrete time series of < h_lm(t) | d >, keyed the same
          as the interpolating functions.
          Their main use is to validate the interpolating functions
    """
    assert data_dict.keys() == psd_dict.keys()
    detectors = data_dict.keys()
    rholms = {}
    rholms_intp = {}
    crossTermsU = {}
    crossTermsV = {}


    # Compute hlms at a reference distance, distance scaling is applied later
    P.dist = distRef

    P.print_params()
    # Compute all hlm modes with l <= Lmax
    detectors = data_dict.keys()
    # Zero-pad to same length as data - NB: Assuming all FD data same resolution
    P.deltaF = data_dict[detectors[0]].deltaF
    hlms_list = lsu.hlmoff(P, Lmax) # a linked list of hlms
    hlms = lsu.SphHarmFrequencySeries_to_dict(hlms_list, Lmax) # a dictionary


    # If the hlm time series is identically zero, remove it
    zero_modes = []
    for mode in hlms.iterkeys():
        if remove_zero and np.sum(np.abs(hlms[mode].data.data)) == 0:
            zero_modes.append(mode)

    for mode in zero_modes:
        del hlms[mode]

    for det in detectors:
        # This is the event time at the detector
        t_det = vector_compute_arrival_time_at_detector(det, P.phi, P.theta,event_time_geo)
        # The is the difference between the time of the leading edge of the
        # time window we wish to compute the likelihood in, and
        # the time corresponding to the first sample in the rholms
        rho_epoch = data_dict[det].epoch - hlms[hlms.keys()[0]].epoch
        t_shift =  float(t_det - t_window - rho_epoch)
        assert t_shift > 0
        # tThe leading edge of our time window of interest occurs
        # this many samples into the rholms
        N_shift = int( t_shift / P.deltaT )
        # Number of samples in the window [t_ref - t_window, t_ref + t_window]
        N_window = int( 2 * t_window / P.deltaT )
        # Compute cross terms < h_lm | h_l'm' >
        crossTermsU[det], crossTermsV[det] = compute_mode_cross_term_ip(hlms, psd_dict[det], P.fmin,
                fMax, 1./2./P.deltaT, P.deltaF, analyticPSD_Q,
                inv_spec_trunc_Q, T_spec)

        # Compute rholm(t) = < h_lm(t) | d >
        rholms[det] = compute_mode_ip_time_series(hlms, data_dict[det],
                psd_dict[det], P.fmin, fMax, 1./2./P.deltaT, N_shift, N_window,
                analyticPSD_Q, inv_spec_trunc_Q, T_spec)
        rhoXX = rholms[det][rholms[det].keys()[0]]
        # The vector of time steps within our window of interest
        # for which we have discrete values of the rholms
        # N.B. I don't do simply rho_epoch + t_shift, b/c t_shift is the
        # precise desired time, while we round and shift an integer number of
        # steps of size deltaT
        t = np.arange(N_window) * P.deltaT\
                + float(rho_epoch + N_shift * P.deltaT )
        if verbose:
            print "For detector", det, "..."
            print "\tData starts at %.20g" % float(data_dict[det].epoch)
            print "\trholm starts at %.20g" % float(rho_epoch)
            print "\tEvent time at detector is: %.18g" % float(t_det)
            print "\tInterpolation window has half width %g" % t_window
            print "\tComputed t_shift = %.20g" % t_shift
            print "\t(t_shift should be t_det - t_window - t_rholm = %.20g)" %\
                    (t_det - t_window - float(rho_epoch))
            print "\tInterpolation starts at time %.20g" % t[0]
            print "\t(Should start at t_event - t_window = %.20g)" %\
                    (float(rho_epoch + N_shift * P.deltaT))
        # The minus N_shift indicates we need to roll left
        # to bring the desired samples to the front of the array
        rholms_intp[det] =  interpolate_rho_lms(rholms[det], t)

    return rholms_intp, crossTermsU, crossTermsV, rholms

def factored_log_likelihood(extr_params, rholms_intp, crossTermsU, crossTermsV, Lmax):
    """
    Compute the log-likelihood = -1/2 < d - h | d - h > from:
        - extr_params is an object containing values of all extrinsic parameters
        - rholms_intp is a dictionary of interpolating functions < h_lm(t) | d >
        - crossTerms is a dictionary of < h_lm | h_l'm' >
        - Lmax is the largest l-index of any h_lm mode considered

    N.B. rholms_intp and crossTerms are the first two outputs of the function
    'precompute_likelihood_terms'
    """
    # Sanity checks
    assert rholms_intp.keys() == crossTermsU.keys()
    detectors = rholms_intp.keys()

    RA = extr_params.phi
    DEC =  extr_params.theta
    tref = extr_params.tref # geocenter time
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    # FIXME: Strictly speaking, this should be inside the detector loop because
    # there *could* be different l,m pairs for different detectors. This never
    # happens in practice, so it's pulled out here, and we use the first
    # detector as a reference.

#   Ylms = compute_spherical_harmonics(Lmax, incl, -phiref, rholms_intp[rholms_intp.keys()[0]])
    Ylms = vector_compute_spherical_harmonics(incl, -phiref, rholms_intp[rholms_intp.keys()[0]])
	
    lnL = 0.
    for det in detectors:
        CTU = crossTermsU[det]
        CTV = crossTermsV[det]
        F = vector_complex_antenna_factor(det, RA, DEC, psi, tref)

        # This is the GPS time at the detector
        t_det = vector_compute_arrival_time_at_detector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        for key in rholms_intp[det]:
            func = rholms_intp[det][key]
            det_rholms[key] = func(float(t_det))

        lnL += single_detector_log_likelihood(det_rholms, CTU, CTV, Ylms, F, dist)

    return lnL

def factored_log_likelihood_time_marginalized(tvals, extr_params, rholms_intp, rholms, crossTermsU, crossTermsV, det_epochs, Lmax, interpolate=False):
    """
    Compute the log-likelihood = -1/2 < d - h | d - h > from:
        - extr_params is an object containing values of all extrinsic parameters
        - rholms_intp is a dictionary of interpolating functions < h_lm(t) | d >
        - crossTerms is a dictionary of < h_lm | h_l'm' >
        - Lmax is the largest l-index of any h_lm mode considered

    tvals is an array of timeshifts relative to the detector,
    used to compute the marginalized integral.
    It provides both the time prior and the sample points used for the integral.

    N.B. rholms_intp and crossTerms are the first two outputs of the function
    'precompute_likelihood_terms'
    """
    # Sanity checks
    assert rholms_intp.keys() == crossTermsU.keys()
    detectors = rholms_intp.keys()

    RA = extr_params.phi
    DEC =  extr_params.theta
    tref = extr_params.tref # geocenter time
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    # N.B.: The Ylms are a function of - phiref b/c we are passively rotating
    # the source frame, rather than actively rotating the binary.
    # Said another way, the m^th harmonic of the waveform should transform as
    # e^{- i m phiref}, but the Ylms go as e^{+ i m phiref}, so we must give
    # - phiref as an argument so Y_lm h_lm has the proper phiref dependence
    # FIXME: Strictly speaking, this should be inside the detector loop because
    # there *could* be different l,m pairs for different detectors. This never
    # happens in practice, so it's pulled out here, and we use the first
    # detector as a reference.
    Ylms = compute_spherical_harmonics(Lmax, incl, -phiref, rholms[rholms.keys()[0]])

    lnL = 0.

    delta_t = tvals[1] - tvals[0]
    for det in detectors:
        CTU = crossTermsU[det]
        CTV = crossTermsV[det]  
        F = vector_complex_antenna_factor(det, RA, DEC, psi, tref)
        rho_epoch = float(det_epochs[det])

        # This is the GPS time at the detector
        t_det = vector_compute_arrival_time_at_detector(det, RA, DEC, tref)
        det_rholms = {}  # rholms evaluated at time at detector
        if ( interpolate ):
            # use the interpolating functions. 
            for key, func in rholms_intp[det].iteritems():
                det_rholms[key] = func(float(t_det)+tvals)
        else:
            # do not interpolate, just use nearest neighbors.
            for key, rhoTS in rholms[det].iteritems():
	        # PRB: these can be moved outside this loop to after t_det
                tfirst = float(t_det)+tvals[0]
                ifirst = int((tfirst - rho_epoch) / delta_t + 0.5)
                ilast = ifirst + len(tvals)
		det_rholms[key] = rhoTS
                #det_rholms[key] = rhoTS[ifirst:ilast]
        lnL += single_detector_log_likelihood(det_rholms, CTU, CTV, Ylms, F, dist)
	

    maxlnL = np.max(lnL) 
    return maxlnL + np.log(np.sum(np.exp(lnL - maxlnL)) * (tvals[1]-tvals[0]))

def single_detector_log_likelihood(rholm_vals, crossTermsU, crossTermsV, Ylms, F, dist):
    """
    Compute the value of the log-likelihood at a single detector from
    several intermediate pieces of data.

    Inputs:
      - rholm_vals: A dictionary of values of inner product between data
            and h_lm modes, < h_lm(t*) | d >, at a single time of interest t*
      - crossTerms: A dictionary of inner products between h_lm modes:
            < h_lm | h_l'm' >
      - Ylms: Dictionary of values of -2-spin-weighted spherical harmonic modes
            for a certain inclination and ref. phase, Y_lm(incl, - phiref)
      - F: Complex-valued antenna pattern depending on sky location and
            polarization angle, F = F_+ + i F_x
      - dist: The distance from the source to detector in meters

    Outputs: The value of ln L for a single detector given the inputs.
    """ 
    invDistMpc = distRef/dist
    Fstar = np.conj(F)

    term1, term20, term21 = 0., 0., 0.
    # PRB: I think this loop can be vectorized with some work
    for pair1, Ylm1 in Ylms.iteritems():
        l1, m1 = pair1
        n_one_l1 = (-1)**l1
        Ylm1_conj = np.conj(Ylm1)
        term1 += Ylm1_conj * rholm_vals[pair1]
        tmp_term20, tmp_term21 = 0., 0.
	    # PRB: should also re-pack the crossterms into arrays
        for pair2, Ylm2 in Ylms.iteritems():
            tmp_term20 += crossTermsU[(pair1, pair2)] * Ylm2
            tmp_term21 += crossTermsV[(pair1, pair2)] * Ylm2
        term20 += tmp_term20 * Ylm1_conj
        term21 += tmp_term21 * Ylm1 * n_one_l1
    term1 = np.real( Fstar * term1 ) * invDistMpc 
    term1 += -0.25 * np.real( F * ( Fstar * term20 + F * term21 ) ) * invDistMpc * invDistMpc 

    return term1

def compute_mode_ip_time_series(hlms, data, psd, fmin, fMax, fNyq,
        N_shift, N_window, analyticPSD_Q=False,
        inv_spec_trunc_Q=False, T_spec=0.):
    """
    Compute the complex-valued overlap between
    each member of a SphHarmFrequencySeries 'hlms'
    and the interferometer data COMPLEX16FrequencySeries 'data',
    weighted the power spectral density REAL8FrequencySeries 'psd'.

    The integrand is non-zero in the range: [-fNyq, -fmin] union [fmin, fNyq].
    This integrand is then inverse-FFT'd to get the inner product
    at a discrete series of time shifts.

    Returns a SphHarmTimeSeries object containing the complex inner product
    for discrete values of the reference time tref.  The epoch of the
    SphHarmTimeSeries object is set to account for the transformation
    """
    rholms = {}
    assert data.deltaF == hlms[hlms.keys()[0]].deltaF
    assert data.data.length == hlms[hlms.keys()[0]].data.length
    deltaT = data.data.length/(2*fNyq)

    # Create an instance of class to compute inner product time series
    IP = lsu.ComplexOverlap(fmin, fMax, fNyq, data.deltaF, psd,
            analyticPSD_Q, inv_spec_trunc_Q, T_spec, full_output=True)

    # Loop over modes and compute the overlap time series
    for pair in hlms.keys():
        rho, rhoTS, rhoIdx, rhoPhase = IP.ip(hlms[pair], data)
        rhoTS.epoch = data.epoch - hlms[pair].epoch
        rholms[pair] = lal.CutCOMPLEX16TimeSeries(rhoTS, N_shift, N_window)

    return rholms

def interpolate_rho_lm(rholm, t):
    h_re = np.real(rholm.data.data)
    h_im = np.imag(rholm.data.data)
    # spline interpolate the real and imaginary parts of the time series
    h_real = interpolate.InterpolatedUnivariateSpline(t, h_re, k=3)
    h_imag = interpolate.InterpolatedUnivariateSpline(t, h_im, k=3)
    return lambda ti: h_real(ti) + 1j*h_imag(ti)

    # Little faster
    #def anon_intp(ti):
        #idx = np.searchsorted(t, ti)
        #return rholm.data.data[idx]
    #return anon_intp

    #from pygsl import spline
    #spl_re = spline.cspline(len(t))
    #spl_im = spline.cspline(len(t))
    #spl_re.init(t, np.real(rholm.data.data))
    #spl_im.init(t, np.imag(rholm.data.data))
    #@profile
    #def anon_intp(ti):
        #re = spl_re.eval_e_vector(ti)
        #return re + 1j*im
    #return anon_intp

    # Doesn't work, hits recursion depth
    #from scipy.signal import cspline1d, cspline1d_eval
    #re_coef = cspline1d(np.real(rholm.data.data))
    #im_coef = cspline1d(np.imag(rholm.data.data))
    #dx, x0 = rholm.deltaT, float(rholm.epoch)
    #return lambda ti: cspline1d_eval(re_coef, ti) + 1j*cspline1d_eval(im_coef, ti)


def interpolate_rho_lms(rholms, t):
    """
    Return a dictionary keyed on mode index tuples, (l,m)
    where each value is an interpolating function of the overlap against data
    as a function of time shift:
    rholm_intp(t) = < h_lm(t) | d >

    'rholms' is a dictionary keyed on (l,m) containing discrete time series of
    < h_lm(t_i) | d >
    't' is an array of the discrete times:
    [t_0, t_1, ..., t_N]
    """
    rholm_intp = {}
    for mode in rholms.keys():
        rholm = rholms[mode]
        # The mode is identically zero, don't bother with it
        if sum(abs(rholm.data.data)) == 0.0:
            continue
        rholm_intp[ mode ] = interpolate_rho_lm(rholm, t)

    return rholm_intp


def compute_mode_cross_term_ip(hlms, psd, fmin, fMax, fNyq, deltaF,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0., verbose=True):
    """
    Compute the 'cross terms' between waveform modes, i.e.
    < h_lm | h_l'm' >.
    The inner product is weighted by power spectral density 'psd' and
    integrated over the interval [-fNyq, -fmin] union [fmin, fNyq]

    Returns a dictionary of inner product values keyed by tuples of mode indices
    i.e. ((l,m),(l',m'))
    """
    # Create an instance of class to compute inner product
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q,
            inv_spec_trunc_Q, T_spec)

    crossTermsU = {}
    crossTermsV = {}

    for mode1 in hlms.keys():
        for mode2 in hlms.keys():
            crossTermsU[ (mode1,mode2) ] = IP.ip(hlms[mode1], hlms[mode2])
	    crossTermsV[ (mode1,mode2) ] = IP.ip(hlms[mode1], hlms[mode2], conj=True)
            if verbose:
                print "       : U populated ", (mode1, mode2), "  = ",\
                        crossTermsU[(mode1,mode2) ]
                print "       : V populated ", (mode1, mode2), "  = ",\
                        crossTermsV[(mode1,mode2) ]

    return crossTermsU, crossTermsV


def complex_antenna_factor(det, RA, DEC, psi, tref):
    """
    Function to compute the complex-valued antenna pattern function:
    F+ + i Fx

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'psi' is the polarization angle
    'tref' is the reference GPS time
    """  
    detector = lalsim.DetectorPrefixToLALDetector(det)
    Fp, Fc = lal.ComputeDetAMResponse(detector.response, RA, DEC, psi, lal.GreenwichMeanSiderealTime(tref))

    return Fp + 1j * Fc

def vector_complex_antenna_factor(rsp, RA, DEC, psi, tref):
        # Everything is now an array except rsp whcih is
        # the same as it used to be.

        detector = lalsim.DetectorPrefixToLALDetector(rsp)
	rsp = detector.response

        RA   = np.array(RA, copy=False, ndmin=1)
        DEC  = np.array(DEC, copy=False, ndmin=1)
        psi  = np.array(psi, copy=False, ndmin=1)
        tref = np.array(tref, copy=False, ndmin=1)

        N = len(RA)
	# Hour angle

	gha    = np.array(tref - RA, dtype=float)

        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(DEC)
        sindec = np.sin(DEC)
        cospsi = np.cos(psi)
        sinpsi = np.sin(psi)


        X = np.zeros((3, N))
        Y = np.zeros((3, N))

        X[0,:] = -cospsi * singha - sinpsi * cosgha * sindec
        X[1,:] = -cospsi * cosgha + sinpsi * singha * sindec
        X[2,:] =  sinpsi * cosdec

        Y[0,:] = sinpsi * singha - cospsi * cosgha * sindec
        Y[1,:] = sinpsi * cosgha + cospsi * singha * sindec
        Y[2,:] = cospsi * cosdec
        Fp, Fc = np.zeros(N), np.zeros(N)
        Fp = np.einsum('ji,jk,ki->i', X, rsp, X) - np.einsum('ji,jk,ki->i', Y, rsp, Y)
        Fc = np.einsum('ji,jk,ki->i', Y, rsp, X) + np.einsum('ji,jk,ki->i', X, rsp, Y)

        return Fp + 1j * Fc

def compute_spherical_harmonics(Lmax, theta, phi, selected_modes=None):
    """
    Return a dictionary keyed by tuples
    (l,m)
    that contains the values of all
    -2Y_lm(theta,phi)
    with
    l <= Lmax
    -l <= m <= l
    """
    # PRB: would this be faster if we made it a 2d numpy array?  
    Ylms = {}
    for l in range(2,Lmax+1):
        for m in range(-l,l+1):
            if selected_modes is not None and (l,m) not in selected_modes:
                continue
            Ylms[ (l,m) ] = lal.SpinWeightedSphericalHarmonic(theta, phi,-2, l, m)

    return Ylms

def vector_compute_spherical_harmonics(theta, phi, selected_modes=None):
    """
    Compute spherical harmonics for various m with l=2.
    """
    # Only copies if input is scalar, else returns original array
    theta   = np.array(theta, copy=False, ndmin=1)
    theta   = np.array(phi, copy=False, ndmin=1)

    Ylms = {}
    one_m_theta = 1.0 - np.cos(theta)
    one_p_theta = 1.0 + np.cos(theta)
    snTheta = np.sin(theta)
    for m in range(-2, 3):
        if selected_modes is not None and (2,m) not in selected_modes:
            continue
        if m == -2:
            Ylms[(2,m)] = _a2m2 * one_m_theta * one_m_theta * np.exp(1j*m*phi)
        elif m == -1:
            Ylms[(2,m)] = _a2m1 * snTheta * one_m_theta * np.exp(1j*m*phi)
        elif m == 0:
            Ylms[(2,m)] = _a20 * snTheta * snTheta
        elif m == 1:
            Ylms[(2,m)] = _a21 * snTheta * one_p_theta * np.exp(1j*m*phi)
        elif m == 2:
            Ylms[(2,m)] = _a22 * one_p_theta * one_p_theta * np.exp(1j*m*phi)

    return Ylms

def compute_arrival_time_at_detector(det, RA, DEC, tref):
    """
    Function to compute the time of arrival at a detector
    from the time of arrival at the geocenter.

    'det' is a detector prefix string (e.g. 'H1')
    'RA' and 'DEC' are right ascension and declination (in radians)
    'tref' is the reference time at the geocenter.  It can be either a float (in which case the return is a float) or a GPSTime object (in which case it returns a GPSTime)
    """
    detector = lalsim.DetectorPrefixToLALDetector(det)
    # if tref is a float or a GPSTime object,
    # it shoud be automagically converted in the appropriate way
    return tref + lal.TimeDelayFromEarthCenter(detector.location, RA, DEC, tref)

def vector_compute_arrival_time_at_detector(det, RA, DEC, tref, tref_geo=None):
    """
    Transfer a geocentered reference time 'tref' to the detector-based arrival time as a function of sky position.
    If the treef_geo argument is specified, then the tref_geo is used to calculate an hour angle as provided by
    `XLALGreenwichMeanSiderealTime`. This can significantly speed up the function with a sacrifice (likely minimal)
    of accuracy.
    """
    RA   = np.array(RA, copy=False, ndmin=1)
    DEC  = np.array(DEC, copy=False, ndmin=1)
    tref = np.array(tref, copy=False, ndmin=1)


        # Calculate hour angle
    if tref_geo is None:
        time = np.array([lal.GreenwichMeanSiderealTime(t) for t in tref], dtype=float)
    else:
        # FIXME: Could be called once and moved outside
        time = lal.GreenwichMeanSiderealTime(tref_geo)
    hr_angle = np.array(time - RA, dtype=float)

    DEC = np.array(DEC, dtype=float)

    # compute spherical coordinate position of the detector
    # based on hour angle and declination
    cos_hr = np.cos(hr_angle)
    sin_hr = np.sin(hr_angle)

    cos_dec = np.cos(DEC)
    sin_dec = np.sin(DEC)
    # compute source vector
    source_xyz = np.array([cos_dec * cos_hr, cos_dec * -sin_hr, sin_dec])
    # get the detector vector
    # must be careful - the C function is designed to compute the
    # light time delay between two detectors as \vec{det2} - \vec{det1}.
    # But timeDelay.c gets the earth center time delay by passing
    # {0., 0., 0.} as the 2nd arg. So det_xyz needs an extra - sign.
    # FIXME: Could be called once and moved outside
    det_xyz = -lalsim.DetectorPrefixToLALDetector(det).location

    return tref + np.dot(np.transpose(det_xyz), source_xyz) / lal.C_SI

def compute_mode_iterator(Lmax):  # returns a list of (l,m) pairs covering all modes, as a list.  Useful for building iterators without nested lists


    mylist = []
    for L in np.arange(2, Lmax+1):
        for m in np.arange(-L, L+1):
            mylist.append((L,m))
    return mylist


def get_cuda_c():

        import pycuda
        import pycuda.autoinit
        import pycuda.driver as cuda
        import pycuda.cumath as cumath
        import pycuda.gpuarray as gpuarray
        from pycuda.tools import dtype_to_ctype

        #_ SourceModule allows us to write CUDA C
        from pycuda.compiler import SourceModule


	print("Compiling CUDA C... \n")

	mod=SourceModule('''
	#include<math.h>
	#include<cuComplex.h>

	__constant__ int nmodes[1];
	__constant__ int nsamps[1];
	__constant__ int ntimes[1];
	__constant__ int nclmns[1];

	__constant__ double det_tns[9];


	__constant__ double CTU[25];
	__constant__ double CTV[25];


	__device__ int get_global_idx_1d_1d() {
		return blockIdx.x * blockDim.x + threadIdx.x;
	}

	__device__ int get_xidx_within_row() {
		return threadIdx.x + blockIdx.x*blockDim.x;
	}
	__device__ int get_global_idx_2d_1d() {
		return threadIdx.x + gridDim.x*blockDim.x*blockIdx.y + blockIdx.x*blockDim.x;
	}

	__device__ cuDoubleComplex cpx_outer_prod(cuDoubleComplex *CT, cuDoubleComplex *V) {
		
		cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
		
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result = cuCadd(result, cuCmul(cuCmul(cuConj(V[i]), CT[i*3 + j]), V[j]));       
			}
		}
		return result;
	}

        __device__ cuDoubleComplex real_outer_prod(cuDoubleComplex *CT, cuDoubleComplex *V) {
                
                cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
                
                for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                                result = cuCadd(result, cuCmul(cuCmul(V[i], CT[i*3 + j]), V[j]));       
                        }
                }
                return result;
        }




	__global__ void compute_sph_harmonics_l_eq_2(double *theta, double *phi, int *sel_modes, cuDoubleComplex *result) {   

		double _a2m2 = sqrt(5.0 / (64.0 * M_PI));
		double _a2m1 = sqrt(5.0 / (16.0 * M_PI));
		double _a20  = sqrt(15.0 /(32.0 * M_PI));
		double _a21  = sqrt(5.0 / (16.0 * M_PI));
		double _a22  = sqrt(5.0 / (64.0 * M_PI));
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		double th = theta[tid];
		double ph = phi[tid];


		for (int modeIdx = 0; modeIdx < *nmodes; modeIdx++) {
		       int m = sel_modes[modeIdx];
		       if (sel_modes[modeIdx] == -2) {
			       double Re = _a2m2 * (1.0 - cos(th)) * (1.0 - cos(th)) * cos(m*ph);
			       double Im = _a2m2 * (1.0 - cos(th)) * (1.0 - cos(th)) * sin(m*ph);
			       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			       result[tid] = Ylm;
			       tid += *nsamps; 
		       }
		
		       if (sel_modes[modeIdx] == -1) {
			       double Re = _a2m1 * sin(th) * (1.0 - cos(th)) * cos(m*ph);
			       double Im = _a2m1 * sin(th) * (1.0 - cos(th)) * sin(m*ph);
			       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			       result[tid] = Ylm;
			       tid += *nsamps;
		       }
		       if (sel_modes[modeIdx] ==  0) {
			       double Re = _a20  * sin(th) * sin(th) * cos(m*ph);
			       double Im = _a20  * sin(th) * sin(th) * sin(m*ph);
			       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			       result[tid] = Ylm;
			       tid += *nsamps;
		       }
		       if (sel_modes[modeIdx] ==  1) {
			       double Re = _a21  * sin(th) * (1.0 + cos(th)) * cos(m*ph);
			       double Im = _a21  * sin(th) * (1.0 + cos(th)) * sin(m*ph);
			       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			       result[tid] = Ylm;
			       tid += *nsamps;
		       }
		       if (sel_modes[modeIdx] ==  2) {
			       double Re = _a22  * (1.0 + cos(th)) * (1.0 + cos(th)) * cos(m*ph);
			       double Im = _a22  * (1.0 + cos(th)) * (1.0 + cos(th)) * sin(m*ph);
			       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			       result[tid] = Ylm;
			       tid += *nsamps;
		       }
	       } 
	}

	__global__ void complex_antenna_factor(double *ra, double *dec, double *psi, double *tref, cuDoubleComplex *result) {
		
		extern __shared__ double Shr[]; 

		int gid = get_global_idx_1d_1d();       
		int id  = threadIdx.x;

		double *X = &Shr[(id * 6)];
		double *Y = &Shr[(id * 6) + 3];

		double gha = tref[gid] - ra[gid]; 
		X[0] = -cos(psi[gid]) * sin(gha) - sin(psi[gid]) * cos(gha) * sin(dec[gid]);  
		X[1] = -cos(psi[gid]) * cos(gha) + sin(psi[gid]) * sin(gha) * sin(dec[gid]);
		X[2] =  sin(psi[gid]) * cos(dec[gid]);
		Y[0] =  sin(psi[gid]) * sin(gha) - cos(psi[gid]) * cos(gha) * sin(dec[gid]); 
		Y[1] =  sin(psi[gid]) * cos(gha) + cos(psi[gid]) * sin(gha) * sin(dec[gid]);
		Y[2] =  cos(psi[gid]) * cos(dec[gid]);  

		__syncthreads();
		
		double Fp;
		double Fc;


		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				Fp += X[i]*det_tns[i*3 + j]*X[j] - Y[i]*det_tns[i*3 + j]*Y[j]; 
				Fc += Y[i]*det_tns[i*3 + j]*X[j] + X[i]*det_tns[i*3 + j]*Y[j]; 
			}
		}
		result[gid] = make_cuDoubleComplex(Fp, Fc);
	}

	__global__ void expand_rhoTS(cuDoubleComplex *rhots, cuDoubleComplex *rhots_all) {
		
		int gid = get_global_idx_2d_1d();
		int linidx = get_xidx_within_row();
		int which_mode = blockIdx.y % *nmodes;

		if (linidx < *ntimes) {
			rhots_all[gid] = rhots[*ntimes*which_mode + linidx];
		}       
		
	}


	__global__ void double_expand_rhoTS(double *rhots, double *rhots_all) {
		
		int gid = get_global_idx_2d_1d();
		int linidx = get_xidx_within_row();
		int which_mode = blockIdx.y % *nmodes;

		if (linidx < *ntimes) {
			rhots_all[gid] = rhots[*ntimes*which_mode + linidx];
		}       
		
	}

	__global__ void insert_ylms(cuDoubleComplex *padded_ts_all, cuDoubleComplex *ylms_all) {
		extern __shared__ cuDoubleComplex shr[]; // My Ylms will reside here     
		if (threadIdx.x < *nmodes) {
			int offset = (blockIdx.y - (blockIdx.y % *nmodes)) / *nmodes;
			shr[threadIdx.x] = ylms_all[*nsamps*threadIdx.x + offset];
		}
	       
		__syncthreads();
	 
		int gid = get_global_idx_2d_1d() ;      
		
		cuDoubleComplex myins_val = cuCmul(padded_ts_all[gid], shr[blockIdx.y % *nmodes]); 

		padded_ts_all[gid] = myins_val; 

	} 

	__global__ void accordion(cuDoubleComplex *contr_arr) {
		int linidx = get_xidx_within_row();     
		int gid = get_global_idx_2d_1d();
		
		cuDoubleComplex *myrow;
		cuDoubleComplex *nxrow;

		for (int mode = 0; mode < *nmodes - 1; mode++) {
			myrow = &contr_arr[*nclmns*(*nmodes * blockIdx.y + mode)];              
			nxrow = &contr_arr[*nclmns*(*nmodes * blockIdx.y + mode + 1)];
			nxrow[linidx] = cuCadd(myrow[linidx], nxrow[linidx]);
		}                       
		myrow[linidx] = nxrow[linidx];
	} 

	__global__ void make_cpx_3x3_outer_prods(cuDoubleComplex *CT, cuDoubleComplex *all_V, cuDoubleComplex *out) {

		extern __shared__ cuDoubleComplex allshr[];

		int gid = get_global_idx_1d_1d();       
		int id  = threadIdx.x; 
				
		cuDoubleComplex *myshr = &allshr[*nmodes*id];

		for (int i = 0; i < *nmodes; i++) {
			myshr[i] = all_V[(*nsamps * i) + id];   
		}

		out[gid] = cpx_outer_prod(CT, myshr); 

	}

        __global__ void make_real_3x3_outer_prods(cuDoubleComplex *CT, cuDoubleComplex *all_V, cuDoubleComplex *out) {

                extern __shared__ cuDoubleComplex allshr[];

                int gid = get_global_idx_1d_1d();       
                int id  = threadIdx.x; 
                                
                cuDoubleComplex *myshr = &allshr[*nmodes*id];

                for (int i = 0; i < *nmodes; i++) {
                        myshr[i] = all_V[(*nsamps * i) + id];   
                }

                out[gid] = real_outer_prod(CT, myshr); 

        }


	__global__ void bcast_vec_to_matrix(double *matrix, double *vector) { 
		__shared__ double myval[1]; 

		int gid = threadIdx.x + blockIdx.x*blockDim.x + *nmodes*gridDim.x*blockDim.x*blockIdx.y;
		int linidx = get_xidx_within_row();
		int vecidx = blockIdx.y;

		if (threadIdx.x == 0) {	
			myval[0] = vector[vecidx]; 
		}
		__syncthreads();

		if (linidx < *ntimes) {
			matrix[gid + (*nmodes-2)*gridDim.x*blockDim.x] += myval[0];  	
			matrix[gid + (*nmodes-1)*gridDim.x*blockDim.x] += myval[0];  	
		}  
	}



        __global__ void mul_bcast_vec_to_matrix(double *matrix, double *vector) { 
                __shared__ double mmyval[1]; 

                int gid = threadIdx.x + blockIdx.x*blockDim.x + *nmodes*gridDim.x*blockDim.x*blockIdx.y;
                int linidx = get_xidx_within_row();
                int vecidx = blockIdx.y;

                if (threadIdx.x == 0) { 
                        mmyval[0] = vector[vecidx]; 
                }
                __syncthreads();

                if (linidx < *ntimes) {
                        matrix[gid + (*nmodes-2)*gridDim.x*blockDim.x] *= mmyval[0];     
                        matrix[gid + (*nmodes-1)*gridDim.x*blockDim.x] *= mmyval[0];     
                }  
        }



	__global__ void find_max_in_shrmem(double *all_rhots) {
		extern double __shared__ share[];

		int linidx = get_xidx_within_row();
		double *my_row = &all_rhots[*nclmns*(*nmodes*blockIdx.y + *nmodes - 2)]; 
		
		share[threadIdx.x] = my_row[linidx];

		__syncthreads();

		for (unsigned int s=blockDim.x/2; s > 0; s /= 2) {
			if (threadIdx.x < s) {
				if (share[threadIdx.x] <= share[threadIdx.x + s]) {
					share[threadIdx.x] = share[threadIdx.x + s];
				}
			}       
			__syncthreads();
		}       

		if (threadIdx.x == 0) {
			my_row[blockIdx.x] = share[0];	
		} 
	} 

	__global__ void nv_reduc(double *indat) {
		extern __shared__ double reduc[]; 

		unsigned int linidx = get_xidx_within_row();

		double *myrow = &indat[*nclmns*(*nmodes*blockIdx.y + *nmodes - 1) ];    

		reduc[threadIdx.x] = myrow[linidx];
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s /= 2) {
			if (threadIdx.x < s) {
				reduc[threadIdx.x] = reduc[threadIdx.x] + reduc[threadIdx.x+s];
			}
			__syncthreads();
		}       
		if (threadIdx.x == 0) {
			myrow[blockIdx.x] = reduc[0];
		}

		__syncthreads();
		if (linidx > gridDim.x) {
			myrow[linidx] = 0.0;	
		} 

	}
	''')

	print("Done compiling CUDA C! \n")

	return mod


def factored_log_likelihood_time_marginalized_gpu(mod, right_ascension, declination, tref, phiref, inclination, psi, distance, det_tns, rholms, CTU, CTV):
	nsamps = len(right_ascension)
	nmodes = len(rholms.keys()) # Number of modes
	ntimes = len(rholms[rholms.keys()[0]]) # Number of times in rhoTS
	
	rhots_contig = np.zeros((nmodes, ntimes)).astype(np.complex128) # Contiguous time series block 
	sort_terms_keys = sorted(rholms, key=lambda tup: tup[1])
	for i in range(0, len(rholms.keys())):
		rhots_contig[i,:] = rholms[sort_terms_keys[i]]


	return likelihood_function_gpu(mod, right_ascension, declination, tref, phiref, inclination, psi, distance, rhots_contig,  rholms.keys(), nsamps, ntimes, det_tns, CTU, CTV) 

	
def likelihood_function_gpu(mod, phi, theta, tref, phiref, incl, psi, distance, rhoTS, selected_modes, nsamps, ntimes, detector_tensor, CTU, CTV):

	'''
	GPU Accelerated, time-marginalized, factored log likelihood
	Description of Variables:
	
	PHI    - Right Ascension
	THETA  - Declination
	TREF   - Fiducial Epoch
	PHIREF - Orbital Phase
	INCL   - Inclination
	PSI    - Polarization Angle
	DIST   - Luminosity Distance
	'''




	#_ Preliminaries
	import math
	import pycuda
	import pycuda.autoinit
	import pycuda.driver as cuda
	import pycuda.cumath as cumath
	import pycuda.gpuarray as gpuarray
	from pycuda.tools import dtype_to_ctype

	#_ SourceModule allows us to write CUDA C
	from pycuda.compiler import SourceModule

	# CUDA C itself


	#FIXME - Should print device properties here 
	device = cuda.Device(0)
	max_tpb = 512

	#______________NECESSARY CONSTANTS ##
	nclmns  = np.int32( ntimes + max_tpb - (ntimes % max_tpb) ) # Number of cols to pad w/ 0s	

	mlist_sort     = sorted([mode[1] for mode in selected_modes])
	mlist_sort     = np.array(mlist_sort).astype(np.int32)
	nmodes = np.int32(len(selected_modes)) # Number of modes

	print("Passing data down to GPU... \n")
	
	#______________PASS DATA TO GPU ##	

	# **-- constants --**
	nmodes_gpu = mod.get_global("nmodes")[0]
	nsamps_gpu = mod.get_global("nsamps")[0]
	ntimes_gpu = mod.get_global("ntimes")[0]
	nclmns_gpu = mod.get_global("nclmns")[0]
	detector_tensor_gpu = mod.get_global("det_tns")[0]

	cuda.memcpy_htod(nmodes_gpu, np.array(nmodes, ndmin=1).astype(np.int32))
	cuda.memcpy_htod(nsamps_gpu, np.array(nsamps, ndmin=1).astype(np.int32))
	cuda.memcpy_htod(ntimes_gpu, np.array(ntimes, ndmin=1).astype(np.int32))
	cuda.memcpy_htod(nclmns_gpu, np.array(nclmns, ndmin=1).astype(np.int32))
	cuda.memcpy_htod(detector_tensor_gpu, detector_tensor)


	CTU_gpu = gpuarray.to_gpu(CTU)
	CTV_gpu = gpuarray.to_gpu(CTV)

	# **---- data -----**

	selected_modes_gpu = gpuarray.to_gpu(mlist_sort)
	
	phi_gpu = gpuarray.to_gpu(phi) # RA
	theta_gpu = gpuarray.to_gpu(theta) # DEC
	tref_gpu = gpuarray.to_gpu(tref) # Fiducial Epoch
	phiref_gpu = gpuarray.to_gpu(phiref) # Ref. Oribtal Phase
	inc_gpu = gpuarray.to_gpu(incl) # Inclination
	psi_gpu = gpuarray.to_gpu(psi) # Polarization Angle
	dist_gpu = gpuarray.to_gpu(distance)
	#dist_gpu = gpuarray.to_gpu(distance)*1.e6*lal.PC_SI # Luminosity Distance
	from common_cl import distRef
	#dist_gpu = np.float64(distRef) / dist_gpu

	
	rhoTS_gpu = gpuarray.to_gpu(rhoTS)	

	print("Done passing data to GPU \n")
	
	#______________GET GPU FUNCTIONS ##	


	GPU_compute_sph_harmonics_l_eq_2 = mod.get_function("compute_sph_harmonics_l_eq_2")
	GPU_complex_antenna_factor = mod.get_function("complex_antenna_factor")
	GPU_expand_rhoTS = mod.get_function("expand_rhoTS")
	GPU_insert_ylms = mod.get_function("insert_ylms")
	GPU_accordion = mod.get_function("accordion")
	GPU_make_cpx_3x3_outer_prods = mod.get_function("make_cpx_3x3_outer_prods")
	GPU_make_real_3x3_outer_prods = mod.get_function("make_real_3x3_outer_prods")
	GPU_bcast_vec_to_matrix = mod.get_function("bcast_vec_to_matrix")
	GPU_mul_bcast_vec_to_matrix = mod.get_function("mul_bcast_vec_to_matrix")
	GPU_find_max_in_shrmem = mod.get_function("find_max_in_shrmem")
	GPU_nv_reduc = mod.get_function("nv_reduc")


	#________   ************  ________ #################################################	
	#________|  MAIN ROUTINE |________ #################################################	
	#________|  ************ |________ #################################################

	print("Building Likelihood term 1... \n")

	'''
	Calculate the spherical harmonics
	'''
	spharms_l_eq_2 = np.zeros(nsamps*nmodes).astype(np.complex128)
	spharms_l_eq_2_gpu = gpuarray.to_gpu(spharms_l_eq_2)
	# One thread for each sample, 1D1D
	nblocks = int(nsamps / max_tpb)
	grd = (nblocks, 1, 1)
	blk = (max_tpb, 1, 1)

	GPU_compute_sph_harmonics_l_eq_2(inc_gpu, -phiref_gpu, selected_modes_gpu, spharms_l_eq_2_gpu, grid=grd, block=blk)

	'''
	Calculate F and multiply them correctly into the Ylms
	'''

	complex_antenna_factor = np.zeros(nsamps).astype(np.complex128)
	caf_gpu = gpuarray.to_gpu(complex_antenna_factor)
	# One thread for each sample, 1D1D

	GPU_complex_antenna_factor(phi_gpu, theta_gpu, psi_gpu, tref_gpu, caf_gpu, grid=grd, block=blk, shared=(max_tpb*8*6))


	# Multiply the F's in: sample-wise, same F for each Y
	# Ylms are in row major order with rows corresponding 
	# to modes

	# Conjugate
        spharms_l_eq_2_orig_gpu = spharms_l_eq_2_gpu
	spharms_l_eq_2_gpu = spharms_l_eq_2_gpu.conj()


	for i in range(0, nmodes):
		strt_mode = i*nsamps
		stop_mode = (i+1)*nsamps
		spharms_l_eq_2_gpu[strt_mode:stop_mode] *= caf_gpu.conj()

	'''
	Build the likelihood function
	'''

	# Large memory block containing all rhoTS 
	all_l_rhots = np.zeros((nsamps * nmodes, nclmns)).astype(np.complex128)
	all_l_rhots_gpu = gpuarray.to_gpu(all_l_rhots)
	# Blanket the array with threads

	nblockx = int(nclmns / max_tpb)
	nblocky = int(nsamps * nmodes)
	grd = (nblockx, nblocky, 1)
	blk = (max_tpb, 1,       1)
	GPU_expand_rhoTS(rhoTS_gpu, all_l_rhots_gpu, grid=grd, block=blk)

	'''
	Drop the Ylms into the correct rhoTS rows
	'''

	nblockx = int(nclmns / max_tpb)
	nblocky = int(nsamps * nmodes)
	# Blanket the array with threads 
	grd = (nblockx, nblocky, 1)
	blk = (max_tpb, 1,       1)

	GPU_insert_ylms(all_l_rhots_gpu, spharms_l_eq_2_gpu, grid=grd, block=blk, shared=(int(nmodes*16)))

	'''
	Sum over the the modes to make the first term of the likelihood	
	'''

	nblockx = int(nclmns / max_tpb)
	nblocky = int(nsamps)
	# One thread for each sample-time. One 1/nmodes of all gridpoints
	# Each thread packages nmodes numbers into a sum that is the Lval 
	grd = (nblockx, nblocky, 1)
	blk = (max_tpb, 1,       1)

	GPU_accordion(all_l_rhots_gpu, grid=grd, block=blk)

	# Take the real part
	all_l_rhots_gpu = all_l_rhots_gpu.real

	# Multiply in the distances	
	GPU_mul_bcast_vec_to_matrix(all_l_rhots_gpu, dist_gpu, grid=grd, block=blk, shared=8)



	'''
	Build the 2nd term of the likelihood
	'''

	print("Building likelihood term 2")

	U = np.zeros(nsamps).astype(np.complex128)
	V = np.zeros(nsamps).astype(np.complex128)
	U_gpu = gpuarray.to_gpu(U)
	V_gpu = gpuarray.to_gpu(V)

	# FIXME - these should exist dynamically within constant memory 

	CTU_gpu = gpuarray.to_gpu(CTU)
	CTV_gpu = gpuarray.to_gpu(CTV)

	griddimx = int(nsamps / max_tpb)
	# One thread per sample, each thread builds one U and V crossterm 

	GPU_make_cpx_3x3_outer_prods(CTU_gpu, spharms_l_eq_2_orig_gpu, U_gpu, grid=grd, block=blk, shared=int(16*nmodes*max_tpb))
	GPU_make_real_3x3_outer_prods(CTV_gpu, spharms_l_eq_2_orig_gpu, V_gpu, grid=grd, block=blk, shared=int(16*nmodes*max_tpb))	

	term_two = 0.25*dist_gpu*dist_gpu*(caf_gpu*caf_gpu.conj()*U_gpu + (caf_gpu*caf_gpu*V_gpu).real).real


	'''
	Subtract U and V terms from big block of rhoTS
	'''

	griddimx = int(nclmns / max_tpb)
	griddimy = int(nsamps)
	# One thread per sample-time
	grd = (griddimx, griddimy, 1)
	blk = (max_tpb,  1,        1)

	GPU_bcast_vec_to_matrix(all_l_rhots_gpu, -term_two, grid=grd, block=blk, shared=8)


	def next_greater_power_of_2(x):  
    		return 2**(x-1).bit_length()

	# Get the maxes
	GPU_find_max_in_shrmem(all_l_rhots_gpu, grid=grd, block=blk, shared=int(max_tpb*8))

	griddimy = int(nsamps)
	blokdimx = next_greater_power_of_2(griddimx) # Only need as many threads as we had blocks in x dimension
	grd = (1, griddimy, 1)
	blk = (blokdimx, 1, 1)

	# Second reduction - this works as long as we don't have rhoTS longer then 1024^2
	GPU_find_max_in_shrmem(all_l_rhots_gpu, grid=grd, block=blk, shared=int(blokdimx*8))
	
	# Collect the maxes through the host	
	maxes = np.array(all_l_rhots_gpu[:,0][nmodes-2::nmodes].get()).astype(np.float64)
	maxes_gpu = gpuarray.to_gpu(maxes)
	
	griddimx = int(nclmns / max_tpb)
	griddimy = int(nsamps)
	# One thread per sample-time
	grd = (griddimx, griddimy, 1)
	blk = (max_tpb,  1,        1)

	GPU_bcast_vec_to_matrix(all_l_rhots_gpu, -maxes_gpu, grid=grd, block=blk, shared=8)

	# Exponentiating a bunch of zeros creates a bunch of extra ones that we don't want in our
	# sum, so this is the number we need to subtract out to offset it
	padwidth = nclmns - ntimes


	all_l_rhots_gpu = cumath.exp(all_l_rhots_gpu) # exponentiate 

	GPU_nv_reduc(all_l_rhots_gpu, grid=grd, block=blk, shared=max_tpb*8) # sum over time 

	griddimy = int(nsamps)
	blokdimx = next_greater_power_of_2(griddimx) # Only need as many threads as we had blocks in x dimension
	grd = (1, griddimy, 1)
	blk = (blokdimx, 1, 1)

	GPU_nv_reduc(all_l_rhots_gpu, grid=grd, block=blk, shared=blokdimx*8) # sum over time 

	lnL = (all_l_rhots_gpu[:,0][nmodes-1::nmodes].get() - padwidth).astype(np.float64)
	lnL_gpu = gpuarray.to_gpu(lnL)
	lnL_gpu = maxes_gpu + cumath.log(lnL_gpu)

	return lnL_gpu.get()
