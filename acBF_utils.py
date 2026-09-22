import math
import py4DSTEM
import numpy as np
from tqdm import tqdm
import abtem


def interpolation(img,scale,method='nn'):
    ## The default here is 'nn' because the zero padding comes from 'shiftfunc' with the tile array
    ## if you tried running 'apply_shift_expand' with method = 'null', the normalization wont be right and you'd end up with artifacts
    x,y = np.shape(img)

    if method=='null':
        up = np.zeros((x*scale,y*scale))
        up[0::scale, 0::scale] = img[0::1, 0::1]
        
    if method=='nn':
        up = np.repeat(np.repeat(img, scale, axis=0), scale,axis=1)
            
    return up


def stack_expand(stack, scale):
    z,x,y = np.shape(stack)
    up_stack = np.ones((z,x*scale,y*scale), dtype=np.float32)

    if scale>1:
        for i in range(z):
            up_stack[i,:,:] = interpolation(stack[i,:,:],scale)
    else:
        up_stack = stack
    
    return up_stack


## acBF ctf correction
def _polar_coordinates(
    self, gpts=None, extent=None, sampling=None, xp=np, tilt=[0.0, 0.0]
):
    # tilt is in A^-1, return is in radians
    grid = abtem.base_classes.Grid(gpts=gpts, extent=extent, sampling=sampling)

    gpts = grid.gpts
    sampling = grid.sampling

    kx, ky = abtem.utils.spatial_frequencies(gpts, sampling)
    kx += tilt[0]
    ky += tilt[1]
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    kx = xp.asarray(kx)
    ky = xp.asarray(ky)
    return abtem.utils.polar_coordinates(
        xp.asarray(kx * self.wavelength), xp.asarray(ky * self.wavelength)


    )

def d_chi_d_q(self, alpha, phi):
    xp = np
    p = self.parameters
    dchi_dk = 1.0 * (
        (p["C12"] * xp.cos(2.0 * (phi - p["phi12"])) + p["C10"]) * alpha
        + (
            p["C23"] * xp.cos(3.0 * (phi - p["phi23"]))
            + p["C21"] * xp.cos(1.0 * (phi - p["phi21"]))
        )
        * alpha**2
        + (
            p["C34"] * xp.cos(4.0 * (phi - p["phi34"]))
            + p["C32"] * xp.cos(2.0 * (phi - p["phi32"]))
            + p["C30"]
        )
        * alpha**3
        + (
            p["C45"] * xp.cos(5.0 * (phi - p["phi45"]))
            + p["C43"] * xp.cos(3.0 * (phi - p["phi43"]))
            + p["C41"] * xp.cos(1.0 * (phi - p["phi41"]))
        )
        * alpha**4
        + (
            p["C56"] * xp.cos(6.0 * (phi - p["phi56"]))
            + p["C54"] * xp.cos(4.0 * (phi - p["phi54"]))
            + p["C52"] * xp.cos(2.0 * (phi - p["phi52"]))
            + p["C50"]
        )
        * alpha**5
    )

    dchi_dphi = -1.0 * (
        1 / 2.0 * (2.0 * p["C12"] * xp.sin(2.0 * (phi - p["phi12"]))) * alpha
        + 1
        / 3.0
        * (
            3.0 * p["C23"] * xp.sin(3.0 * (phi - p["phi23"]))
            + 1.0 * p["C21"] * xp.sin(1.0 * (phi - p["phi21"]))
        )
        * alpha**2
        + 1
        / 4.0
        * (
            4.0 * p["C34"] * xp.sin(4.0 * (phi - p["phi34"]))
            + 2.0 * p["C32"] * xp.sin(2.0 * (phi - p["phi32"]))
        )
        * alpha**3
        + 1
        / 5.0
        * (
            5.0 * p["C45"] * xp.sin(5.0 * (phi - p["phi45"]))
            + 3.0 * p["C43"] * xp.sin(3.0 * (phi - p["phi43"]))
            + 1.0 * p["C41"] * xp.sin(1.0 * (phi - p["phi41"]))
        )
        * alpha**4
        + 1
        / 6.0
        * (
            6.0 * p["C56"] * xp.sin(6.0 * (phi - p["phi56"]))
            + 4.0 * p["C54"] * xp.sin(4.0 * (phi - p["phi54"]))
            + 2.0 * p["C52"] * xp.sin(2.0 * (phi - p["phi52"]))
        )
        * alpha**5
    )

    dchi_dx = xp.cos(phi) * dchi_dk - xp.sin(phi) * dchi_dphi
    dchi_dy = xp.sin(phi) * dchi_dk + xp.cos(phi) * dchi_dphi

    return dchi_dx, dchi_dy

    

##
##
def acBF_STEM(tcBF, upscale, ctf, *, return_complex=False,
              regularization=1e-3, support_threshold=1e-6):
    """Return tcBF, sign-corrected tcBF, and phase-only acBF.

    With return_complex=True, append the real image from complex inversion.
    In this package's notation, the regularized Fourier estimate is
    sum(conj(ctf_t) * F) / (sum(abs(ctf_t)**2) + regularization * CTF_ref),
    where F and ctf_t have the same shift correction. CTF_ref is the lower
    median of positive transfer power (matching fast-acbf's torch.median).
    Frequencies below support_threshold * CTF_ref are set to zero.
    Complex inversion corrects amplitude and phase; its scale differs from
    the phase-only sum. Defaults match fast-acbf's inversion settings.
    """
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    if not np.isfinite(support_threshold) or support_threshold < 0:
        raise ValueError("support_threshold must be finite and non-negative")
    im_stack = stack_expand(tcBF._stack_BF_unshifted, upscale)
    F_tcBF = np.zeros(im_stack[0].shape, dtype=np.complex128)
    F_acBF = np.zeros_like(F_tcBF)
    if return_complex:
        F_acBF_complex = np.zeros_like(F_tcBF)
        CTF_power = np.zeros(im_stack[0].shape, dtype=np.float64)
    # Fourier indices for the Hermitian projection FFT(real(iFFT(F))).
    neg_x = (-np.arange(im_stack.shape[1])) % im_stack.shape[1]
    neg_y = (-np.arange(im_stack.shape[2])) % im_stack.shape[2]
    CTF_mean = np.zeros(im_stack[0].shape,dtype=np.complex128)

    R = np.array(
        [
            [np.cos(tcBF.rotation_Q_to_R_rads), -np.sin(tcBF.rotation_Q_to_R_rads)],
            [np.sin(tcBF.rotation_Q_to_R_rads), np.cos(tcBF.rotation_Q_to_R_rads)],
        ],
    )

    if tcBF.transpose:
        R = np.array([[0, 1], [1, 0]]) @ R

    for kxy, img in tqdm(zip(tcBF._kxy @ R.T, im_stack),total=tcBF._kxy.shape[0]):
        coords_t = _polar_coordinates(ctf, gpts=im_stack.shape[1:], sampling=tuple(x / upscale for x in tcBF._scan_sampling), tilt=kxy)
        coords_mt = _polar_coordinates(ctf, gpts=im_stack.shape[1:], sampling=tuple(x / upscale for x in tcBF._scan_sampling), tilt=-kxy)

        tr = np.hypot(*kxy) * ctf.wavelength
        az = np.arctan2(kxy[1], kxy[0])

        # corrected Rose Ultramic 1977, Eq 33 (see Ma et al arXiv preprint arXiv:2510.01493.):
        ctf_t = 0.5j * (
            ctf.evaluate_aperture(coords_mt[0],coords_mt[1]+np.pi) * np.exp(-1.0j * (ctf.evaluate_chi(tr,az)-ctf.evaluate_chi(coords_mt[0],coords_mt[1]+np.pi)  ))
            - ctf.evaluate_aperture(coords_t[0],coords_t[1]) * np.exp(1.0j * (ctf.evaluate_chi(tr,az)-ctf.evaluate_chi(coords_t[0],coords_t[1])))
        ) 
        ctf_t = -np.conj(ctf_t)
        mask = np.logical_or(
            ctf.evaluate_aperture(*coords_t), ctf.evaluate_aperture(*coords_mt)
        )
        # triple_overlap = np.logical_and(ctf.evaluate_aperture(*coords_t), ctf.evaluate_aperture(*coords_mt))

        qx, qy = abtem.utils.spatial_frequencies(
            img.shape, tuple(x / upscale for x in tcBF._scan_sampling)
        )
        dx, dy = d_chi_d_q(ctf, tr, az)
        shift_op = np.exp(-2.0j * np.pi * (dx * qx[:, None] + dy * qy[None, :]))
        ctf_t = np.asarray(ctf_t * shift_op).reshape(img.shape)

        F = np.fft.fft2(img) * shift_op
        # Linearity: sum_k real(iFFT(F_k)) = real(iFFT(sum_k F_k)).
        # Sum the shifted slice spectra first, then use one iFFT after the
        # loop rather than one iFFT for every detector slice.
        F_tcBF += F
        CTF_mean += ctf_t

        if return_complex:
            # Matched-filter inversion as in fast-acbf, using this package's
            # forward-transfer convention: conjugate(ctf_t) corrects phase.
            # Apply the same shift to data and transfer, so it cancels in
            # conjugate(ctf_t) * F. Accumulate before dividing by CTF power.
            F_acBF_complex += np.conj(ctf_t) * F
            CTF_power += np.abs(ctf_t)**2

        # Preserve the old phase-only path's real-image projection without
        # its iFFT/FFT round trip (including Nyquist bins for even shapes).
        F_real = 0.5 * (F + np.conj(F[np.ix_(neg_x, neg_y)]))
        F_acBF += F_real * np.exp(-1.0j * np.angle(ctf_t))

    tcBF_orig = np.real(np.fft.ifft2(F_tcBF))
    F_tcBF_real = 0.5 * (F_tcBF + np.conj(F_tcBF[np.ix_(neg_x, neg_y)]))
    # CTF_mean stays in native FFT order, including for odd-sized images.
    tcBF_corr = np.real(np.fft.ifft2(F_tcBF_real * np.sign(np.real(CTF_mean))))
    acBF = np.real(np.fft.ifft2(F_acBF))
    if not return_complex:
        return tcBF_orig, tcBF_corr, acBF

    positive_power = CTF_power[CTF_power > 0]
    if positive_power.size:
        median_index = (positive_power.size - 1) // 2
        CTF_ref = np.partition(positive_power, median_index)[median_index]
    else:
        CTF_ref = 1.0
    support = CTF_power > support_threshold * CTF_ref
    denominator = CTF_power + regularization * CTF_ref
    F_inverted = np.zeros_like(F_acBF_complex)
    np.divide(F_acBF_complex, denominator, out=F_inverted, where=support)
    acBF_complex = np.real(np.fft.ifft2(F_inverted))
    return tcBF_orig, tcBF_corr, acBF, acBF_complex
