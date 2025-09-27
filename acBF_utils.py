from rigidregistration import stackregistration as stackreg
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
    kx -= tilt[0]
    ky -= tilt[1]
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
def acBF_STEM(tcBF, upscale, ctf):
    im_stack = stack_expand(tcBF._stack_BF_unshifted, upscale)
    tcBF_orig = np.zeros_like(im_stack[0])
    tcBF_corr = np.zeros_like(im_stack[0])
    acBF = np.zeros_like(im_stack[0])
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

        # H. Rose 1976
        ctf_t = 0.5j * (
            ctf.evaluate_aperture(*coords_t)
            * np.exp(-1.0j * (ctf.evaluate_chi(*coords_t) - ctf.evaluate_chi(tr, az)))
            - ctf.evaluate_aperture(*coords_mt)
            * np.exp(1.0j * (ctf.evaluate_chi(*coords_mt) - ctf.evaluate_chi(tr, az)))
        )
        mask = np.logical_or(
            ctf.evaluate_aperture(*coords_t), ctf.evaluate_aperture(*coords_mt)
        )
        # triple_overlap = np.logical_and(ctf.evaluate_aperture(*coords_t), ctf.evaluate_aperture(*coords_mt))

        qxqy = abtem.utils.spatial_frequencies([img.shape[0]],[tcBF._scan_sampling[0]/upscale,])[0]
        dx, dy = d_chi_d_q(ctf, tr, az)
        shift_op = np.exp(-2.0j * np.pi * ((dx * qxqy[:, None]) + (dy * qxqy[None, :])))
        ctf_t *= shift_op

        # tcBF shift correction
        F = np.fft.fft2(np.real(np.fft.ifft2(np.fft.fft2(img) * shift_op)))
        CTF_mean += np.fft.fftshift(ctf_t)
        tcBF_orig += np.real(np.fft.ifft2(F))

        ## acBF ctf correction
        F *= np.exp(-1.0j * np.angle(ctf_t))

        apply_wiener = False
        epsilon = 2e-2
        if apply_wiener:  #Optional Wiener reweighting
            abs_ctf = np.abs(ctf_t)
            F *= abs_ctf / (abs_ctf**2 + epsilon**2)
        acBF += np.real(np.fft.ifft2(F))
    # tcBF sign correction
    tcBF_corr = np.real(np.fft.ifft2(np.fft.fft2(tcBF_orig)*np.sign(np.real(np.fft.fftshift(CTF_mean)))))
    return tcBF_orig, tcBF_corr, acBF
