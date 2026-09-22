"""Numerical checks with analytic optics; no microscopy dependencies required."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def frequencies(shape, sampling):
    return tuple(np.fft.fftfreq(n, d) for n, d in zip(shape, sampling))


def polar(ctf, gpts, sampling, tilt):
    qx, qy = frequencies(gpts, sampling)
    x, y = qx[:, None] + tilt[0], qy[None, :] + tilt[1]
    return np.hypot(x, y)[None], np.arctan2(y, x)[None]


class Optics:
    wavelength = 1.0

    def evaluate_chi(self, alpha, phi):
        return 2.3 * alpha**2 + 0.6 * alpha**3 * np.cos(phi)

    def evaluate_aperture(self, alpha, phi):
        return (alpha < 0.85).astype(float)


def load_reconstruction():
    tree = ast.parse((ROOT / 'acBF_utils.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(np=np, tqdm=lambda seq, **kw: seq,
              abtem=SimpleNamespace(utils=SimpleNamespace(spatial_frequencies=frequencies)))
    exec(compile(tree, 'acBF_utils.py', 'exec'), ns)
    ns['_polar_coordinates'] = polar
    ns['d_chi_d_q'] = lambda ctf, tr, az: (0.23, -0.17)
    return ns


def reference(tc, ctf, upscale, regularization=1e-3):
    """Independent per-slice real-space sum and matched-filter estimate."""
    stack = np.repeat(np.repeat(tc._stack_BF_unshifted, upscale, 1), upscale, 2)
    if upscale > 1:
        stack = stack.astype(np.float32)
    shape = stack.shape[1:]
    sampling = tuple(x / upscale for x in tc._scan_sampling)
    qx, qy = frequencies(shape, sampling)
    shift = np.exp(-2j * np.pi * (0.23 * qx[:, None] - 0.17 * qy[None, :]))
    orig = np.zeros(shape)
    phase = np.zeros(shape)
    mean = np.zeros(shape, complex)
    numerator = np.zeros(shape, complex)
    power = np.zeros(shape)
    for k, img in zip(tc._kxy, stack):
        a, p = polar(ctf, shape, sampling, k)
        b, r = polar(ctf, shape, sampling, -k)
        chi0 = ctf.evaluate_chi(np.hypot(*k), np.arctan2(k[1], k[0]))
        transfer = -np.conj(0.5j * (
            ctf.evaluate_aperture(b, r + np.pi) * np.exp(-1j * (chi0 - ctf.evaluate_chi(b, r + np.pi)))
            - ctf.evaluate_aperture(a, p) * np.exp(1j * (chi0 - ctf.evaluate_chi(a, p)))))
        transfer = transfer.reshape(shape)
        raw = np.fft.fft2(img)
        shifted = np.fft.ifft2(raw * shift).real
        orig += shifted
        phase += np.fft.ifft2(np.fft.fft2(shifted) * np.exp(-1j * np.angle(transfer * shift))).real
        mean += transfer * shift
        # Unshifted data/transfer must yield the same inversion.
        numerator += transfer.conj() * raw
        power += abs(transfer)**2
    corrected = np.fft.ifft2(np.fft.fft2(orig) * np.sign(mean.real)).real
    positive = np.sort(power[power > 0])
    ref = positive[(len(positive) - 1) // 2] if len(positive) else 1.0
    estimate = np.zeros(shape, complex)
    np.divide(numerator, power + regularization * ref, out=estimate, where=power > 1e-6 * ref)
    return orig, corrected, phase, np.fft.ifft2(estimate).real


class ReconstructionTests(unittest.TestCase):
    def test_reconstruction_and_fft_count(self):
        for shape, upscale in [((8, 8), 1), ((7, 9), 1), ((4, 6), 2)]:
            with self.subTest(shape=shape, upscale=upscale):
                tc = SimpleNamespace(
                    _stack_BF_unshifted=np.random.default_rng(4).normal(size=(3, *shape)),
                    _kxy=np.array([[0.1, 0.2], [-0.2, 0.1], [0.0, -0.1]]),
                    _scan_sampling=(0.8, 1.1), rotation_Q_to_R_rads=0, transpose=False)
                ns = load_reconstruction()
                expected = reference(tc, Optics(), upscale)
                from unittest.mock import patch
                with patch.object(np.fft, 'ifft2', wraps=np.fft.ifft2) as inverse:
                    actual = ns['acBF_STEM'](tc, upscale, Optics(), return_complex=True)
                    self.assertEqual(inverse.call_count, 4)
                for result, target in zip(actual, expected):
                    np.testing.assert_allclose(result, target, atol=1e-6)
                self.assertEqual(len(ns['acBF_STEM'](tc, upscale, Optics())), 3)

    def test_zero_support_and_invalid_regularization(self):
        tc = SimpleNamespace(_stack_BF_unshifted=np.ones((2, 6, 6)),
                             _kxy=np.zeros((2, 2)), _scan_sampling=(1, 1),
                             rotation_Q_to_R_rads=0, transpose=False)
        ctf = Optics()
        ctf.evaluate_aperture = lambda a, p: np.zeros_like(a)
        fn = load_reconstruction()['acBF_STEM']
        with np.errstate(all='raise'):
            result = fn(tc, 1, ctf, return_complex=True, regularization=0)[3]
        np.testing.assert_array_equal(result, 0)
        for keyword in ['regularization', 'support_threshold']:
            for bad in [-1, float('nan'), float('inf')]:
                with self.assertRaises(ValueError):
                    fn(tc, 1, ctf, **{keyword: bad})


if __name__ == '__main__':
    unittest.main()
