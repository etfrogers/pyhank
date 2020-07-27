"""
Doc str
"""

import matplotlib.pyplot as plt
import numpy as np

from hankel import HankelTransform, spline, HankelTransformMode


def gauss1d(x, x0, fwhm):
    return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)


def example(plotting=True):

    # Initialise grid
    print('Initialising data ...')
    nr = 1024  # Number of sample points
    r_max = .05  # Maximum radius (5cm)
    dr = r_max / (nr - 1)  # Radial spacing
    ri = np.arange(0, nr)  # Radial pixels
    r = ri * dr  # Radial positions
    Dr = 5e-3  # Beam radius (5mm)
    Kr = 0  # Propagation direction
    Nz = 200  # Number of z positions
    z_max = 0.5  # Maximum propagation distance
    dz = z_max / (Nz - 1)
    z = np.arange(0, Nz) * dz  # Propagation axis

    # Setup Hankel transform structure
    print('Setting up Hankel transform structure ...')
    H = HankelTransform(0, r_max, nr)
    K = 2 * np.pi * H.v_max  # Maximum K vector

    # Generate electric field:
    print('Generating electric field ...')
    Er = gauss1d(r, 0, Dr) * np.exp(1j * Kr * r)  # Initial field
    ErH = spline(r, Er, H.r)  # Resampled field

    # Perform Hankel Transform
    print('Performing Hankel transform ...')
    EkrH = H.qdht(ErH, HankelTransformMode.UNSCALED)  # Convert from physical field to physical wavevector

    # Propagate beam
    print('Propagating beam ...')
    EkrH_ = EkrH / H.JV  # Convert to scaled form for faster transform

    Irz = np.zeros((nr, Nz))
    Irz[:, 0] = np.abs(Er) ** 2
    for n in range(1, Nz):
        z_loop = z[n]
        phiz = (np.sqrt(K ** 2 - H.kr ** 2) - K) * z_loop  # Propagation phase
        EkrHz = EkrH_ * np.exp(1j * phiz)  # Apply propagation
        ErHz = H.iqdht(EkrHz, HankelTransformMode.BOTH_SCALED)  # iQDHT (no scaling)
        Erz = spline(H.r, ErHz * H.JR, r)  # Interpolate & scale output
        Irz[:, n] = np.abs(Erz) ** 2

    Irz_norm = Irz / Irz[0, :]

    if plotting:
        # Plot field
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(r * 1e3, np.abs(Er) ** 2, r * 1e3, np.unwrap(np.angle(Er)),
                 H.r * 1e3, np.abs(ErH) ** 2, H.r * 1e3, np.unwrap(np.angle(ErH)), '+')
        plt.title('Initial electric field distribution')
        plt.xlabel('Radial co-ordinate (r) /mm')
        plt.ylabel('Field intensity /arb.')
        plt.legend(['|E(r)|^2', '\\phi(r)', '|E(H.r)|^2', '\\phi(H.r)'])
        plt.axis([0, 10, 0, 1])

        plt.subplot(2, 1, 2)
        plt.plot(H.kr, np.abs(EkrH) ** 2)
        plt.title('Radial wave-vector distribution')
        plt.xlabel('Radial wave-vector (k_r) /rad m^{-1}')
        plt.ylabel('Field intensity /arb.')
        plt.axis([0, 1e4, 0, np.max(np.abs(EkrH) ** 2)])

        plt.figure(2)
        plt.subplot(2, 1, 1)
        imagesc(z, r * 1e3, Irz)
        plt.title('Radial field intensity as a function of propagation for annular beam')
        plt.ylabel('Radial position (r) /mm')
        plt.ylim([0, 25])
        plt.subplot(2, 1, 2)
        imagesc(z, r * 1e3, Irz_norm)
        plt.xlabel('Propagation distance (z) /m')
        plt.ylabel('Radial position (r) /mm')
        plt.ylim([0, 25])
        plt.show()

        plt.show()

    print('Complete!')


def imagesc(x: np.ndarray, y: np.ndarray, intensity: np.ndarray, axes=None, **kwargs):
    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1d arrays"
    assert intensity.ndim == 2, "Intensity must be a 2d array"
    extent = (x[0], x[-1], y[-1], y[0])
    if axes is None:
        img = plt.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    else:
        img = axes.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    img.axes.invert_yaxis()
    return img


if __name__ == '__main__':
    example()
