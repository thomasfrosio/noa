# Generate data for noa/tests/cpu/fourier/TestResize.h
import os
import numpy as np
import mrcfile


def save_mrc(filename, data):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)


def generate_fftshift():
    # 3D
    shape = np.random.randint(2, 128, size=3)
    array = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    save_mrc("array_3D.mrc", array)
    save_mrc("array_fftshift_3D.mrc", np.fft.fftshift(array))
    save_mrc("array_ifftshift_3D.mrc", np.fft.ifftshift(array))

    # 2D
    shape = np.random.randint(2, 128, size=2)
    array = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    save_mrc("array_2D.mrc", array)
    save_mrc("array_fftshift_2D.mrc", np.fft.fftshift(array))
    save_mrc("array_ifftshift_2D.mrc", np.fft.ifftshift(array))


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    generate_fftshift()
