# Generate data for noa/tests/cpu/fourier/TestResize.h
import os
import numpy as np
import mrcfile


def save_mrc(filename, data):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)


def get_random_large():
    return np.random.randint(50, 64)


def get_random_small():
    return np.random.randint(2, 20)


def generate_crop():
    # 2D:
    yold, xold = get_random_large(), get_random_large()
    y, x = yold - get_random_small(), xold - get_random_small()
    before = np.arange(0, xold * yold, dtype=np.float32).reshape((yold, xold))
    after = np.zeros((y, x), dtype=np.float32)
    after[0:y // 2, :] = before[0:y // 2, 0:x]
    after[y // 2::, :] = before[y // 2 + yold - y::, 0:x]
    save_mrc("./crop_2D_before.mrc", before)
    save_mrc("./crop_2D_after.mrc", after)

    # 3D:
    zold, yold, xold = get_random_large(), get_random_large(), get_random_large()
    z, y, x = zold - get_random_small(), yold - get_random_small(), xold - get_random_small()
    before = np.arange(0, xold * yold * zold, dtype=np.float32).reshape((zold, yold, xold))
    after = np.zeros((z, y, x), dtype=np.float32)
    after[0:z // 2, 0:y // 2, :] = before[0:z // 2, 0:y // 2, 0:x]
    after[0:z // 2, y // 2::, :] = before[0:z // 2, y // 2 + yold - y::, 0:x]
    after[z // 2::, 0:y // 2, :] = before[z // 2 + zold - z::, 0:y // 2, 0:x]
    after[z // 2::, y // 2::, :] = before[z // 2 + zold - z::, y // 2 + yold - y::, 0:x]
    save_mrc("./crop_3D_before.mrc", before)
    save_mrc("./crop_3D_after.mrc", after)


def generate_crop_full():
    # 2D: 128x128 to 64x64 logical size
    yold, xold = get_random_large(), get_random_large()
    y, x = yold - get_random_small(), xold - get_random_small()
    before = np.arange(0, xold * yold, dtype=np.float32).reshape((yold, xold))
    after = np.zeros((y, x), dtype=np.float32)
    after[0:y // 2, 0:x // 2] = before[0:y // 2, 0:x // 2]
    after[0:y // 2, x // 2::] = before[0:y // 2, x // 2 + xold - x::]
    after[y // 2::, 0:x // 2] = before[y // 2 + yold - y::, 0:x // 2]
    after[y // 2::, x // 2::] = before[y // 2 + yold - y::, x // 2 + xold - x::]
    save_mrc("./crop_full_2D_before.mrc", before)
    save_mrc("./crop_full_2D_after.mrc", after)

    # 3D: 11x10x9 to 8x8x8 logical size
    zold, yold, xold = get_random_large(), get_random_large(), get_random_large()
    z, y, x = zold - get_random_small(), yold - get_random_small(), xold - get_random_small()
    before = np.arange(0, xold * yold * zold, dtype=np.float32).reshape((zold, yold, xold))
    after = np.zeros((z, y, x), dtype=np.float32)
    after[0:z // 2, 0:y // 2, 0:x // 2] = before[0:z // 2, 0:y // 2, 0:x // 2]
    after[0:z // 2, 0:y // 2, x // 2::] = before[0:z // 2, 0:y // 2, x // 2 + xold - x::]
    after[0:z // 2, y // 2::, 0:x // 2] = before[0:z // 2, y // 2 + yold - y::, 0:x // 2]
    after[0:z // 2, y // 2::, x // 2::] = before[0:z // 2, y // 2 + yold - y::, x // 2 + xold - x::]
    after[z // 2::, 0:y // 2, 0:x // 2] = before[z // 2 + zold - z::, 0:y // 2, 0:x // 2]
    after[z // 2::, 0:y // 2, x // 2::] = before[z // 2 + zold - z::, 0:y // 2, x // 2 + xold - x::]
    after[z // 2::, y // 2::, 0:x // 2] = before[z // 2 + zold - z::, y // 2 + yold - y::, 0:x // 2]
    after[z // 2::, y // 2::, x // 2::] = before[z // 2 + zold - z::, y // 2 + yold - y::, x // 2 + xold - x::]
    save_mrc("./crop_full_3D_before.mrc", before)
    save_mrc("./crop_full_3D_after.mrc", after)


def generate_pad():
    # 2D:
    yold, xold = get_random_large(), get_random_large()
    y, x = yold + get_random_small(), xold + get_random_small()
    before = np.arange(0, xold * yold, dtype=np.float32).reshape((yold, xold))
    after = np.zeros((y, x), dtype=np.float32)

    yhalf = (yold + 1) // 2
    yoffset = y - yold

    after[0:yhalf, 0:xold] = before[0:yhalf, :]
    after[yoffset + yhalf::, 0:xold] = before[yhalf::, :]
    save_mrc("./pad_2D_before.mrc", before)
    save_mrc("./pad_2D_after.mrc", after)

    # 3D:
    zold, yold, xold = get_random_large(), get_random_large(), get_random_large()
    z, y, x = zold + get_random_small(), yold + get_random_small(), xold + get_random_small()
    before = np.arange(0, xold * yold * zold, dtype=np.float32).reshape((zold, yold, xold))
    after = np.zeros((z, y, x), dtype=np.float32)

    zhalf, yhalf = (zold + 1) // 2, (yold + 1) // 2
    zoffset, yoffset = z - zold, y - yold

    after[0:zhalf, 0:yhalf, 0:xold] = before[0:zhalf, 0:yhalf, :]
    after[0:zhalf, yoffset + yhalf::, 0:xold] = before[0:zhalf, yhalf::, :]
    after[zoffset + zhalf::, 0:yhalf, 0:xold] = before[zhalf::, 0:yhalf, :]
    after[zoffset + zhalf::, yoffset + yhalf::, 0:xold] = before[zhalf::, yhalf::, :]
    save_mrc("./pad_3D_before.mrc", before)
    save_mrc("./pad_3D_after.mrc", after)


def generate_pad_full():
    # 2D:
    yold, xold = get_random_large(), get_random_large()
    y, x = yold + get_random_small(), xold + get_random_small()
    before = np.arange(0, xold * yold, dtype=np.float32).reshape((yold, xold))
    after = np.zeros((y, x), dtype=np.float32)

    yhalf, xhalf = (yold + 1) // 2, (xold + 1) // 2
    yoffset, xoffset = y - yold, x - xold

    after[0:yhalf, 0:xhalf] = before[0:yhalf, 0:xhalf]
    after[0:yhalf, xoffset + xhalf::] = before[0:yhalf, xhalf::]
    after[yoffset + yhalf::, 0:xhalf] = before[yhalf::, 0:xhalf]
    after[yoffset + yhalf::, xoffset + xhalf::] = before[yhalf::, xhalf::]
    save_mrc("./pad_full_2D_before.mrc", before)
    save_mrc("./pad_full_2D_after.mrc", after)

    # 3D:
    zold, yold, xold = get_random_large(), get_random_large(), get_random_large()
    z, y, x = zold + get_random_small(), yold + get_random_small(), xold + get_random_small()
    before = np.arange(0, xold * yold * zold, dtype=np.float32).reshape((zold, yold, xold))
    after = np.zeros((z, y, x), dtype=np.float32)

    zhalf, yhalf, xhalf = (zold + 1) // 2, (yold + 1) // 2, (xold + 1) // 2
    zoffset, yoffset, xoffset = z - zold, y - yold, x - xold

    after[0:zhalf, 0:yhalf, 0:xhalf] = before[0:zhalf, 0:yhalf, 0:xhalf]
    after[0:zhalf, 0:yhalf, xoffset + xhalf::] = before[0:zhalf, 0:yhalf, xhalf::]
    after[0:zhalf, yoffset + yhalf::, 0:xhalf] = before[0:zhalf, yhalf::, 0:xhalf]
    after[0:zhalf, yoffset + yhalf::, xoffset + xhalf::] = before[0:zhalf, yhalf::, xhalf::]
    after[zoffset + zhalf::, 0:yhalf, 0:xhalf] = before[zhalf::, 0:yhalf, 0:xhalf]
    after[zoffset + zhalf::, 0:yhalf, xoffset + xhalf::] = before[zhalf::, 0:yhalf, xhalf::]
    after[zoffset + zhalf::, yoffset + yhalf::, 0:xhalf] = before[zhalf::, yhalf::, 0:xhalf]
    after[zoffset + zhalf::, yoffset + yhalf::, xoffset + xhalf::] = before[zhalf::, yhalf::, xhalf::]
    save_mrc("./pad_full_3D_before.mrc", before)
    save_mrc("./pad_full_3D_after.mrc", after)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # generate_crop()
    # generate_crop_full()
    # generate_pad()
    # generate_pad_full()
