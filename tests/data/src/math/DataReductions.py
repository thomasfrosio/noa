# Generate data for noa/tests/cpu/math/TestCPUReductions.h
import os
import numpy as np
import mrcfile


def save_mrc(filename, data):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)


def save_txt(filename, string):
    with open(filename, "w+") as file:
        file.write(string)


def generate_stats():
    batches = 5
    data = np.random.randn(batches, 100, 100).astype(np.float32)
    data += np.linspace(0, 3, batches * 10000).reshape(data.shape)
    save_mrc("./stats_random_array.mrc", data)
    stats = ""
    for batch in range(batches):
        stats += "{}\n".format(np.min(data[batch, ...]))
    for batch in range(batches):
        stats += "{}\n".format(np.max(data[batch, ...]))
    for batch in range(batches):
        stats += "{}\n".format(np.sum(data[batch, ...]))
    for batch in range(batches):
        stats += "{}\n".format(np.mean(data[batch, ...]))
    for batch in range(batches):
        stats += "{}\n".format(np.var(data[batch, ...]))
    for batch in range(batches):
        stats += "{}\n".format(np.std(data[batch, ...]))
    save_txt("./stats_random_array.txt", stats)


def generate_reductions():
    batches = 5
    vectors = np.random.randn(batches, 10, 100).astype(np.float32)  # 5 batches, 10 vectors of 1000 elements to reduce.
    vectors += np.linspace(0, 3, batches * 1000).reshape(vectors.shape)
    save_mrc("./reduction_random_vectors.mrc", vectors)
    save_mrc("./reduction_reduce_add.mrc", np.sum(vectors, axis=1))
    save_mrc("./reduction_reduce_mean.mrc", np.mean(vectors, axis=1))

    weights = np.random.randn(1, 10, 100).astype(np.float32) + 1
    save_mrc("./reduction_random_weights.mrc", weights)
    save_mrc("./reduction_reduce_weighted_mean.mrc", np.sum(vectors * weights, axis=1) / np.sum(weights, axis=1))


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    generate_stats()
    generate_reductions()
