import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def plot_sample():

    for i in range(3, 30, 3):
        ixi_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/ixi/Wed Feb 23 22:02:28 2022/sample_distribution/epoch_{}.npy'.format(i)
        data = np.load(ixi_path, allow_pickle=True)
        plt.figure(figsize=(5, 4))
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc=1)
        plt.title('{}: {}'.format(descript, step))
        plt.savefig(img_path)
        plt.close()


if __name__ == '__main__':
    plot_sample()