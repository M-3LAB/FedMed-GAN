from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_feature(real_a, fake_a, real_b, fake_b, descript, img_path, step):

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

def plot_multi_imgs_ixi_centralized():

    plt.figure(figsize=(5*9, 4))
    for i, j in enumerate(range(3, 31, 3)):
        ixi_centralized_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/ixi/t2-pd/sample_distribution/epoch_{}.npy'.format(j)
        real_a, fake_a, real_b, fake_b = np.load(ixi_centralized_path, allow_pickle=True)
        descript = 'Step'
        step = i+1

        plt.subplot(1, 10, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc=1, fontsize=8)
        plt.title('{}: {}'.format(descript, step))

    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/centralized.pdf'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def plot_multi_imgs_ixi_federated():

    plt.figure(figsize=(5*9, 4))
    for i, j in enumerate(range(1, 11, 1)):
        ixi_centralized_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/brats2021/Wed Feb 23 22:03:53 2022/sample_distribution/round_{}.npy'.format(j)
        real_a, fake_a, real_b, fake_b = np.load(ixi_centralized_path, allow_pickle=True)
        descript = 'Step'
        step = i+1

        plt.subplot(1, 10, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc=1, fontsize=8)
        plt.title('{}: {}'.format(descript, step))

    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/federated.pdf'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
        

def plot_sample():
    for i in range(1, 31, 1):
        ixi_centralized_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/ixi/Thu Feb 24 12:03:11 2022/sample_distribution/epoch_{}.npy'.format(i)
        real_a, fake_a, real_b, fake_b = np.load(ixi_centralized_path, allow_pickle=True)
        descript = 'Step'
        step = i
        img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/centralized/epoch_{}.png'.format(step)

        plot_feature(real_a, fake_a, real_b, fake_b, descript, img_path, step)

    for i in range(1, 11, 1):
        ixi_centralized_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/brats2021/Wed Feb 23 22:03:53 2022/sample_distribution/round_{}.npy'.format(i)
        real_a, fake_a, real_b, fake_b = np.load(ixi_centralized_path, allow_pickle=True)
        descript = 'Step'
        step = i
        img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/federated/round_{}.png'.format(step)

        plot_feature(real_a, fake_a, real_b, fake_b, descript, img_path, step)


def plot_multi_imgs_centralized_in_paper(input_path, output_path):

    plt.figure(figsize=(5*6, 4))
    for i, j in enumerate([3, 6,  9, 12, 15, 30]):
        input_img = '{}/sample_distribution/epoch_{}.npy'.format(input_path, j)
        real_a, fake_a, real_b, fake_b = np.load(input_img, allow_pickle=True)
        descript = 'Step'
        step = i+1

        plt.subplot(1, 6, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        if j == 30:
            step = 10
        plt.xlim((0.1, 0.4))
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc=1, fontsize=8)
        plt.title('{}: {}'.format(descript, step))

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


def plot_multi_imgs_federated_in_paper(input_path, output_path):
    plt.figure(figsize=(5*6, 4))
    for i, j in enumerate([1, 2, 3, 4, 5, 10]):
        input_img = '{}/sample_distribution/round_{}.npy'.format(input_path, j)
        real_a, fake_a, real_b, fake_b = np.load(input_img, allow_pickle=True)
        descript = 'Step'
        step = i+1

        plt.subplot(1, 6, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        if i == 0:
            plt.xlim(0.2, 0.5)
        if j == 10:
            step = 10 
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc=1, fontsize=8)
        plt.title('{}: {}'.format(descript, step))

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

if __name__ == '__main__':
    # plot_sample()
    # plot_multi_imgs_ixi_centralized()
    # plot_multi_imgs_ixi_federated()

    ixi_central = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/ixi/pd-t2'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_central_pd_t2.png'
    plot_multi_imgs_centralized_in_paper(ixi_central, img_path)

    ixi_central_dp = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/ixi/pd-t2-dp'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_centeral_pd_t2_dp.png'
    plot_multi_imgs_centralized_in_paper(ixi_central, img_path)
    
    ixi_federate = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/ixi/pd-t2'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_federate_pd_t2.png'
    plot_multi_imgs_federated_in_paper(ixi_federate, img_path)

    ixi_federate_no_dp = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/ixi/pd-t2-no-dp'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_federate_pd_t2-no-dp.png'
    plot_multi_imgs_federated_in_paper(ixi_federate_no_dp, img_path)

