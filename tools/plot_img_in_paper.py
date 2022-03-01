from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

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

    plt.figure(figsize=(5*5, 4))
    for i, j in enumerate([3, 6, 9, 15, 30]):
        input_img = '{}/sample_distribution/epoch_{}.npy'.format(input_path, j)
        real_a, fake_a, real_b, fake_b = np.load(input_img, allow_pickle=True)
        descript = 'Epoch'
        step = j

        plt.subplot(1, 5, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        # plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        # plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc='lower left', fontsize=12, markerscale=5)
        plt.title('{}: {}'.format(descript, step), fontdict={'weight':'normal','size':20})

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def plot_multi_imgs_federated_in_paper(input_path, output_path):
    plt.figure(figsize=(5*5, 4))
    for i, j in enumerate([1, 2, 3, 5, 10]):
        input_img = '{}/sample_distribution/round_{}.npy'.format(input_path, j)
        real_a, fake_a, real_b, fake_b = np.load(input_img, allow_pickle=True)
        descript = 'Round'
        step = j

        plt.subplot(1, 5, i+1)
        plt.scatter(real_a[:400, 0], real_a[:400, 1], color='b', label='Real A', s=1, alpha=0.5)
        plt.scatter(fake_a[:400, 0], fake_a[:400, 1], color='r', label='Fake A', s=1, alpha=0.5)
        # plt.scatter(real_b[:400, 0], real_b[:400, 1], color='k', label='Real B', s=1, alpha=0.5)
        # plt.scatter(fake_b[:400, 0], fake_b[:400, 1], color='g', label='Fake B', s=1, alpha=0.5)

        plt.xlim((0.1, 0.4))
        if i == 0:
            plt.xlim(0.2, 0.5)

        plt.ylim((0.1, 0.4))
        plt.grid()
        plt.legend(loc='lower left', fontsize=12, markerscale=5)
        plt.title('{}: {}'.format(descript, step), fontdict={'weight':'normal','size':20})

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()



def cyclegan_ixi_pd_t2_fed_avg_no_dp():

    plt.figure(figsize=(7, 4))

    xtick = 10
    x = range(1, xtick+1, 1)
    nonfed = [25.88, 25.19, 24.98, 24.46, 22.93, 23.95, 20.18, 21.16, 22.50, 19.42]
    server = [19.02, 25.40, 26.19, 26.73, 26.81, 26.76, 27.04, 26.93, 26.78, 26.51]

    client1 = [23.95, 23.92, 25.44, 25.42, 24.01, 24.02, 25.69, 24.56, 23.52, 25.45]
    client2 = [23.95, 24.29, 25.18, 25.97, 25.87, 26.37, 25.09, 24.54, 26.14, 24.03]
    client3 = [23.32, 24.01, 25.46, 26.41, 26.48, 26.55, 26.70, 26.97, 26.93, 26.96]
    client4 = [22.87, 23.71, 24.43, 24.84, 25.61, 25.58, 25.98, 25.85, 26.29, 24.76]


    plt.plot(x, nonfed[:xtick], color='black', linewidth=1, linestyle='-', marker='.', label='Non-Fed')
    plt.plot(x, server[:xtick], color='red', linewidth=1, alpha=1, linestyle='-', marker='.', label='Server')


    plt.plot(x, client1[:xtick], color='blue', linewidth=1, alpha=0.4+0.5, linestyle='-', marker='.', label='Client1(0.4)')
    plt.plot(x, client2[:xtick], color='blue', linewidth=1, alpha=0.3+0.4, linestyle='-.', marker='.', label='Client2(0.3)')
    plt.plot(x, client3[:xtick], color='blue', linewidth=1, alpha=0.2+0.3, linestyle='dashed', marker='.', label='Client3(0.2)')
    plt.plot(x, client4[:xtick], color='blue', linewidth=1, alpha=0.1+0.2, linestyle='dotted', marker='.', label='Client4(0.1)')

    #plt.xlim(0, xtick)
    plt.ylim(8, 28)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('PSNR', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))

    #plt.title('Fed-Avg ', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./legacy_code/cyclegan_ixi_pd_t2_fed_avg_no_dp.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def cyclegan_ixi_pd_t2_fed_avg():

    plt.figure(figsize=(5, 4))

    xtick = 10
    x = range(1, xtick+1, 1)
    nonfed = [25.88, 25.19, 24.98, 24.46, 22.93, 23.95, 20.18, 21.16, 22.50, 19.42]
    server = [19.99, 25.28, 26.34, 27.09, 27.51, 27.73, 27.87, 27.29, 27.44, 26.22]

    client1 = [24.11, 25.17, 26.00, 25.96, 26.32, 27.01, 26.79, 24.47, 24.29, 21.21]
    client2 = [23.96, 24.59, 25.30, 26.19, 26.22, 26.74, 27.16, 27.16, 27.16, 27.21]
    client3 = [23.77, 23.72, 24.50, 26.25, 26.67, 27.26, 26.54, 26.64, 26.53, 27.11]
    client4 = [23.01, 22.68, 25.28, 25.43, 26.15, 26.56, 26.53, 26.66, 26.78, 26.97]


    plt.plot(x, nonfed[:xtick], color='black', linewidth=1, linestyle='-', label='Non-Fed')
    plt.plot(x, server[:xtick], color='red', linewidth=1, alpha=1, linestyle='-', marker='.', label='Server')


    plt.plot(x, client1[:xtick], color='blue', linewidth=1, alpha=0.4+0.5, linestyle='-', label='Client1(0.4)')
    plt.plot(x, client2[:xtick], color='blue', linewidth=1, alpha=0.3+0.4, linestyle='-.', label='Client2(0.3)')
    plt.plot(x, client3[:xtick], color='blue', linewidth=1, alpha=0.2+0.3, linestyle='dashed', label='Client3(0.2)')
    plt.plot(x, client4[:xtick], color='blue', linewidth=1, alpha=0.1+0.2, linestyle='dotted', label='Client4(0.1)')

    #plt.xlim(0, xtick)
    plt.ylim(12, 28)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('PSNR', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))

    #plt.title('Fed-Avg ', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./legacy_code/cyclegan_ixi_pd_t2_fed_avg.pdf', bbox_inches='tight', pad_inches=0.1)
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
    plot_multi_imgs_centralized_in_paper(ixi_central_dp, img_path)
    
    ixi_federate = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/ixi/pd-t2'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_federate_pd_t2.png'
    plot_multi_imgs_federated_in_paper(ixi_federate, img_path)

    ixi_federate_no_dp = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/ixi/pd-t2-no-dp'
    img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/ixi/ixi_federate_pd_t2-no-dp.png'
    plot_multi_imgs_federated_in_paper(ixi_federate_no_dp, img_path)

    # brats2021
    # brats_central = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/brats2021/t1-flair'
    # img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/brats2021/brats_central_t1_flair.png'
    # plot_multi_imgs_centralized_in_paper(brats_central, img_path)

    # brats_central_dp = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/centralized/brats2021/t1-flair-dp'
    # img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/brats2021/brats_centeral_t1_flair_dp.png'
    # plot_multi_imgs_centralized_in_paper(brats_central_dp, img_path)
    
    # brats_federate = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/brats2021/t1-flair'
    # img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/brats2021/brats_federate_t1_flair.png'
    # plot_multi_imgs_federated_in_paper(brats_federate, img_path)

    # brats_federate_no_dp = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/work_dir/federated/brats2021/t1-flair-no-dp'
    # img_path = '/home/xgy/jb-wang/M-3LAB/FedMed-GAN/legacy_code/vis_sample/brats2021/brats_federate_t1_flair-no-dp.png'
    # plot_multi_imgs_federated_in_paper(brats_federate_no_dp, img_path)

    # cyclegan_ixi_pd_t2_fed_avg()
    # cyclegan_ixi_pd_t2_fed_avg_no_dp()