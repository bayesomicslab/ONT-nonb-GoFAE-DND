import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import os


def trace_plots(x_true, x_pred, obs, titlestring, path2save):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.title(titlestring, fontsize=15)
    plt.plot(x_true,label='True')
    plt.plot(x_pred, label='Reconstruction')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(prop={'size': 15})
    plt.savefig(path2save,dpi=200,bbox_inches='tight')
    plt.close(fig)


def plot_test(bdna_mh, bdna_rec, nonb_mh, nonb_rec, idxs, img_path, selection, fdr_level):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(bdna_mh, bdna_rec, s=2, alpha=.5, color='#1f77b4', label='BDNA')
    plt.scatter(nonb_mh, nonb_rec, s=2, alpha=.5, color='#ff7f0e', label='NonB')
    plt.plot(nonb_mh[idxs], nonb_rec[idxs], 'o', ms=2, color='#d62728', label='Called Novelties')
    plt.xlabel('Mahalanobis Distance', size=20)
    plt.ylabel('Reconstruction Error', size=20)
    # plt.title(r'Detection with FDR $\alpha$ = {}, Selected Using {} on BDNA Test Set'.format(fdr_level, str(selection))) # no title for ISMB
    # ax.legend(markerscale=6)
    # lgnd = plt.legend(loc="best", scatterpoints=1, fontsize=10)
    # for handle in lgnd.legendHandles:
    #    handle.set_sizes([6.0])
    # Create dummy Line2D objects for legend
    h1 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='#1f77b4', linestyle='None')
    h2 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='#ff7f0e', linestyle='None')
    h3 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='#d62728', linestyle='None')
    # Plot legend.
    plt.legend([h1, h2, h3], ['BDNA', 'NonB', 'Called Novelties'], loc="best", markerscale=2, scatterpoints=1,
               fontsize=15)

    # plt.legend()

    # Create meshgrid of feature 1 and feature 2 values

    xx, yy = np.meshgrid(
        np.linspace(plt.xlim()[0], plt.xlim()[1], 1000),
        np.linspace(plt.ylim()[0], plt.ylim()[1], 1000),
    )
    zz = np.c_[xx.ravel(), yy.ravel()]
    '''
    # Calculate the MCD based Mahalanobis distances
    mahal_robust_cov = MH2.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = plt.contour(
        xx, yy, mahal_robust_cov**(s2_pow), 40, cmap='RdGy', linestyles="dotted"
    )
    plt.ylim(0,500)
    #plt.xlim(0,50)
    plt.colorbar()
    '''

    # plt.ylim(0,500)
    # plt.xlim(0,50)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(img_path, 'test_identified_by_' + str(selection) + '_.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(img_path, 'test_identified_by_' + str(selection) + '_.tif'), dpi=300, bbox_inches='tight')  # for ISMB
    plt.close(fig)
    # plt.show()


def plot_test_sim(bdna_mh, bdna_rec, nonb_mh, nonb_rec, idxs_true_nonb, img_path):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(bdna_mh, bdna_rec, s=2, alpha=.5, label='BDNA')
    plt.scatter(nonb_mh, nonb_rec, s=2, alpha=.5, label='NonB')
    plt.plot(nonb_mh[idxs_true_nonb], nonb_rec[idxs_true_nonb], 'o', ms=2, color='r', label='True NonB')
    plt.xlabel('Mahalanobis Distance', size=20)
    plt.ylabel('Reconstruction Error', size=20)
    plt.title(r'True NonB Locations after using BDNA Test as Null')
    plt.legend()

    # Create meshgrid of feature 1 and feature 2 values

    xx, yy = np.meshgrid(
        np.linspace(plt.xlim()[0], plt.xlim()[1], 1000),
        np.linspace(plt.ylim()[0], plt.ylim()[1], 1000),
    )
    zz = np.c_[xx.ravel(), yy.ravel()]
    '''
    # Calculate the MCD based Mahalanobis distances
    mahal_robust_cov = MH2.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = plt.contour(
        xx, yy, mahal_robust_cov**(s2_pow), 40, cmap='RdGy', linestyles="dotted"
    )
    plt.ylim(0,500)
    #plt.xlim(0,50)
    plt.colorbar()
    '''

    # plt.ylim(0,500)
    # plt.xlim(0,50)
    plt.savefig(os.path.join(img_path, 'test_true_novelties.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(img_path, 'test_true_novelties.tif'), dpi=300, bbox_inches='tight')  # for ISMB
    plt.close(fig)
    # plt.show()


def plot_test_sim_split(bdna_mh, bdna_rec, nonb_mh, nonb_rec, idxs_tp, idxs_fp, selection, img_path):
    # cmap = ['#e41a1c', '#984ea3', '#a65628', '#999999']
    # cmap0 = cm.get_cmap('Set1', 8)    # PiYG
    # cmap = [str(matplotlib.colors.rgb2hex(cmap0(i))) for i in range(cmap0.N)]
    cmap = ['#CAE1FF', '#79CDCD', 'b', '#800000']
    # cmap = ['#C0FF3E','#9ACD32','b','#FF4500']

    # def plot_test(bdna_mh, bdna_rec, nonb_mh, nonb_rec, idxs, img_path, selection, fdr_level):
    fig = plt.figure(figsize=(10, 10))
    # plt.scatter(bdna_mh, bdna_rec, s=2, alpha=.5, color='#1f77b4', label='BDNA')
    plt.scatter(bdna_mh, bdna_rec, s=2, alpha=.5, color=cmap[0], label='B-DNA')
    # plt.scatter(nonb_mh, nonb_rec, s=2, alpha=.5, color='#ff7f0e', label='NonB')
    plt.scatter(nonb_mh, nonb_rec, s=2, alpha=.5, color=cmap[1], label='Non-B DNA')
    # plt.plot(nonb_mh[idxs_tp], nonb_rec[idxs_tp], 'o', ms=2, color='#2ca02c', label='NonB Novelties (TP)')
    plt.plot(nonb_mh[idxs_tp], nonb_rec[idxs_tp], 's', ms=2, color=cmap[2], label='Non-B DNA Novelties (TP)')
    plt.plot(nonb_mh[idxs_fp], nonb_rec[idxs_fp], '^', ms=2, color=cmap[3], label='B-DNA Novelties (FP)')
    plt.xlabel('Mahalanobis Distance', size=20)
    plt.ylabel('Reconstruction Error', size=20)
    # h1 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='#1f77b4', linestyle='None')
    h1 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color=cmap[0], linestyle='None')
    # h2 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='#ff7f0e', linestyle='None')
    h2 = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color=cmap[1], linestyle='None')
    h3 = Line2D([0], [0], marker='s', markersize=np.sqrt(20), color=cmap[2], linestyle='None')
    h4 = Line2D([0], [0], marker='^', markersize=np.sqrt(20), color=cmap[3], linestyle='None')

    # Plot legend.
    # plt.legend([h1, h2, h3, h4], ['B-DNA w/o Motif', 'Non-B DNA', 'Non-B DNA Novelties (TP)', 'B-DNA Novelties (FP)' ], loc="best", markerscale=2, scatterpoints=1, fontsize=15)
    # plt.legend([h1, h2, h3, h4], ['DNA w/o Motif', 'DNA with Motif (N)', 'Non-B DNA with Motif (TP)', 'B-DNA with Motif (FP)' ], loc="best", markerscale=2, scatterpoints=1, fontsize=15)
    plt.legend([h1, h2, h3, h4],
               ['DNA w/o Motif (Null)', 'Non-rejected DNA with Motif (FN+TN)', 'Rejected Non-B DNA with Motif (TP)',
                'Rejected B-DNA with Motif (FP)'], loc="best", markerscale=2, scatterpoints=1, fontsize=15)
    # plt.title(r'True NonB Locations after using BDNA Test as Null')
    # plt.legend()

    # Create meshgrid of feature 1 and feature 2 values

    xx, yy = np.meshgrid(
        np.linspace(plt.xlim()[0], plt.xlim()[1], 1000),
        np.linspace(plt.ylim()[0], plt.ylim()[1], 1000),
    )
    zz = np.c_[xx.ravel(), yy.ravel()]
    '''
    # Calculate the MCD based Mahalanobis distances
    mahal_robust_cov = MH2.mahalanobis(zz)
    mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
    robust_contour = plt.contour(
        xx, yy, mahal_robust_cov**(s2_pow), 40, cmap='RdGy', linestyles="dotted"
    )
    plt.ylim(0,500)
    #plt.xlim(0,50)
    plt.colorbar()
    '''

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(img_path, 'test_identified_by_' + str(selection) + '_GT.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(img_path, 'test_identified_by_' + str(selection) + '_GT.tif'), dpi=300, bbox_inches='tight')  # for ISMB
    plt.close(fig)
    # plt.show()