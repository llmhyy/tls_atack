import os
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.widgets import Slider, Button, RadioButtons

#defaults to rcParams["figure.figsize"] = [6.4, 4.8]

def plot_prediction_on_pktlen(predict_train, true_train, predict_test, true_test, 
                                save_every_epoch, save_dir, show=False):
    """
    Given an array, visualize the traffic over time in a given dimension. This helps us to understand
    whether the model is learning anything at all. We can callback this method at every epoch to observe 
    the changes in predicted traffic
    """
    count = 5
    def choose_random(min, max, count):
        return random.sample(range(min, max), count)

    fig, ax = plt.subplots(nrows=2, ncols=count)
    plt.subplots_adjust(left=0.15, bottom=0.25, wspace=0.4, hspace=0.4)
    fig.set_size_inches(10, 8)
    train_random = choose_random(0, predict_train[0].shape[0], count)
    test_random = choose_random(0, predict_test[0].shape[0], count)
    for i, index in enumerate(zip(train_random,test_random)):
        ax[0,i].set_ylim([0,1])
        ax[1,i].set_ylim([0,1])
        predict = ax[0,i].plot(predict_train[0,index[0],:,0],)
        true = ax[0,i].plot(true_train[0,index[0],:,0], alpha=0.8)
        ax[0,i].set_title('Train #{}'.format(index[0]))
        ax[1,i].plot(predict_test[0,index[1],:,0])
        ax[1,i].plot(true_train[0,index[1],:,0], alpha=0.8)
        ax[1,i].set_title('Test #{}'.format(index[1]))
    fig.legend((predict[0], true[0]),('Predict','True'),loc='center left')
    
    axcolor = 'lightgoldenrodyellow'
    axslider = plt.axes([0.15, 0.1, 0.75, 0.03], facecolor=axcolor)
    s = Slider(axslider, 'Epoch', valmin=1, valmax=len(predict_train), valinit=1, valstep=1)

    def update(val):
        epoch = int(s.val)
        for i, index in enumerate(zip(train_random,test_random)):
            ax[0,i].clear()
            ax[1,i].clear()
            ax[0,i].set_ylim([0,1])
            ax[1,i].set_ylim([0,1])
            ax[0,i].plot(predict_train[epoch-1,index[0],:,0])
            ax[0,i].plot(true_train[0,index[0],:,0], alpha=0.8)
            ax[0,i].set_title('Train #{}'.format(index[0]))
            ax[1,i].plot(predict_test[epoch-1,index[1],:,0])
            ax[1,i].plot(true_train[0,index[1],:,0], alpha=0.8)
            ax[1,i].set_title('Test #{}'.format(index[1]))
        fig.canvas.draw_idle()

    def manual_update(epoch):
        for i, index in enumerate(zip(train_random,test_random)):
            ax[0,i].clear()
            ax[1,i].clear()
            ax[0,i].set_ylim([0,1])
            ax[1,i].set_ylim([0,1])
            ax[0,i].plot(predict_train[epoch-1,index[0],:,0])
            ax[0,i].plot(true_train[0,index[0],:,0], alpha=0.8)
            ax[0,i].set_title('Train #{}'.format(index[0]))
            ax[1,i].plot(predict_test[epoch-1,index[1],:,0])
            ax[1,i].plot(true_train[0,index[1],:,0], alpha=0.8)
            ax[1,i].set_title('Test #{}'.format(index[1]))
        fig.canvas.draw_idle()
    
    s.on_changed(update)

    traffic_len = os.path.join(save_dir, 'traffic_len')
    if not os.path.exists(traffic_len):
        os.mkdir(traffic_len)
    epochs = len(predict_train)
    for epoch in range(0, epochs):
        manual_update(epoch)
        plt.savefig(os.path.join(traffic_len, 'traffic_len_epoch{}'.format((epoch*save_every_epoch)+save_every_epoch)))
    if show:
        plt.show()
    plt.clf()

def plot_distribution(final_acc, overall_mean_acc, feature_filename, save_dir, show=False):
    ax = plt.gca()
    ax.set_ylim(0.0, 1.0)
    plt.plot(final_acc, '|')
    plt.title('Dist of mean cosine similarity for true packets')
    plt.xlabel('Traffic #')
    plt.ylabel('Mean Cosine Similarity')
    plt.axhline(y=round(overall_mean_acc,5), color='r', linestyle='-')
    plt.text(0.05, round(overall_mean_acc,5)-0.025, '{:.5f}'.format(overall_mean_acc), color='r', fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    plt.savefig(os.path.join(save_dir, 'dist_true_({})'.format(feature_filename)))
    if show:
        plt.show()
    plt.clf()


def plot_accuracy_and_distribution(mean_acc_train, median_acc_train, mean_acc_test, median_acc_test, final_acc_train, final_acc_test, 
                                    first, save_every_epoch, save_dir, show=False):
    """
    Plots train and test cosine similarity (mean/median) over training epochs 
    AND distribution of mean cosine similarity for train and test dataset.

    Parameters
    mean_acc_train:     list of mean cosine similarity on train dataset for each epoch
    median_acc_train:   list of median cosine similarity on train dataset for each epoch
    mean_acc_test:      list of mean cosine similarity on test dataset for each epoch
    median_acc_test:    list of median cosine similarity on test dataset for each epoch
    final_acc_train:    list of mean cosine similarity for each traffic in train dataset after last epoch
    final_acc_test:     list of mean cosine similarity for each traffic in test dataset after last epoch
    
    first:              integer to indicate first ___ packets to apply cosine similarity over 1 traffic
    save_every_epoch:   integer to indicate the number of epochs for each save
    save_dir:           string to indicate the directory to save the plots
    show:               boolean to show the plots or not

    """
    plt.subplots_adjust(hspace=0.7)

    plt.subplot(311)
    epochs = len(mean_acc_train)
    # epochs = len(predict_train)
    # for epoch in range(0, epochs):
    #     manual_update(epoch)
    #     plt.savefig(os.path.join(traffic_len, 'traffic_len_epoch{}'.format((epoch*save_every_epoch)+5)))
    x_values = [(epoch*save_every_epoch)+save_every_epoch for epoch in range(0, epochs)]
    plt.plot(x_values, mean_acc_train, alpha=0.7)
    plt.plot(x_values, median_acc_train, alpha=0.7)
    plt.plot(x_values, mean_acc_test, alpha=0.7)
    plt.plot(x_values, median_acc_test, alpha=0.7)
    plt.title('Model cosine similarity for first {} pkts'.format(first))
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Epoch')
    plt.legend(['Train(mean)', 'Train(median)' , 'Val(mean)', 'Val(median)'], loc='upper left')

    plt.subplot(312)
    plt.plot(final_acc_train,'|')
    plt.title('Dist of mean cosine similarity for first {} pkts (train)'.format(first))
    plt.ylabel('Mean Cosine Similarity')
    plt.xlabel('Traffic #')

    plt.subplot(313)
    plt.plot(final_acc_test,'|')
    plt.title('Dist of mean cosine similarity for first {} pkts (validation)'.format(first))
    plt.ylabel('Mean Cosine Similarity')
    plt.xlabel('Traffic #')

    acc_dist = os.path.join(save_dir, 'acc_dist')
    if not os.path.exists(acc_dist):
        os.mkdir(acc_dist)
    plt.savefig(os.path.join(acc_dist,'acc_dist_{}pkts').format(first))
    if show:
        plt.show()
    plt.clf()
