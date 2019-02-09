import os
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.widgets import Slider, Button, RadioButtons


#defaults to rcParams["figure.figsize"] = [6.4, 4.8]

def visualize_traffic(predict_train, true_train, predict_test, true_test, save_every_epoch, save, show=False):
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

    traffic_len = os.path.join(save, 'traffic_len')
    if not os.path.exists(traffic_len):
        os.mkdir(traffic_len)
    for epoch in range(0, len(predict_train), max(len(predict_train)//10, 1)):
        manual_update(epoch)
        plt.savefig(os.path.join(traffic_len, 'traffic_len_epoch{}'.format(epoch*save_every_epoch)))
    if show:
        plt.show()
    plt.clf()