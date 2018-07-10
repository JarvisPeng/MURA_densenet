import matplotlib.pylab as plt
import json
import numpy as np


def plot_MURA(save=True):

    with open("./log/experiment_log_MURA.json", "r") as f:
        d = json.load(f)

    train_accuracy = 100 * (np.array(d["train_loss"])[:, 1])
    valid_accuracy = 100 * (np.array(d["valid_loss"])[:, 1])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Accuracy')
    ax1.plot(train_accuracy, color="tomato", linewidth=2, label='train_acc')
    ax1.plot(valid_accuracy, color="steelblue", linewidth=2, label='valid_acc')
    ax1.legend(loc=0)

    train_loss = np.array(d["train_loss"])[:, 0]
    valid_loss = np.array(d["valid_loss"])[:, 0]

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(train_loss, '--', color="tomato", linewidth=2, label='train_loss')
    ax2.plot(valid_loss, '--', color="steelblue", linewidth=2, label='valid_loss')
    ax2.legend(loc=1)

    ax1.grid(True)

    if save:
        fig.savefig('./figures/plot_MURA.jpg')

    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plot_MURA()
