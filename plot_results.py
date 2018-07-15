# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import json
import numpy as np


def plot_MURA(save=True):

	model_name = '1test'
	with open("./log/experiment_log_MURA_%s.json"%(model_name), "r") as f:
		d = json.load(f)

	train_accuracy = 100 * (np.array(d["train_loss"])[:, 1])
	valid_accuracy = 100 * (np.array(d["valid_loss"])[:, 1])
	x = list(range(1,len(d["train_loss"])+1))           #从epoch1开始作图

	fig = plt.figure(model_name)
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel('Accuracy')
	ax1.plot(x,train_accuracy, color="tomato", linewidth=2, label='train_acc')
	ax1.plot(x,valid_accuracy, color="steelblue", linewidth=2, label='valid_acc')
	ax1.legend(loc=0)

	train_loss = np.array(d["train_loss"])[:, 0]
	valid_loss = np.array(d["valid_loss"])[:, 0]

	ax2 = ax1.twinx()
	ax2.set_ylabel('Loss')
	ax2.plot(x,train_loss, '--', color="tomato", linewidth=2, label='train_loss')
	ax2.plot(x,valid_loss, '--', color="steelblue", linewidth=2, label='valid_loss')
	ax2.legend(loc=1)

	ax1.grid(True)

	if save:
		# fig.savefig('./figures/plot_MURA.svg')
		fig.savefig('./figures/plot_MURA_%s.jpg'%(model_name))

	plt.show()
	plt.clf()
	plt.close()

def plot_MURA_1(save=True):

	with open("./log/experiment_log_MURA.json", "r") as f:
		d = json.load(f)

	train_accuracy = 100 * (np.array(d["train_loss_all"])[:, 1])
	x = list(range(1,len(d["train_loss_all"])+1))           #从epoch1开始作图

	fig1 = plt.figure('Accuracy of each batch')
	# plt.set_ylabel('Accuracy')
	plt.plot(x,train_accuracy, color="tomato", linewidth=2, label='train_acc')
	plt.legend(loc=0)
	plt.grid(True)

	train_loss = np.array(d["train_loss_all"])[:, 0]

	fig2 = plt.figure('Loss of each batch')
	# plt.set_ylabel('Loss')
	plt.plot(x,train_loss, '--', color="tomato", linewidth=2, label='train_loss')
	plt.legend(loc=1)

	plt.grid(True)

	if save:
		fig1.savefig('./figures/plot_MURA_1.jpg')
		fig2.savefig('./figures/plot_MURA_2.jpg')

	plt.show()
	plt.clf()
	plt.close()

if __name__ == '__main__':
	plot_MURA()
	plot_MURA_1()
