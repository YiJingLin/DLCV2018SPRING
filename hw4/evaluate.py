import torch
import torch.utils.data as Data
from torch.autograd import Variable

from model.CVAE import CVAE
from model.GAN import GAN
from model.ACGAN import ACGAN


import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess

import cv2, pickle, argparse
from tqdm import tqdm


output_path = None

def Problem1(X):
	cvae = CVAE()
	if torch.cuda.is_available():
	    cvae = cvae.cuda()
	else:
	    cvae = cvae.cpu()
	cvae.load_state_dict(torch.load('./output/cvae.state_dict', map_location=lambda storage, loc: storage))

	Problem1_2()
	Problem1_3(cvae, X)
	Problem1_4(cvae)

def Problem1_2():
	loss_history = None
	with open('./output/cvae_1e-3_loss_history.pkl', 'rb') as file:
	    loss_history = pickle.load(file)

	itr_list = []
	loss_list = []
	KLD_list = []
	MSE_list = []

	num = 4096*3

	for itr, data in enumerate(tqdm(loss_history)):
	    loss_list.append(data[0]/num)
	    MSE_list.append(data[1]/num)
	    KLD_list.append(data[2])
	    itr_list.append(itr)


	fig = plt.figure()

	### KLD
	plt.subplot(221)
	plt.plot(itr_list, KLD_list, color='orange', label='KLD')
	plt.xlabel('iteration')
	plt.ylabel('KLD')

	### MSE
	plt.subplot(222)
	plt.plot(itr_list, MSE_list, color='orange', label='MSE')
	plt.xlabel('iteration')
	plt.ylabel('MSE')
	plt.ylim(5e-3, 5e-2)

	fig.savefig(output_path+'fig1_2.jpg', bbox_inches='tight')
	print('Problem1-2 is finish.')

def Problem1_3(cvae, X):
	cvae.eval()
	x = X[:10]

	z, mu, logvar = cvae(Variable(torch.FloatTensor(x)))

	z = z.detach().numpy()
	z = z.transpose(2,0,3,1)
	x = x.transpose(2,0,3,1)

	z_ = z.reshape(64, -1, 3)
	x_ = x.reshape(64, -1, 3)

	fig = plt.figure()
	result = np.concatenate((x_, z_), axis=0)

	plt.imshow(result)
	plt.xticks([]), plt.yticks([])
	plt.axis('off')

	fig.set_figheight(fig.get_figheight()*2)
	fig.set_figwidth(fig.get_figwidth()*2)
	fig.savefig(output_path+'fig1_3.jpg', bbox_inches='tight', pad_inches = 0)
	print('Problem1-3 is finish.')

def Problem1_4(cvae):
	latent_size = 1024
	np.random.seed(80)

	# noises = np.random.rand(32,1024)
	noises = np.random.normal(0,1,(32, 1024))
	result = cvae.decode(Variable(torch.FloatTensor(noises)))
	result = result.detach().numpy().transpose(2,0,3,1).reshape(64, -1, 3)
	result = np.concatenate((result[:,:64*8,:], result[:,64*8:64*16,:], result[:,64*16:64*24], result[:,64*24:,:]), axis=0)

	# render
	fig = plt.figure()

	plt.imshow(result)
	plt.xticks([]), plt.yticks([])
	plt.axis('off')

	fig.set_figheight(fig.get_figheight()*2)
	fig.set_figwidth(fig.get_figwidth()*2)
	fig.savefig(output_path+'fig1_4.jpg', bbox_inches='tight', pad_inches = 0)
	print('Problem1-4 is finish.')

def Problem1_5():
	pass

#####################################################################################

def Problem2():
	gan = GAN()
	if torch.cuda.is_available():
	    gan.G = gan.G.cuda()
	    gan.D = gan.D.cuda()
	else:
	    gan.G = gan.G.cpu()
	    gan.D = gan.D.cpu()

	gan.load()

	Problem2_2()
	Problem2_3(gan)


def Problem2_2():
	loss_history = None
	with open('./output/GAN_history.pkl', 'rb') as file:
	    loss_history = pickle.load(file)

	D_loss = loss_history['D_loss']
	G_loss = loss_history['G_loss']

	### mean in 9 range
	range_ = 101
	append_ = range_//2

	G_loss_ = G_loss + ([G_loss[-1]]*append_)
	G_loss_ = ([G_loss[0]]*append_) + G_loss_
	G_loss_.__len__()
	    
	G_loss_m = [sum(G_loss_[itr:itr+range_])/range_ for itr in range(40000)]

	D_loss_ = D_loss + ([D_loss[-1]]*append_)
	D_loss_ = ([D_loss[0]]*append_) + D_loss_
	D_loss_.__len__()
	    
	D_loss_m = [sum(D_loss_[itr:itr+range_])/range_ for itr in range(40000)]

	# plot 

	itr_list = [itr for itr in range(G_loss.__len__())]

	fig = plt.figure()
	plt.plot(itr_list, D_loss, alpha=.5, label='D_loss', color='orange') # alpha=Transparency
	plt.plot(itr_list, G_loss, alpha=.5, label='G_loss', color='b') # alpha=Transparency
	plt.plot(itr_list, D_loss_m, label='m101 D_loss', color='orange')
	plt.plot(itr_list, G_loss_m, label='m101 G_loss', color='b')

	plt.ylabel('BCE loss')
	plt.xlabel('iteration')
	plt.legend()
	fig.savefig(output_path+'fig2_2.jpg', bbox_inches='tight')
	print('Problem2-2 is finish.')


def Problem2_3(gan):
	noise_dim = 62
	np.random.seed(90)

	# noises = np.random.rand(32,1024)
	noises = np.random.normal(0,1,(32, noise_dim))
	result = gan.G(Variable(torch.FloatTensor(noises)))
	result = result.detach().numpy().transpose(2,0,3,1).reshape(64, -1, 3)
	result = np.concatenate((result[:,:64*8,:], result[:,64*8:64*16,:], result[:,64*16:64*24], result[:,64*24:,:]), axis=0)

	# render
	fig = plt.figure()

	plt.imshow(result)
	plt.xticks([]), plt.yticks([])
	plt.axis('off')

	fig.set_figheight(fig.get_figheight()*2)
	fig.set_figwidth(fig.get_figwidth()*2)
	fig.savefig(output_path+'fig2_4.jpg', bbox_inches='tight', pad_inches = 0)
	print('Problem2-3 is finish.')

###############################################################

def Problem3():
	acgan = ACGAN()
	if torch.cuda.is_available():
	    acgan.G = acgan.G.cuda()
	    acgan.D = acgan.D.cuda()
	else:
	    acgan.G = acgan.G.cpu()
	    acgan.D = acgan.D.cpu()

	acgan.load()

	Problem3_2()
	Problem3_3(acgan)

def Problem3_2():
	loss_history = None
	with open('./output/ACGAN_history.pkl', 'rb') as file:
	    loss_history = pickle.load(file)

	D_loss = loss_history['D_loss']
	G_loss = loss_history['G_loss']

	### mean in 9 range
	range_ = 101
	append_ = range_//2

	G_loss_ = G_loss + ([G_loss[-1]]*append_)
	G_loss_ = ([G_loss[0]]*append_) + G_loss_
	G_loss_.__len__()
	    
	G_loss_m = [sum(G_loss_[itr:itr+range_])/range_ for itr in range(G_loss.__len__())]

	D_loss_ = D_loss + ([D_loss[-1]]*append_)
	D_loss_ = ([D_loss[0]]*append_) + D_loss_
	D_loss_.__len__()
	    
	D_loss_m = [sum(D_loss_[itr:itr+range_])/range_ for itr in range(D_loss.__len__())]


	# plot 

	itr_list = [itr for itr in range(G_loss.__len__())]

	fig = plt.figure()
	plt.plot(itr_list, D_loss, alpha=.5, label='D_loss', color='orange') # alpha=Transparency
	plt.plot(itr_list, G_loss, alpha=.5, label='G_loss', color='b') # alpha=Transparency
	plt.plot(itr_list, D_loss_m, label='m101 D_loss', color='orange')
	plt.plot(itr_list, G_loss_m, label='m101 G_loss', color='b')

	plt.ylabel('BCE loss')
	plt.xlabel('iteration')
	plt.legend()
	fig.savefig(output_path+'fig3_2.jpg', bbox_inches='tight')


def Problem3_3(acgan):
	noise_dim = 62
	label_dim = 13
	np.random.seed(80)

	# noises = np.random.rand(32,1024)
	noises = np.random.normal(0,1,(10, noise_dim))
	labels = np.array(np.random.randint(0,2,(10, label_dim)).tolist(),dtype='float')
	labels_before = labels.copy()
	labels_after = labels.copy()

	idx = 9
	for itr in range(10):
	    labels_before[itr,idx] = 0. # smile
	    labels_after[itr,idx] = 1. # not smile

	# print(labels_before[:,idx], labels_after[:,idx])

	result_before = acgan.G(Variable(torch.FloatTensor(noises)), Variable(torch.FloatTensor(labels_before)))
	result_before = result_before.detach().numpy().transpose(2,0,3,1).reshape(64, -1, 3)
	result_after = acgan.G(Variable(torch.FloatTensor(noises)), Variable(torch.FloatTensor(labels_after)))
	result_after = result_after.detach().numpy().transpose(2,0,3,1).reshape(64, -1, 3)
	result = np.concatenate((result_before, result_after), axis=0)

	# render
	fig = plt.figure()

	plt.imshow(result)
	plt.xticks([]), plt.yticks([])
	plt.axis('off')

	fig.set_figheight(fig.get_figheight()*2)
	fig.set_figwidth(fig.get_figwidth()*2)
	fig.savefig(output_path+'fig3_3.jpg', bbox_inches='tight', pad_inches = 0)

##########################################################################################

def main(direc_path='./data/hw4_data/'): 
	if direc_path[-1] is not '/':
		direc_path +='/'

	preprocess_ = preprocess()	
	X, Xc = preprocess_.main(direc_path=direc_path, wrap=False, mode='test', model='ACGAN')

	Problem1(X)
	Problem2()
	Problem3()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-i', '--input', type=str, help='input file path')
	parser.add_argument('-o','--output', type=str, help='output files dirctory')

	args = parser.parse_args()

	output_path = args.output
	if output_path[-1] is not '/':
		output_path+='/'

	main(direc_path=args.input)