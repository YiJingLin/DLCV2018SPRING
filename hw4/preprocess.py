import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.utils.data as Data


class preprocess():
	def __init__(self):
		super(preprocess, self).__init__()


	def loadImg_pathList(self, direc_path ='./data/hw4_data/train/'):

		if direc_path[-1] is not '*':
			direc_path += '*'
		return glob.glob(direc_path)


	def loadImg_label(self, csv_path='./data/hw4_data/train.csv'):
		train_feat = pd.read_csv(csv_path, encoding='utf-8')
		Xc = np.array(train_feat.values[:,1:], dtype='float')
		return Xc


	def loadImg_npAry(self, path_list):
		X = tuple()
		for path in tqdm(path_list):
			X += (plt.imread(path).reshape(1, 64, 64, 3),)
		X = np.concatenate(X, axis=0)
		return X


	def main(self, direc_path='./data/hw4_data/', transpose=(0,3,1,2), model='GAN', mode='train',
		wrap=False, batch_size =128, shuffle=True):
		# check mode
		if not mode in ['train', 'test']:
			raise ValueError('wrong mode : only train and test is required .')
		# check model
		if not model in ['VAE', 'CVAE', 'GAN', 'ACGAN', 'InfoGAN']:
			raise ValueError('wrong model type : only VAE, CVAE, GAN , ACGAN, InfoGAN is required .')


		path_list = self.loadImg_pathList(direc_path+'./'+mode+'/*')
		X = self.loadImg_npAry(path_list)

		Xc = None # each image have labels in train.csv/test.csv
		if model in ['ACGAN', 'InfoGAN']:
			Xc = self.loadImg_label(direc_path+'./'+mode+'.csv')


		if not transpose is None:
			X = X.transpose(transpose)
		print('finish loading images, ', mode, ' data shape : ', X.shape)

		if wrap:
			if model in ['ACGAN', 'InfoGAN']:
				dataset = Data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Xc))
			else:
				dataset = Data.TensorDataset(torch.FloatTensor(X))


			dataloader = Data.DataLoader(
						dataset = dataset,
						batch_size = batch_size,
						shuffle = shuffle,
					)			
			return dataloader

		return X, Xc


	if __name__ == '__main__' :
		main()