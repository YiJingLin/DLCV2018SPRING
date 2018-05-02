from process import read_sats, read_masks, One_Hot, save_predMasks
from keras.models import load_model
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-in', '--input', help='data input directory path', type=str)
	parser.add_argument('-out', '--output', help='data out directory path', type=str)
	parser.add_argument('-m', '--model', help='model class, default VGG16_FCN32s.\noptions : 1) VGG16_FCN32s,\n 2) VGG16_FCN8s', type=str)

	args = parser.parse_args()


	#### load data ####
	print('starting loading data ...')

	X, file_list = read_sats(args.input)

	print('finish data loading and preprocessing.')


	#### load model ####
	if args.model =='VGG16_FCN8s' :
		model_path = './VGG16_FCN8s_epoch30_model.h5'
	else:
		model_path = './VGG16_FCN32s_epoch30_model.h5'

	print('loading model, path : ', model_path)
	model = load_model(model_path)


	#### prediction ####
	preds = model.predict(X)

	#### save model ####
	directory = args.output
	if directory[-1] != '/':
		directory+='/'

	file_list = [filename.replace('sat.jpg', 'mask.png') for filename in file_list]
	save_predMasks(preds, directory=directory, name_list=file_list)
