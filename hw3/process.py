import numpy as np
import argparse, scipy.misc, os
from keras.utils.np_utils import to_categorical
import skimage.io as io

############# preprocess ##############

def _flip_channel(img,img_num): # assume img chennel at dim 3
    result = []
    for i in range(img_num):
        result.append(img[:,:,i])
    return np.array(result)

def _GET_MEAN_VALUE(MEAN_VALUE = np.array([103.939, 116.779, 123.68])):  # BGR
    matrx = []
    for i in range(512):
        tmp = []
        for j in range(512):
            tmp.append(MEAN_VALUE)
        matrx.append(tmp)

    matrx = np.array(matrx)
    matrx = _flip_channel(matrx,3)
    return matrx

def _preprocess(img, MEAN_VALUE):
        # img is (channels, height, width), values are 0-255
        img = img[::-1].astype(MEAN_VALUE.dtype)  # switch to BGR
        img -= MEAN_VALUE
        return img

def read_sats(filepath):
    '''
    Read sats from directory and tranform to categorical
    '''
    MEAN_VALUE = _GET_MEAN_VALUE()

    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    file_list = np.array(file_list)
    sats = []

    sats = np.empty((len(file_list),3, 512, 512))
    for i in range(len(file_list)):
        sat = np.array(scipy.misc.imread(os.path.join(filepath, file_list[i])))
        sat = _flip_channel(sat,3)
        sat = _preprocess(sat, MEAN_VALUE)
        sat = np.array(sat)
        sats[i] = sat
        
    return sats, file_list



def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = scipy.misc.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land  
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 4] = 6  # (Red: 100) Unknown 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 
        
    return masks

def One_Hot(Y):
	print('img_processing | starting one-hot-encoding process for Y ...')

	Y = to_categorical(Y.reshape(-1), 7)
	Y = Y.reshape(-1, 512* 512, 7)

	print('finish | Y.shape : ', Y.shape, '\n')
	return Y



################ postprocess ##################

def save_predMasks(preds, directory='', name_list=['val_008.png', 'val_097.png', 'val_107.png']):
    
    if not len(preds) == len(name_list):
        print('incorrect data length between prediction array and name_list.')
        return None
    
    def cmap(idx):
        result = None
        if idx == 0 :
            result = [0,1,1]
        elif idx == 1 :
            result = [1,1,0]
        elif idx == 2 :
            result = [1,0,1]
        elif idx == 3 :
            result = [0,1,0]
        elif idx == 4 :
            result = [0,0,1]
        elif idx == 5 :
            result = [1,1,1]
        elif idx == 6 :
            result = [0,0,0]
        return result
    
    # flatten
    preds = preds.reshape(-1, 7)
    # extract idx which have highest score in last dim
    preds = np.argmax(preds, axis=1)
    # trans idx to 3 channels
    preds = np.array([ cmap(pred) for pred in preds])
    # scaling
    preds = preds * 255
    # reshape back to (data_num, img_width, image_height, channels)
    preds = preds.reshape(-1, 512, 512, 3)

    # finally, save the prediction images
    for idx, pred in enumerate(preds):
        io.imsave(directory+name_list[idx], pred)
        
    print('successfully save the prediction')



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-in', '--input', help='data input directory path', type=str)
	args = parser.parse_args()


	print('starting loading data ...')

	X, file_list = read_sats(args.input)
	Y = read_masks(args.input)

	One_Hot(Y)

	print('finish data loading and preprocessing.')
