from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import numpy as np



def VGG16_FCN32s(img_size=(512, 512),
				 data_format = 'channels_first',
				 channel_size=3,
				 load_VGG_weights=True,
				 VGG_weights_path='./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
				):

	img_flatten_size = img_size[0]*img_size[1]


	# Input Layer
	img_input = Input(shape=(channel_size,)+img_size)

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', 
	           data_format=data_format, trainable=False)(img_input) 
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', 
	           data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=data_format)(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
	           data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',data_format=data_format)(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
	           data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',data_format=data_format)(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
	           data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',data_format=data_format)(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
	           data_format=data_format, trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
	           data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',data_format=data_format)(x)

	# FCN32 block
	out = x
	out = BatchNormalization(epsilon=1e-03, mode=0, momentum=0.9)(out)
	out = Conv2D(4096, (7, 7) , activation='selu' , padding='same', data_format=data_format)(out)
	out = Dropout(0.5)(out)
	out = Conv2D(4096, (1, 1) , activation='relu' , padding='same', data_format=data_format)(out)
	out = Dropout(0.5)(out)
	out = Conv2D(7,(1, 1) ,kernel_initializer='he_normal' , padding='same', data_format=data_format)(out)
	# Conv. Transpose
	out = Conv2DTranspose(7,kernel_size=(62,62),strides=(30,30) , use_bias=False , data_format=data_format)(out)

	out = Reshape((-1, img_flatten_size))(out)
	out = Permute((2, 1))(out)
	out = Activation('softmax')(out)

	model = Model(img_input, out)

	if load_VGG_weights:
		try:
			model.load_weights(VGG_weights_path, by_name=True)
			print('./successfully load pretrained weights for model : \n weights path | ', VGG_weights_path)
		except Exception as e:
			print(e)

	return model



def VGG16_FCN8s(img_size=(512, 512),
				 data_format = 'channels_first',
				 channel_size=3,
				 load_VGG_weights=True,
				 VGG_weights_path='./data/vgg16_weights_tf_dim_ordering_tf_kernels.h5',
				):
	'''
	reference :
	https://github.com/divamgupta/image-segmentation-keras/blob/master/Models/FCN8.py
	'''
	
	def crop(out1, out2, i):
		out_shape2 = Model(i, out2).output_shape
		outputHeight2 = out_shape2[2]
		outputWidth2 = out_shape2[3]

		out_shape1 = Model(i, out1).output_shape
		outputHeight1 = out_shape1[2]
		outputWidth1 = out_shape1[3]

		cx = abs( outputWidth1 - outputWidth2 )
		cy = abs( outputHeight2 - outputHeight1 )

		if outputWidth1 > outputWidth2:
		    out1 = Cropping2D(cropping=((0,0), (0,cx)),
		                      data_format=data_format)(out1)
		else:
		    out2 = Cropping2D( cropping=((0,0), (0,cx)),
		                      data_format=data_format)(out2)

		if outputHeight1 > outputHeight2 :
		    out1 = Cropping2D( cropping=((0,cy), (0,0)),
		                      data_format=data_format)(out1)
		else:
		    out2 = Cropping2D( cropping=((0, cy), (0,0)),
		                      data_format=data_format)(out2)

		return out1, out2 


	data_format = 'channels_first'

	img_input = Input(shape=(3,512,512))
	  
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=data_format, trainable=False)(img_input)   
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=data_format, trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=data_format , trainable=False)(x)
	f1 = x

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
	           data_format=data_format , trainable=False)(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
	           data_format=data_format , trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
	                 data_format=data_format , trainable=False )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
	           data_format=data_format , trainable=False )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
	           data_format=data_format , trainable=False)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
	           data_format=data_format , trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
	                 data_format=data_format , trainable=False)(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
	           data_format=data_format , trainable=False )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
	           data_format=data_format , trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
	           data_format=data_format , trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
	                 data_format=data_format , trainable=False )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=data_format , trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=data_format , trainable=False)(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=data_format , trainable=False)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=data_format , trainable=False)(x)
	f5 = x

	# FCN 32s

	out = f5
	out = normalization.BatchNormalization(epsilon=1e-03, mode=0, momentum=0.9)(out) 
	out = Conv2D(4096, (7,7), activation='selu' , padding='same', data_format=data_format)(out)
	out = Dropout(0.5)(out)
	out = normalization.BatchNormalization(epsilon=1e-03, mode=0, momentum=0.9)(out) 
	out = Conv2D(4096, (1,1) , activation='selu', padding='same', data_format=data_format)(out)
	out = Dropout(0.5)(out)

	out = Conv2D(7, (1, 1) ,kernel_initializer='he_normal' , data_format=data_format)(out)
	out = Conv2DTranspose(7, kernel_size=(4,4), strides=(2,2), use_bias=False, data_format=data_format)(out)

	# FCN 16s

	out2 = f4
	out2 = Conv2D(7, (1,1), kernel_initializer='he_normal', data_format=data_format)(out2)

	out, out2 = crop(out, out2, img_input)

	out = Add()([out, out2])

	out = Conv2DTranspose(7, kernel_size=(4,4),  strides=(2,2), use_bias=False, data_format=data_format)(out)

	# FCN 8s

	out2 = f3 
	out2 = (Conv2D(7, (1, 1) ,kernel_initializer='he_normal', data_format=data_format))(out2)
	out2, out = crop( out2 , out , img_input )
	out = Add()([out2, out])
	out = Conv2DTranspose(7, kernel_size=(8,8), strides=(8,8), use_bias=False, data_format=data_format)(out)


	out_shape = Model(img_input, out).output_shape

	outputHeight = out_shape[2]
	outputWidth = out_shape[3]

	out = Reshape((-1, outputHeight*outputWidth))(out)
	out = Permute((2, 1))(out)
	out = Activation('softmax')(out)
	model = Model(img_input , out)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	if load_VGG_weights:
		try:
			model.load_weights(VGG_weights_path, by_name=True)
			print('./successfully load pretrained weights for model : \n weights path | ', VGG_weights_path)
		except Exception as e:
			print(e)

	return model
