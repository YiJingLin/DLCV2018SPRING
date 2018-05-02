

def plot_Model(model, outpath=None):
	from keras.utils import plot_model

	if outpath==None:
		outpath = 'model_structure.png'
	elif '.png' not in outpath:
		outpath +='.png'

	plot_model( model , show_shapes=True , to_file=outpath)


def train(model, X, Y, epochs=30, batch_size=8,
			loss_func='categorical_crossentropy',
			optimizer='adam',
			save_model = True,
			model_out_path=None):
	
	if model_out_path==None:
		model_out_path = './model.h5'

	model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
	history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=True)

	print('finish training')

	if save_model:
		print('save the model, directory path : ', model_out_path)
		model.save(model_out_path)

	return history