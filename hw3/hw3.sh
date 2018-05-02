#!/bin/bash 

# $1: testing images directory (images are named 'xxxx_sat.jpg')
# $2: output images directory

wget -O 'VGG16_FCN32s_epoch30_model.h5' 'https://www.dropbox.com/s/x5k1hx1y3ft6g0l/VGG16_FCN32s_epoch30_model.h5?dl=1'

python predict.py  --input $1 --output $2
