#!/bin/bash 

# $1: testing images directory (images are named 'xxxx_sat.jpg')
# $2: output images directory

wget -O 'VGG16_FCN8s_epoch30_model.h5' 'https://www.dropbox.com/s/71oknrs72zu8mhy/VGG16_FCN8s_epoch30_model.h5?dl=1'

python predict.py  --model VGG16_FCN8s --input $1 --output $2
