wget -O './output/GAN_G.pkl' 'https://www.dropbox.com/s/0akk6zxrdapysoq/GAN_G.pkl?dl=1'
wget -O './output/GAN_D.pkl' 'https://www.dropbox.com/s/821v6mmwi3lh41r/GAN_D.pkl?dl=1'
wget -O './output/cvae.state_dict' 'https://www.dropbox.com/s/5dvbkr13hfiebcq/cvae.state_dict?dl=1'
wget -O './output/ACGAN_G.pkl' 'https://www.dropbox.com/s/egqfqcwiuatrzr7/ACGAN_G.pkl?dl=1'
wget -O './output/ACGAN_D.pkl' 'https://www.dropbox.com/s/fs9sqore5d2gd78/ACGAN_D.pkl?dl=1'

python evaluate.py --input $1 --output $2