import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import pickle, os, time
import numpy as np

'''
reference:
https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/GAN.py
'''

'''
reference:
https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/GAN.py
'''


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=62):
        super(generator, self).__init__()
        
        self.input_height = 64 # image height
        self.input_width = 64 # image width
        self.input_dim = input_dim # noise vector
        self.output_dim = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
    
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.input_height = 64
        self.input_width = 64
        self.input_dim = 3 # in_channels
        self.output_dim = 1 # 

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
    
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)

        return x

class GAN(object):
    def __init__(self, model_name='GAN', path='./output/',
                 sample_num=3, mu=0, sigma=1,
                 z_dim=62):
        # parameters
        self.gpu_mode = torch.cuda.is_available()
        self.model_name = model_name
        self.path = path
        
        self.sample_num = sample_num
        self.mu = mu # mean
        self.sigma = sigma # Variance
        self.z_dim = z_dim # noise dimension
        
        # networks init
        self.G = generator(input_dim=z_dim)
        self.D = discriminator()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-3)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=1e-3)
    
        # submodel, loss 
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        # print('---------- Networks architecture -------------')
        # self.print_network(self.G)
        # self.print_network(self.D)
        # print('-----------------------------------------------')
        
        self.sample_z = self.sample_noise(num=sample_num)
    
    def train(self, dataloader, n_epoch=1000, g_early_stop=None, n_loop_G=3, n_loop_D=1):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        batch_size = dataloader.batch_size
        
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(batch_size, 1)), Variable(torch.zeros(batch_size, 1))

        
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(n_epoch):
            self.G.train()
            epoch_start_time = time.time()
            for itr, x_ in enumerate(dataloader):
                x_ = x_[0]
                if itr == dataloader.dataset.__len__() // batch_size:
                    break

                z_ = self.sample_noise(num=batch_size)

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)
                
                # update D network
                mean_D_loss = 0
                for _ in range(n_loop_D):
                    self.D_optimizer.zero_grad()
                    # real
                    D_real = self.D(x_)
                    D_real_loss = self.BCE_loss(D_real, self.y_real_)
                    # fake
                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                    D_loss = D_real_loss + D_fake_loss
                    mean_D_loss += D_loss.data[0]

                    D_loss.backward()
                    self.D_optimizer.step()
                mean_D_loss /= n_loop_D
                self.train_hist['D_loss'].append(float(mean_D_loss))
                
                # update G network
                mean_G_loss = 0
                for _ in range(n_loop_G):
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.BCE_loss(D_fake, self.y_real_)
                    
                    mean_G_loss += G_loss.data[0]
                    
                    G_loss.backward()
                    self.G_optimizer.step()
                mean_G_loss /= n_loop_G
                self.train_hist['G_loss'].append(float(mean_G_loss))

                
            print("Epoch: [%2d], cost: %d sec | D_loss: %.8f, G_loss: %.8f" %
                  ((epoch + 1),(time.time() - epoch_start_time), D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            self.visualize_results(num=self.sample_num)
            
            if g_early_stop is not None and G_loss.data[0] < g_early_stop:
                self.save(class_='_es'+str(g_early_stop))
                print('Reach early stop loss for Generator : [ %.4f | %.4f ], stop training and save the model.')
                return self.train_hist
        
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              n_epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        
        return self.train_hist
        
    def visualize_results(self, fix=True, num=3):
        self.G.eval()

#         if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
#             os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

#         tot_num_samples = min(self.sample_num, self.batch_size)
#         image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            samples = self.G(self.sample_z)
        else:
            sample_z = self.sample_noise(num=num)
            samples = self.G(sample_z)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        
        fig=plt.figure()
        for idx, sample in enumerate(samples):
            fig.add_subplot(1,num,(idx+1)) # (row, column, idx)
            
#             max_n = max(sample.max(),1)
#             min_n = min(sample.min(),0)
#             plt.imshow((sample-min_n)/(max_n-min_n))
            plt.imshow(sample)
        plt.show()
            
#         utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
#                           self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
    
    def evaluate():
#         if self.gpu_mode:
#             torch.cuda.manual_seed(2)
#         else:
#             torch.manual_seed(2)
        ### report ###
        pass
    def sample_noise(self, num=3):
        sample_z = np.random.normal(self.mu, self.sigma, num*self.z_dim).reshape(num, self.z_dim)
        
        if self.gpu_mode:
            sample_z = Variable(torch.FloatTensor(sample_z).cuda(), volatile=True)
        else:
            sample_z = Variable(torch.FloatTensor(sample_z), volatile=True)
        return sample_z
        
    def print_network(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)
    
    def save(self, path=None, class_=''):
        if path is None:
            path = self.path
        
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.G.state_dict(), os.path.join(path, self.model_name + class_ + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(path, self.model_name + class_ + '_D.pkl'))

        with open(os.path.join(path, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, path=None, class_=''):
        if path is None:
            path = self.path
        if self.gpu_mode:
            self.G.load_state_dict(torch.load(os.path.join(path, self.model_name + class_ + '_G.pkl')))
            self.D.load_state_dict(torch.load(os.path.join(path, self.model_name + class_ + '_D.pkl')))
        else:
            self.G.load_state_dict(torch.load(os.path.join(path, self.model_name + class_ + '_G.pkl'), map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(os.path.join(path, self.model_name + class_ + '_D.pkl'), map_location=lambda storage, loc: storage))