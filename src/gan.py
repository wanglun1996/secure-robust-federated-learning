# BSD 2-Clause License

# Copyright (c) 2022, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import utils
import networks
import argparse
import numpy as np
import os
from networks import ConvNet


class DCGAN(object):
    def __init__(self, data_dim, args):
        self.args = args
        self.epoch = 0
        self.criterion = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0
        self.fixed_noise = torch.randn(1, args.nz)
        self.init_model_optimizer(data_dim)

    def init_model_optimizer(self, data_dim):
        print('Initializing Model and Optimizer...')
        self.G = networks.__dict__['decoder_fl'](dim=data_dim, nc=self.args.nz)
        self.G = torch.nn.DataParallel(self.G).cuda()
        self.optim_G = torch.optim.Adam(
                        self.G.module.parameters(),
                        lr=self.args.gan_lr*2,
                        betas=(self.args.beta1, 0.999)
                        )

        self.D = networks.__dict__['discriminator_fl'](dim=1, nc=data_dim)
        self.D = torch.nn.DataParallel(self.D).cuda()
        self.optim_D = torch.optim.Adam(
                        self.D.module.parameters(),
                        lr=self.args.gan_lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'G': self.G.module.state_dict(),
            'D': self.D.module.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.G.module.load_state_dict(ckpt['G'])
        self.D.module.load_state_dict(ckpt['D'])

    def save_image(self, path=None):
        sample_num = 10
        for i in range(sample_num):
            torch.randn(1, self.args.nz)
            if i == 0:
                fake = self.G(self.fixed_noise)
            else:
                fake += self.G(self.fixed_noise)
        return fake / sample_num

    def train(self, data_loader):
        self.epoch += 1
        self.G.train()
        self.D.train()
        record_G = utils.Record()
        record_D = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, data in enumerate(data_loader, 0):
            progress.update(i + 1)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            self.D.zero_grad()
            real_data = data[0].cuda()
            batch_size = real_data.size(0)
            label = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).cuda()

            output = self.D(real_data)
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.args.nz).cuda()
            fake = self.G(noise)
            label.fill_(self.fake_label)
            output = self.D(fake.detach())
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optim_D.step()
            record_D.add(errD)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.G.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            output = self.D(fake)
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optim_G.step()
            record_G.add(errG)

        progress.finish()
        utils.clear_progressbar()
        print('----------------------------------------')
        print('Epoch: %d' % self.epoch)
        print('Costs time: %.2f s' % (time.time() - start_time))
        print('Loss of G is: %f' % (record_G.mean()))
        print('Loss of D is: %f' % (record_D.mean()))
        print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))
        
class WGAN(object):
    def __init__(self, data_dim, args):
        self.args = args
        self.epoch = 0
        self.fixed_noise = torch.randn(1, args.nz)
        self.init_model_optimizer(data_dim)

    def init_model_optimizer(self, data_dim):
        print('Initializing Model and Optimizer...')
        self.G = networks.__dict__['decoder_fl'](dim=data_dim, nc=self.args.nz)
        self.G = torch.nn.DataParallel(self.G).cuda()
        self.optim_G = torch.optim.Adam(
                        self.G.module.parameters(),
                        lr=self.args.gan_lr,
                        betas=(self.args.beta1, 0.9)
                        )

        self.D = networks.__dict__['discriminator_wgan'](dim=1, nc=data_dim)
        self.D = torch.nn.DataParallel(self.D).cuda()
        self.optim_D = torch.optim.Adam(
                        self.D.module.parameters(),
                        lr=self.args.gan_lr,
                        betas=(self.args.beta1, 0.9)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'G': self.G.module.state_dict(),
            'D': self.D.module.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.G.module.load_state_dict(ckpt['G'])
        self.D.module.load_state_dict(ckpt['D'])

    def save_image(self, path=None):
        sample_num = 10
        for i in range(sample_num):
            torch.randn(1, self.args.nz)
            if i == 0:
                fake = self.G(self.fixed_noise)
            else:
                fake += self.G(self.fixed_noise)
        return fake / sample_num

    def calc_gradient_penalty(self, real_data, fake_data):
        LAMBDA = 10
        BATCH_SIZE = real_data.shape[0]
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() * LAMBDA
        return gradient_penalty

    def train(self, data_loader):
        #with torch.autograd.set_detect_anomaly(True):
        self.epoch += 1
        self.G.train()
        self.D.train()
        record_G = utils.Record()
        record_D = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        for i, data in enumerate(data_loader, 0):
            progress.update(i + 1)
            ############################
            # (1) Update D network: [average critic score on real images] – [average critic score on fake images]
            ###########################
            # train with real
            self.D.zero_grad()
            real_data = data[0].cuda()
            batch_size = real_data.size(0)

            output = self.D(real_data)
            errD_real = -output.mean()
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, self.args.nz).cuda()
            fake = self.G(noise)
            output = self.D(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            gradient_penalty = self.calc_gradient_penalty(real_data, fake.detach())
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            self.optim_D.step()
            record_D.add(errD)

            ############################
            # (2) Update G network: -[average critic score on fake images]
            ###########################
            self.G.zero_grad()
            output = self.D(fake)
            errG = -output.mean()
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optim_G.step()
            record_G.add(errG)

        progress.finish()
        utils.clear_progressbar()
        print('----------------------------------------')
        print('Epoch: %d' % self.epoch)
        print('Costs time: %.2f s' % (time.time() - start_time))
        print('Loss of G is: %f' % (record_G.mean()))
        print('Loss of D is: %f' % (record_D.mean()))
        print('D(x) is: %f, D(G(z1)) is: %f, D(G(z2)) is: %f' % (D_x, D_G_z1, D_G_z2))


def generate_random_data(client_num=5000, input_dim=50):
    random_data = np.random.normal(3, 1, size=(client_num, input_dim))
    print(np.average(random_data, axis=0))
    for i in range(500):
        random_data[i*10] = random_data[i*10] * 100
    return torch.from_numpy(random_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--nc', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--gan_lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--agg', default='dcgan')
    parser.add_argument('--next_round', type=int, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    orig_shape = []
    average_grad = []
    if args.next_round >= 8:
        args.gan_lr /= 10

    for layer_id in range(8):

        temp_file_name = './results/gan_' + str(args.next_round-1) + '_' + str(layer_id) + '.npy'
        with open(temp_file_name, 'rb') as f:
            client_data = np.load(f)
        orig_shape.append(client_data[0].shape)
        training_loads = []
        for i in range(len(client_data)):
            training_loads.append(client_data[i].flatten())
        training_loads = np.array(training_loads)
        np.random.shuffle(training_loads)
        training_samples = []
        if training_loads.shape[1] > 1.5e5:
            split_num = training_loads.shape[1] // 2
            training_samples.append(training_loads[:,:split_num])
            training_samples.append(training_loads[:,split_num:])
        else:
            training_samples.append(training_loads)
        
        fakes = []

        for training_data in training_samples:
            print(training_data.shape[1])
            avg = torch.from_numpy(np.average(training_data, axis=0)).cuda()
            if args.agg == 'dcgan':
                gan_fl = DCGAN(training_data.shape[-1], args)
                training_data = torch.from_numpy(training_data.reshape(100, 1, 10, -1)).float()
                for i in range(args.epoch):
                    gan_fl.train(training_data)
                    fake = gan_fl.save_image()
                    print('err:', (torch.norm(fake - avg) / avg.shape[-1]).item())
                fakes.append(fake)

            elif args.agg == 'wgan':
                gan_fl = WGAN(training_data.shape[-1], args)
                training_data = torch.from_numpy(training_data.reshape(100, 1, 10, -1)).float()
                for i in range(args.epoch):
                    gan_fl.train(training_data)
                    fake = gan_fl.save_image()
                    print('err:', (torch.norm(fake - avg) / avg.shape[-1]).item())
                fakes.append(fake)

            elif args.agg == 'avg':
                fakes.append(torch.from_numpy(np.average(training_data, axis=0)).cuda())
        fake = torch.cat(fakes, dim=1)
        average_grad.append(fake)

    network = ConvNet(input_size=28, input_channel=1, classes=10, filters1=30, filters2=30, fc_size=200).cuda()

    with torch.no_grad():
        for idx, p in enumerate(list(network.parameters())):
            temp_file_name = './results/gan_Global_' + str(args.next_round-1) + '_' + str(idx) + '.npy'
            with open(temp_file_name, 'rb') as f:
                params_copy = np.load(f)
                params_copy = torch.from_numpy(params_copy).cuda()
            p.copy_(params_copy)
        
    params = list(network.parameters())
    with torch.no_grad():
        gan_results = []
        for idx in range(len(average_grad)):
            average_grad[idx] = average_grad[idx].reshape(orig_shape[idx])
            params[idx].data.sub_(average_grad[idx])
            gan_results.append(params[idx].data.cpu().numpy())
        temp_file_name = './results/ganAgg_' + str(args.next_round) + '.npy'
        with open(temp_file_name, 'wb') as f:
            np.save(f, gan_results)

    correct = 0
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = DataLoader(torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform))
    with torch.no_grad():
        for feature, target in test_loader:
            feature = feature.cuda()
            target = target.type(torch.long).cuda()
            output = network(feature)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    print('\nTest set, GAN aggregator: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
