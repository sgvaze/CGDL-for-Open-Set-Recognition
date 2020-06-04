from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
#from keras.utils import to_categorical
from model import Classifier32
from datasets_and_loaders.omniglot import OmniglotLoader

from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 1e-3)')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.01, help='momentum (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='60,100,150', help='decreasing strategy')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=5, help='how many epochs to wait before another val')
parser.add_argument('--test_interval', type=int, default=5, help='how many epochs to wait before another test')
parser.add_argument('--lamda', type=int, default=100, help='lamda in loss function')
args = parser.parse_args()

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class DeterministicWarmup(object):
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t

model = Classifier32(latent_size=32)

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

# data loader
train_dataset = datasets.MNIST('/scratch/shared/beegfs/sagar/datasets', download=True, train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

val_dataset = datasets.MNIST('/scratch/shared/beegfs/sagar/datasets', download=True, train=False,
                              transform=transforms.Compose([

                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

omn_loader = OmniglotLoader(batch_size=args.batch_size, train=False, drop_last=False)
# omn_loader = deepcopy(val_loader)

# Model
model.cuda()
ce_loss = nn.CrossEntropyLoss().to(device)

save_dir = 'reg_classifier'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# optimzer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
beta = DeterministicWarmup(n=50, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch

def train(args, lvae):
    best_val_loss = float('inf')
    # train
    for epoch in range(args.epochs):

        model.train()
        print("Training... Epoch = %d" % epoch)
        correct_train = 0
        open('{}/train_fea.txt'.format(save_dir), 'w').close()
        open('{}/train_tar.txt'.format(save_dir), 'w').close()
        open('{}/train_rec.txt'.format(save_dir), 'w').close()

        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])

        for batch_idx, (data, target) in enumerate(train_loader):

            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)

            output, mu = model(data, return_features=True)
            loss = ce_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outlabel = output.data.max(1)[1]  # get the index of the max log-probability
            correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()

            cor_fea = mu[(outlabel == target)]
            cor_tar = target[(outlabel == target)]
            cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
            cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()

            with open('{}/train_fea.txt'.format(save_dir), 'ab') as f:
                np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/train_tar.txt'.format(save_dir), 'ab') as t:
                np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
                t.write(b'\n')

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train_batch_loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx * len(data) / len(train_loader.dataset),
                           loss.data / (len(data))
                    ))

        train_acc = float(100 * correct_train) / len(train_loader.dataset)
        print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, len(train_loader.dataset), train_acc))
# val
        if epoch % args.val_interval == 0 and epoch >= 0:

            model.eval()

            correct_val = 0
            total_val_loss = 0

            for data_val, target_val in val_loader:

                if args.cuda:
                    data_val, target_val = data_val.cuda(), target_val.cuda()
                with torch.no_grad():
                    data_val, target_val = Variable(data_val), Variable(target_val)

                output_val, mu = model(data_val, return_features=True)
                loss_val = ce_loss(output_val, target_val)

                total_val_loss += loss_val.data.detach().item()

                vallabel = output_val.data.max(1)[1]  # get the index of the max log-probability

                correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

            val_loss = total_val_loss / len(val_loader.dataset)
            print('====> Epoch: {} Val loss: {})'.format(epoch, val_loss))
            val_acc = float(100 * correct_val) / len(val_loader.dataset)
            print('Val_Acc: {}/{} ({:.2f}%)'.format(correct_val, len(val_loader.dataset), val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                train_fea = np.loadtxt('{}/train_fea.txt'.format(save_dir))
                train_tar = np.loadtxt('{}/train_tar.txt'.format(save_dir))
                print('!!!Best Val Epoch: {}, Best Val Loss:{:.4f}'.format(best_val_epoch, best_val_loss))
                #torch.save(lvae, 'lvae%d.pt' % args.lamda)
# test
                open('{}/omn_fea.txt'.format(save_dir), 'w').close()
                open('{}/omn_tar.txt'.format(save_dir), 'w').close()
                open('{}/omn_pre.txt'.format(save_dir), 'w').close()
                open('{}/omn_rec.txt'.format(save_dir), 'w').close()

                open('{}/mnist_noise_fea.txt'.format(save_dir), 'w').close()
                open('{}/mnist_noise_tar.txt'.format(save_dir), 'w').close()
                open('{}/mnist_noise_pre.txt'.format(save_dir), 'w').close()
                open('{}/mnist_noise_rec.txt'.format(save_dir), 'w').close()

                open('{}/noise_fea.txt'.format(save_dir), 'w').close()
                open('{}/noise_tar.txt'.format(save_dir), 'w').close()
                open('{}/noise_pre.txt'.format(save_dir), 'w').close()
                open('{}/noise_rec.txt'.format(save_dir), 'w').close()

                for data_test, target_test in val_loader:

                    if args.cuda:
                        data_test, target_test = data_test.cuda(), target_test.cuda()
                    with torch.no_grad():
                        data_test, target_test = Variable(data_test), Variable(target_test)

                    output_test, mu_test = model(data_test, return_features=True)
                    output_test = torch.exp(output_test)
                    prob_test = output_test.max(1)[0]  # get the value of the max probability
                    pre_test = output_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    mu_test = torch.Tensor.cpu(mu_test).detach().numpy()
                    target_test = torch.Tensor.cpu(target_test).detach().numpy()
                    pre_test = torch.Tensor.cpu(pre_test).detach().numpy()

                    with open('{}/omn_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/omn_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/omn_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')

                    with open('{}/mnist_noise_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/mnist_noise_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/mnist_noise_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')

                    with open('{}/noise_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/noise_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/noise_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
# omn_test
                print('Saving model...')
                torch.save(lvae.state_dict(), '{}/model.pt'.format(save_dir))

                i_omn = 0
                for data_omn, target_omn in omn_loader:

                    i_omn += 1
                    tar_omn = torch.from_numpy(args.num_classes * np.ones(target_omn.shape[0]))

                    if args.cuda:
                        data_omn = data_omn.cuda()
                    with torch.no_grad():
                        data_omn = Variable(data_omn)

                    output_omn, mu_omn = model(data_omn, return_features=True)
                    output_omn = torch.exp(output_omn)
                    prob_omn = output_omn.max(1)[0]  # get the value of the max probability
                    pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability

                    mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
                    tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
                    pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()

                    with open('{}/omn_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/omn_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/omn_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')

# mnist_noise_test

                for data_test, target_test in val_loader:

                    tar_mnist_noise = torch.from_numpy(args.num_classes * np.ones(target_test.shape[0]))
                    noise = torch.from_numpy(np.random.rand(data_test.shape[0], 1, 28, 28)).float()
                    data_mnist_noise = data_test.add(noise)
                    if args.cuda:
                        data_mnist_noise = data_mnist_noise.cuda()
                    with torch.no_grad():
                        data_mnist_noise = Variable(data_mnist_noise)

                    output_mnist_noise, mu_mnist_noise = model(data_mnist_noise, return_features=True)

                    output_mnist_noise = torch.exp(output_mnist_noise)
                    prob_mnist_noise = output_mnist_noise.max(1)[0]  # get the value of the max probability
                    pre_mnist_noise = output_mnist_noise.max(1, keepdim=True)[1]  # get the index of the max log-probability

                    mu_mnist_noise = torch.Tensor.cpu(mu_mnist_noise).detach().numpy()
                    tar_mnist_noise = torch.Tensor.cpu(tar_mnist_noise).detach().numpy()
                    pre_mnist_noise = torch.Tensor.cpu(pre_mnist_noise).detach().numpy()

                    with open('{}/mnist_noise_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_mnist_noise, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/mnist_noise_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, tar_mnist_noise, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/mnist_noise_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_mnist_noise, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')

# noise_test
                for data_test, target_test in val_loader:
                    tar_noise = torch.from_numpy(args.num_classes * np.ones(target_test.shape[0]))
                    data_noise = torch.from_numpy(np.random.rand(data_test.shape[0], 1, 28, 28)).float()
                    if args.cuda:
                        data_noise = data_noise.cuda()
                    with torch.no_grad():
                        data_noise = Variable(data_noise)

                    output_noise, mu_noise = model(data_noise, return_features=True)

                    output_noise = torch.exp(output_noise)
                    prob_noise = output_noise.max(1)[0]  # get the value of the max probability
                    pre_noise = output_noise.max(1, keepdim=True)[1]  # get the index of the max log-probability

                    mu_noise = torch.Tensor.cpu(mu_noise).detach().numpy()
                    tar_noise = torch.Tensor.cpu(tar_noise).detach().numpy()
                    pre_noise = torch.Tensor.cpu(pre_noise).detach().numpy()

                    with open('{}/noise_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_noise, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/noise_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, tar_noise, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/noise_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_noise, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')


    open('{}/train_fea.txt'.format(save_dir), 'w').close()  # clear
    np.savetxt('{}/train_fea.txt'.format(save_dir), train_fea, delimiter=' ', fmt='%f')
    open('{}/train_tar.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/train_tar.txt'.format(save_dir), train_tar, delimiter=' ', fmt='%d')

    fea_omn = np.loadtxt('{}/omn_fea.txt'.format(save_dir))
    tar_omn = np.loadtxt('{}/omn_tar.txt'.format(save_dir))
    pre_omn = np.loadtxt('{}/omn_pre.txt'.format(save_dir))
    fea_omn = fea_omn[:20000, :]
    tar_omn = tar_omn[:20000]
    pre_omn = pre_omn[:20000]
    open('{}/omn_fea.txt'.format(save_dir), 'w').close()  # clear
    np.savetxt('{}/omn_fea.txt'.format(save_dir), fea_omn, delimiter=' ', fmt='%f')
    open('{}/omn_tar.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/omn_tar.txt'.format(save_dir), tar_omn, delimiter=' ', fmt='%d')
    open('{}/omn_pre.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/omn_pre.txt'.format(save_dir), pre_omn, delimiter=' ', fmt='%d')

    return best_val_loss, best_val_epoch

best_val_loss, best_val_epoch = train(args, model)
print('Finally!Best Epoch: {},  Best Val Loss: {:.4f}'.format(best_val_epoch, best_val_loss))
