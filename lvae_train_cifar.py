from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import argparse
import os
#from keras.utils import to_categorical
from model import LVAE
from datasets_and_loaders.omniglot import OmniglotLoader
from datasets_and_loaders.custom_cifar10 import SubSampleLoaderCreator, CustomCIFAR10

parser = argparse.ArgumentParser(description='PyTorch OSR Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
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

parser.add_argument('--latent_dim', type=int, default=32, help='Dimensionality of extracted feature')

args = parser.parse_args()

latent_dim = args.latent_dim

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

lvae = LVAE(in_ch=3,
            out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
            kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, stride1=1, stride2=2,
            flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
            latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=latent_dim,
            num_class=args.num_classes)

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")

# data loader
default_transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


cifar_10_trainset = CustomCIFAR10(root='/scratch/local/ssd/sagar', train=True, download=False,
                                  transform=default_transform)
cifar_10_testset = CustomCIFAR10(root='/scratch/local/ssd/sagar', train=False, download=False,
                                  transform=default_transform)

cifar_100_trainset = torchvision.datasets.CIFAR100(root='/scratch/local/ssd/sagar', train=True, download=False,
                                  transform=default_transform)

loader_getter = SubSampleLoaderCreator(batch_size=args.batch_size)

train_loader = loader_getter(selected_classes=[0, 1, 8, 9], valid=None, dataset=cifar_10_trainset)
val_loader = loader_getter(selected_classes=[0, 1, 8, 9], valid=None, dataset=cifar_10_testset, num_examples=4000)

unknown_loader = loader_getter(selected_classes=list(range(10)), valid=None, dataset=cifar_100_trainset,
                               num_examples=4000)

# Model
lvae.cuda()
nllloss = nn.NLLLoss().to(device)

save_dir = 'cifar_{}_latent_size'.format(latent_dim)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# optimzer
optimizer = optim.SGD(lvae.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
print('decreasing_lr: ' + str(decreasing_lr))
beta = DeterministicWarmup(n=50, t_max=1)  # Linear warm-up from 0 to 1 over 50 epoch

def train(args, lvae):
    best_val_loss = 1000
    # train
    for epoch in range(args.epochs):
        lvae.train()
        print("Training... Epoch = %d" % epoch)
        correct_train = 0
        open('{}/train_fea.txt'.format(save_dir), 'w').close()
        open('{}/train_tar.txt'.format(save_dir), 'w').close()
        open('{}/train_rec.txt'.format(save_dir), 'w').close()
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
            print("~~~learning rate:", optimizer.param_groups[0]['lr'])
        for batch_idx, (data, target) in enumerate(train_loader):
            target_en = torch.Tensor(target.shape[0], args.num_classes)
            target_en.zero_()
            target_en.scatter_(1, target.view(-1, 1), 1)  # one-hot encoding
            target_en = target_en.to(device)
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)

            loss, mu, output, output_mu, x_re, rec, kl, ce = lvae.loss(data, target, target_en, next(beta), args.lamda)
            rec_loss = (x_re - data).pow(2).sum((3, 2, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outlabel = output.data.max(1)[1]  # get the index of the max log-probability
            correct_train += outlabel.eq(target.view_as(outlabel)).sum().item()

            cor_fea = mu[(outlabel == target)]
            cor_tar = target[(outlabel == target)]
            cor_fea = torch.Tensor.cpu(cor_fea).detach().numpy()
            cor_tar = torch.Tensor.cpu(cor_tar).detach().numpy()
            rec_loss = torch.Tensor.cpu(rec_loss).detach().numpy()
            with open('{}/train_fea.txt'.format(save_dir), 'ab') as f:
                np.savetxt(f, cor_fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/train_tar.txt'.format(save_dir), 'ab') as t:
                np.savetxt(t, cor_tar, fmt='%d', delimiter=' ', newline='\r')
                t.write(b'\n')
            with open('{}/train_rec.txt'.format(save_dir), 'ab') as m:
                np.savetxt(m, rec_loss, fmt='%f', delimiter=' ', newline='\r')
                m.write(b'\n')

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train_batch_loss: {:.6f}={:.6f}+{:.6f}+{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx * len(data) / len(train_loader.dataset),
                           loss.data / (len(data)),
                           rec.data / (len(data)),
                           kl.data / (len(data)),
                           ce.data / (len(data))
                    ))

        train_acc = float(100 * correct_train) / len(train_loader.dataset)
        print('Train_Acc: {}/{} ({:.2f}%)'.format(correct_train, len(train_loader.dataset), train_acc))
# val
        if epoch % args.val_interval == 0 and epoch >= 0:
            lvae.eval()
            correct_val = 0
            total_val_loss = 0
            total_val_rec = 0
            total_val_kl = 0
            total_val_ce = 0
            for data_val, target_val in val_loader:
                target_val_en = torch.Tensor(target_val.shape[0], args.num_classes)
                target_val_en.zero_()
                target_val_en.scatter_(1, target_val.view(-1, 1), 1)  # one-hot encoding
                target_val_en = target_val_en.to(device)
                if args.cuda:
                    data_val, target_val = data_val.cuda(), target_val.cuda()
                with torch.no_grad():
                    data_val, target_val = Variable(data_val), Variable(target_val)

                loss_val, mu_val, output_val, output_mu_val, val_re, rec_val, kl_val, ce_val = lvae.loss(data_val, target_val, target_val_en, next(beta), args.lamda)
                total_val_loss += loss_val.data.detach().item()
                total_val_rec += rec_val.data.detach().item()
                total_val_kl += kl_val.data.detach().item()
                total_val_ce += ce_val.data.detach().item()

                vallabel = output_val.data.max(1)[1]  # get the index of the max log-probability
                correct_val += vallabel.eq(target_val.view_as(vallabel)).sum().item()

            val_loss = total_val_loss / len(val_loader.dataset)
            val_rec = total_val_rec / len(val_loader.dataset)
            val_kl = total_val_kl / len(val_loader.dataset)
            val_ce = total_val_ce / len(val_loader.dataset)
            print('====> Epoch: {} Val loss: {}/{} ({:.4f}={:.4f}+{:.4f}+{:.4f})'.format(epoch, total_val_loss, len(val_loader.dataset), val_loss, val_rec, val_kl, val_ce))
            val_acc = float(100 * correct_val) / len(val_loader.dataset)
            print('Val_Acc: {}/{} ({:.2f}%)'.format(correct_val, len(val_loader.dataset), val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                train_fea = np.loadtxt('{}/train_fea.txt'.format(save_dir))
                train_tar = np.loadtxt('{}/train_tar.txt'.format(save_dir))
                train_rec = np.loadtxt('{}/train_rec.txt'.format(save_dir))
                print('!!!Best Val Epoch: {}, Best Val Loss:{:.4f}'.format(best_val_epoch, best_val_loss))
                #torch.save(lvae, 'lvae%d.pt' % args.lamda)
# test
                open('{}/c100_fea.txt'.format(save_dir), 'w').close()
                open('{}/c100_tar.txt'.format(save_dir), 'w').close()
                open('{}/c100_pre.txt'.format(save_dir), 'w').close()
                open('{}/c100_rec.txt'.format(save_dir), 'w').close()

                for data_test, target_test in val_loader:
                    target_test_en = torch.Tensor(target_test.shape[0], args.num_classes)
                    target_test_en.zero_()
                    target_test_en.scatter_(1, target_test.view(-1, 1), 1)  # one-hot encoding
                    target_test_en = target_test_en.to(device)
                    if args.cuda:
                        data_test, target_test = data_test.cuda(), target_test.cuda()
                    with torch.no_grad():
                        data_test, target_test = Variable(data_test), Variable(target_test)

                    mu_test, output_test, de_test = lvae.test(data_test, target_test_en)
                    output_test = torch.exp(output_test)
                    prob_test = output_test.max(1)[0]  # get the value of the max probability
                    pre_test = output_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    rec_test = (de_test - data_test).pow(2).sum((3, 2, 1))
                    mu_test = torch.Tensor.cpu(mu_test).detach().numpy()
                    target_test = torch.Tensor.cpu(target_test).detach().numpy()
                    pre_test = torch.Tensor.cpu(pre_test).detach().numpy()
                    rec_test = torch.Tensor.cpu(rec_test).detach().numpy()

                    with open('{}/c100_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_test, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/c100_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, target_test, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/c100_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_test, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
                    with open('{}/c100_rec.txt'.format(save_dir), 'ab') as l_test:
                        np.savetxt(l_test, rec_test, fmt='%f', delimiter=' ', newline='\r')
                        l_test.write(b'\n')

                print('Saving model...')
                torch.save(lvae.state_dict(), '{}/model.pt'.format(save_dir))
# cifar100_test
                i_omn = 0
                for data_omn, target_omn in unknown_loader:
                    i_omn += 1
                    tar_omn = torch.from_numpy(args.num_classes * np.ones(target_omn.shape[0]))

                    if args.cuda:
                        data_omn = data_omn.cuda()
                    with torch.no_grad():
                        data_omn = Variable(data_omn)

                    mu_omn, output_omn, de_omn = lvae.test(data_omn, target_test_en)
                    output_omn = torch.exp(output_omn)
                    prob_omn = output_omn.max(1)[0]  # get the value of the max probability
                    pre_omn = output_omn.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    rec_omn = (de_omn - data_omn).pow(2).sum((3, 2, 1))
                    mu_omn = torch.Tensor.cpu(mu_omn).detach().numpy()
                    tar_omn = torch.Tensor.cpu(tar_omn).detach().numpy()
                    pre_omn = torch.Tensor.cpu(pre_omn).detach().numpy()
                    rec_omn = torch.Tensor.cpu(rec_omn).detach().numpy()

                    with open('{}/c100_fea.txt'.format(save_dir), 'ab') as f_test:
                        np.savetxt(f_test, mu_omn, fmt='%f', delimiter=' ', newline='\r')
                        f_test.write(b'\n')
                    with open('{}/c100_tar.txt'.format(save_dir), 'ab') as t_test:
                        np.savetxt(t_test, tar_omn, fmt='%d', delimiter=' ', newline='\r')
                        t_test.write(b'\n')
                    with open('{}/c100_pre.txt'.format(save_dir), 'ab') as p_test:
                        np.savetxt(p_test, pre_omn, fmt='%d', delimiter=' ', newline='\r')
                        p_test.write(b'\n')
                    with open('{}/c100_rec.txt'.format(save_dir), 'ab') as l_test:
                        np.savetxt(l_test, rec_omn, fmt='%f', delimiter=' ', newline='\r')
                        l_test.write(b'\n')

    open('{}/train_fea.txt'.format(save_dir), 'w').close()  # clear
    np.savetxt('{}/train_fea.txt'.format(save_dir), train_fea, delimiter=' ', fmt='%f')
    open('{}/train_tar.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/train_tar.txt'.format(save_dir), train_tar, delimiter=' ', fmt='%d')
    open('{}/train_rec.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/train_rec.txt'.format(save_dir), train_rec, delimiter=' ', fmt='%f')

    fea_omn = np.loadtxt('{}/c100_fea.txt'.format(save_dir))
    tar_omn = np.loadtxt('{}/c100_tar.txt'.format(save_dir))
    pre_omn = np.loadtxt('{}/c100_pre.txt'.format(save_dir))
    rec_omn = np.loadtxt('{}/c100_rec.txt'.format(save_dir))
    fea_omn = fea_omn[:20000, :]
    tar_omn = tar_omn[:20000]
    pre_omn = pre_omn[:20000]
    rec_omn = rec_omn[:20000]
    open('{}/c100_fea.txt'.format(save_dir), 'w').close()  # clear
    np.savetxt('{}/c100_fea.txt'.format(save_dir), fea_omn, delimiter=' ', fmt='%f')
    open('{}/c100_tar.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/c100_tar.txt'.format(save_dir), tar_omn, delimiter=' ', fmt='%d')
    open('{}/c100_pre.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/c100_pre.txt'.format(save_dir), pre_omn, delimiter=' ', fmt='%d')
    open('{}/c100_rec.txt'.format(save_dir), 'w').close()
    np.savetxt('{}/c100_rec.txt'.format(save_dir), rec_omn, delimiter=' ', fmt='%d')

    return best_val_loss, best_val_epoch

best_val_loss, best_val_epoch = train(args, lvae)
print('Finally!Best Epoch: {},  Best Val Loss: {:.4f}'.format(best_val_epoch, best_val_loss))
