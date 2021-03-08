from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

from collections import OrderedDict

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--blocksize', type=int, default=5,
                    help='block size')
parser.add_argument('--numlayers', type=int, default=20,
                    help='maximum number of layers')
parser.add_argument('--numblocks', type=int, default=20,
                    help='maximum number of blocks in each layer')
parser.add_argument('--blockThreshold', type=float, default=1e-3,
                    help='threshold for stop progression in number of layers')
parser.add_argument('--layerThreshold', type=float, default=1e-3,
                    help='threshold for stop progression in number of blocks in each layer')
parser.add_argument('--topology', type=list, default=[],
                    help='topology of the model')
parser.add_argument('--model_saved_name', type=str, default='pgcn',
                    help='name of the saved model')
parser.add_argument('--dataset', type=str, default='cora',
                    help='name of dataset')
parser.add_argument('--datapath', type=str, default='../data/',
                    help='path of dataset')

args = parser.parse_args()
######## load data #########
adj, features, labels, idx_train, idx_val, idx_test, train_mask, Lplcn, labels_onehot = load_data(path="../data/",
                                                                                                  dataset=args.dataset)


######## load model #########
def load_model(args):
    model = GCN(infeat=features.shape[1],
                bsize=args.blocksize,
                topology=args.topology,
                n_class=labels.max().item() + 1,
                dropout=args.dropout)
    return model


########## model initialization with finetuned weights ####################

def init_weights(model, args, layer_init, block_iter):
    if block_iter == 0:
        weights = torch.load('./saved_models/' + args.model_saved_name + '-' + str(len(args.topology) - 1) + '-' + str(
            args.topology[-2]) + '.pt')
    else:
        weights = torch.load('./saved_models/' + args.model_saved_name + '-' + str(len(args.topology)) + '-' + str(
            args.topology[-1] - 1) + '.pt')

    weights = OrderedDict([[k, v.cuda()] for k, v in weights.items()])
    old_keys = list(weights.keys())
    for current_key in model.state_dict():
        if ('bias') in current_key:
            if current_key in old_keys:
                W_ = model.state_dict()[current_key]
                old_sh = weights[current_key].shape
                W_[:old_sh[0]] = weights[current_key]
                new_state_dict = OrderedDict({current_key: W_})
                model.load_state_dict(new_state_dict, strict=False)
        if 'weight' in current_key:
            if current_key in old_keys:
                W_ = model.state_dict()[current_key]
                old_sh = weights[current_key].shape
                W_[:, :old_sh[1]] = weights[current_key]
                new_state_dict = OrderedDict({current_key: W_})
                model.load_state_dict(new_state_dict, strict=False)
        return model

    ######## training #########


def train(model, args, optimizer, epoch, adj, features, labels, idx_train, idx_val, idx_test, save_model=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    if save_model:
        state_dict = model.state_dict()
        weights = OrderedDict([[k, v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, './saved_models/' + args.model_saved_name + '-' + str(len(args.topology)) + '-' + str(
            args.topology[-1]) + '.pt')

    return loss_train, acc_train, loss_val, acc_val


######## testing #########
def test(model, args, adj, features, labels, idx_train, idx_val, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test


################# PGCN algorithm ###############
def PGCN(args, adj, features, labels, idx_train, idx_val, idx_test, train_mask, Lplcn, labels_onehot):
    acc_layer_old = 1e-10
    acc_block_old = 1e-10
    acc_layer_new = 1e-10
    acc_block_new = 1e-10
    loss_train_list = []
    acc_train_list = []
    loss_val_list = []
    acc_val_list = []
    loss_test_list = []
    acc_test_list = []
    t_total = time.time()
    for layer_iter in range(args.numlayers):
        args.topology.append(0)  ### add one layer
        for block_iter in range(args.numblocks):
            print('######################################################################\n')
            print('layer.' + str(layer_iter) + '_block.' + str(block_iter))
            print('\n######################################################################\n')
            args.topology[layer_iter] = args.topology[layer_iter] + 1  ### add one block
            model = load_model(args)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            ########## initialize the weights #############
            if (layer_iter > 0 or block_iter > 0):
                model = init_weights(model, args, layer_iter, block_iter)  ################
            ############# initialize the output layer (least_square) ##################
            H = model(features, adj, ls=True)
            H = H.detach().cpu().numpy()
            N_ = H.shape[0]
            D_ = H.shape[1]
            I_N = np.eye(N_, N_)
            coef = np.divide(1, (0.1 * (N_ ** 2)))
            Lplcn = I_N + (coef * Lplcn)
            tmp = np.matmul(np.transpose(H), Lplcn)
            tmp = np.matmul(tmp, H)
            xTx = np.linalg.pinv(tmp + ((1 / 0.1) * np.eye(D_, D_)))
            mask = np.array(train_mask, dtype=int)
            H_train = np.matmul(np.diag(mask), H)
            labels_train = np.matmul(np.diag(mask), labels_onehot)
            xTy = np.matmul(np.transpose(H_train), labels_train)
            O_ = np.matmul(xTx, xTy)
            O_ = torch.from_numpy(O_)
            for current_key in model.state_dict():
                if ('outlayer.weight') in current_key:
                    new_state_dict = OrderedDict({current_key: O_})
                    model.load_state_dict(new_state_dict, strict=False)
            #####################################################
            if args.cuda:
                model.cuda()
                features = features.cuda()
                adj = adj.cuda()
                labels = labels.cuda()
                idx_train = idx_train.cuda()
                idx_val = idx_val.cuda()
                idx_test = idx_test.cuda()
            ############ train a new block ############
            for epoch in range(args.epochs):
                save_model = (epoch + 1 == args.epochs)
                loss_train, acc_train, loss_val, acc_val = train(model, args, optimizer, epoch, adj, features, labels,
                                                                 idx_train, idx_val, idx_test, save_model=save_model)
                acc_block_new = acc_val
                loss_train_list.append(loss_train.item())
                acc_train_list.append(acc_train.item())
                loss_val_list.append(loss_val.item())
                acc_val_list.append(acc_val.item())
            print("Training the new block is finished! \n ")
            ########## test the new block #############
            loss_test, acc_test = test(model, args, adj, features, labels, idx_train, idx_val, idx_test)
            loss_test_list.append(loss_test.item())
            acc_test_list.append(acc_test.item())
            ############# check the block progression #########
            if block_iter > 0:
                r_b = (acc_block_new - acc_block_old) / acc_block_old
                if r_b <= args.blockThreshold:
                    args.topology[layer_iter] = args.topology[layer_iter] - 1
                    print('block' + str(block_iter) + 'of layer' + str(layer_iter) + 'is removed \n')
                    print('block progression is stopped in layer' + str(layer_iter))
                    model = load_model(args)
                    weights = torch.load(
                        './saved_models/' + args.model_saved_name + '-' + str(len(args.topology)) + '-' + str(
                            args.topology[-1]) + '.pt')
                    weights = OrderedDict([[k, v.cuda()] for k, v in weights.items()])
                    model.load_state_dict(weights)
                    break
            acc_block_old = acc_block_new
        acc_layer_new = acc_block_old
        if layer_iter > 0:
            r_l = (acc_layer_new - acc_layer_old) / acc_layer_old
            if r_l <= args.layerThreshold:
                args.topology.pop()  ### remove the topology of last layer
                print('layer' + str(layer_iter) + 'is removed \n')
                print('layer progression is stopped')
                model = load_model(args)
                weights = torch.load(
                    './saved_models/' + args.model_saved_name + '-' + str(len(args.topology)) + '-' + str(
                        args.topology[-1]) + '.pt')
                weights = OrderedDict([[k, v.cuda()] for k, v in weights.items()])
                model.load_state_dict(weights)
                break
        acc_layer_old = acc_block_new
    print('finetuning the optimized model')
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val = train(model, args, optimizer, epoch, adj, features, labels,
                                                         idx_train, idx_val, idx_test, save_model=True)
    loss_test, acc_test = test(model, args, adj, features, labels, idx_train, idx_val, idx_test)
    print("\n Total time elapsed: {:.4f}s".format(time.time() - t_total))
    np.save('./results/' + args.dataset + '/loss_train_list.npy', loss_train_list)
    np.save('./results/' + args.dataset + '/acc_train_list.npy', acc_train_list)
    np.save('./results/' + args.dataset + '/loss_val_list.npy', loss_val_list)
    np.save('./results/' + args.dataset + '/acc_val_list.npy', acc_val_list)
    np.save('./results/' + args.dataset + '/loss_test_list.npy', loss_test_list)
    np.save('./results/' + args.dataset + '/acc_test_list.npy', acc_test_list)
    print(args.topology)


args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

PGCN(args, adj, features, labels, idx_train, idx_val, idx_test, train_mask, Lplcn, labels_onehot)


