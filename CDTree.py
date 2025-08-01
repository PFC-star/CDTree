"""
Main script for training and evaluating the CDTree.
"""
from __future__ import print_function
import argparse
import os
import sys
import json
import time
import copy
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import ConcatDataset

import matplotlib
import utils.ops as ops
matplotlib.use('agg')

# from data import get_dataloaders, get_dataset_details
from utils.data_manager import DataManager
from utils.models import Tree, One
from utils.ops import get_params_node, ChunkSampler
from utils.utils import define_node, get_scheduler, set_random_seed

# Experiment settings
parser = argparse.ArgumentParser(description='CDTree')
parser.add_argument('--experiment', '-e', dest='experiment', default='tree', help='experiment name')
parser.add_argument('--subexperiment','-sube', dest='subexperiment', default='', help='experiment name')

parser.add_argument('--dataset', default='mnist', help='dataset type')
parser.add_argument('--testonly', default=False, type=bool, help='test-only')
parser.add_argument('--traininit', default=False, type=bool, help='train-init')
parser.add_argument('--testeachclass', default=False, type=bool, help='train-init')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--gpu', type=str, default="", help='which GPU to use')
parser.add_argument('--seed', type=int, default=2024, metavar='S', help='random seed')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of threads for data-loader')

# Optimization settings:
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--augmentation_on', action='store_true', default=False, help='perform data augmentation')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--scheduler', type=str, default="", help='learning rate scheduler')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum')
parser.add_argument('--valid_ratio', '-vr', dest='valid_ratio', type=float, default=0.1, metavar='LR', help='validation set ratio')

parser.add_argument('--criteria', default='avg_valid_loss', help='growth criteria')
parser.add_argument('--epochs_node', type=int, default=50, metavar='N', help='max number of epochs to train per node during the growth phase')
parser.add_argument('--epochs_finetune', type=int, default=100, metavar='N', help='number of epochs for the refinement phase')
parser.add_argument('--epochs_patience', type=int, default=5, metavar='N', help='number of epochs to be waited without improvement at each node during the growth phase')
parser.add_argument('--epochs_da', type=int, default=10, metavar='N', help='number of epochs for domain adaptation')
parser.add_argument('--maxdepth', type=int, default=10, help='maximum depth of tree')
parser.add_argument('--finetune_during_growth', action='store_true', default=False, help='refine the tree globally during the growth phase')
parser.add_argument('--epochs_finetune_node', type=int, default=1, metavar='N', help='number of epochs to perform global refinement at each node during the growth phase')
parser.add_argument('--finetune', action='store_true', default=False, help='refine the tree globally during the growth phase')
parser.add_argument('--finetune_replay', action='store_true', default=False, help='refine the tree globally with ori data')
parser.add_argument('--onlyda', action='store_true', default=False, help='only domain adaptation')
parser.add_argument('--onlytree', action='store_true', default=False, help='only grow the tree')
parser.add_argument('--optimal', action='store_true', default=False, help='optimal')


# Solver, router and transformer modules:
parser.add_argument('--router_ver', '-r_ver', dest='router_ver', type=int, default=1, help='default router version')
parser.add_argument('--router_ngf', '-r_ngf', dest='router_ngf', type=int, default=1, help='number of feature maps in routing function')
parser.add_argument('--router_k', '-r_k', dest='router_k', type=int, default=28, help='kernel size in routing function')
parser.add_argument('--router_dropout_prob', '-r_drop', dest='router_dropout_prob', type=float, default=0.0, help='drop-out probabilities for router modules.')

parser.add_argument('--transformer_ver', '-t_ver', dest='transformer_ver', type=int, default=1, help='default transformer version: identity')
parser.add_argument('--transformer_ngf', '-t_ngf', dest='transformer_ngf', type=int, default=3, help='number of feature maps in residual transformer')
parser.add_argument('--transformer_k', '-t_k', dest='transformer_k', type=int, default=5, help='kernel size in transfomer function')
parser.add_argument('--transformer_expansion_rate', '-t_expr', dest='transformer_expansion_rate', type=int, default=1, help='default transformer expansion rate')
parser.add_argument('--transformer_reduction_rate', '-t_redr', dest='transformer_reduction_rate', type=int, default=2, help='default transformer reduction rate')

parser.add_argument('--solver_ver', '-s_ver', dest='solver_ver', type=int, default=1, help='default router version')
parser.add_argument('--solver_inherit', '-s_inh', dest='solver_inherit',  action='store_true', help='inherit the parameters of the solver when defining two new ones for splitting a node')
parser.add_argument('--solver_dropout_prob', '-s_drop', dest='solver_dropout_prob', type=float, default=0.0, help='drop-out probabilities for solver modules.')

parser.add_argument('--downsample_interval', '-ds_int', dest='downsample_interval', type=int, default=0, help='interval between two downsampling operations via transformers i.e. 0 = downsample at every transformer')
parser.add_argument('--batch_norm', '-bn', dest='batch_norm', action='store_true', default=False, help='turn batch norm on')

args = parser.parse_args()

# GPUs devices:
args.cuda = not args.no_cuda and torch.cuda.is_available()
# if args.gpu:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set the seed for repeatability
set_random_seed(args.seed, args.cuda)



# Define a dictionary for post-inspection of the model:
def init_records():
    records = vars(args)
    save_dir = "./experiments/{}/{}/{}/{}".format(
        args.dataset, args.experiment, args.subexperiment, 'checkpoints',
    )
    record_name = save_dir + '/records.json'
    if os.path.exists(record_name):
        with open(record_name, 'r') as file:
            records = json.load(file)
    else:
        records['time'] = 0.0
        records['counter'] = 0  # number of optimization steps

        records['train_nodes'] = []  # node indices for each logging interval
        records['train_loss'] = []   # avg. train. loss for each log interval
        records['train_best_loss'] = np.inf  # best train. loss
        records['train_epoch_loss'] = []  # epoch wise train loss

        records['valid_nodes'] = []
        records['valid_best_loss_nodes'] = []
        records['valid_best_loss_nodes_split'] = []
        records['valid_best_loss_nodes_ext'] = []
        records['valid_best_root_nosplit'] = np.inf
        records['valid_best_loss'] = np.inf
        records['valid_best_accuracy'] = 0.0
        records['valid_epoch_loss'] = []
        records['valid_epoch_accuracy'] = []

        records['test_best_loss'] = np.inf
        records['test_best_accuracy'] = 0.0
        records['test_epoch_loss'] = []
        records['test_epoch_accuracy'] = []
    return records



# -----------------------------  Components ----------------------------------
def train(model, data_loader, optimizer, node_idx):
    """ Train step"""
    model.train()
    train_loss = 0
    no_points = 0
    train_epoch_loss = 0
    #criterion = nn.CrossEntropyLoss().cuda()
    

    # train the model
    
    for batch_idx, (i, x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y_pred, p_out = model(x)

        loss = F.nll_loss(y_pred, y)
        train_epoch_loss += loss.item() * y.size(0)
        train_loss += loss.item() * y.size(0)
        loss.backward()
        optimizer.step()

        records['counter'] += 1
        no_points += y.size(0)

        if batch_idx % args.log_interval == 0:
            # show the interval-wise average loss:
            train_loss /= no_points
            records['train_loss'].append(train_loss)
            records['train_nodes'].append(node_idx)

            sys.stdout.flush()
            sys.stdout.write('\t      [{}/{} ({:.0f}%)]      Loss: {:.6f} \r'.
                    format(batch_idx*len(x), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader.dataset), train_loss))
            

            train_loss = 0
            no_points = 0
    

    # compute average train loss for the epoch
    train_epoch_loss /= len(data_loader.dataset)
    records['train_epoch_loss'].append(train_epoch_loss)
    if train_epoch_loss < records['train_best_loss']:
        records['train_best_loss'] = train_epoch_loss

    print('\nTrain set: Average loss: {:.4f}'.format(train_epoch_loss))

def train_replay(model, data_loader_init, data_loader, optimizer, node_idx):
    """ Train step"""
    model.train()
    train_loss = 0
    no_points = 0
    train_epoch_loss = 0
    
    tree_da_heads = torch.nn.Sequential()
    for i, node in enumerate(tree_da_head[0]):
        if i == 0:
            tree_da_heads.add_module('transform', tree_da_head[0][node])

    # train the model
    # 融到一起
    dataloader_iterator1 = iter(data_loader_init)
    for batch_idx, (_, x2, y2) in enumerate(data_loader): # data_loader
        try:
            (_, x1, y1) = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(data_loader_init)
            (_, x1, y1) = next(dataloader_iterator1)

        optimizer.zero_grad()
        if args.cuda:
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        x1, y1, x2, y2 = Variable(x1), Variable(y1), Variable(x2), Variable(y2)
        inputs = tree_da_heads.transform(x1)
        y_pred, p_out = model.forward_without_0(inputs)
        y_pred_, p_out_ = model(x2)

        loss = F.nll_loss(y_pred, y1)
        loss += F.nll_loss(y_pred_, y2)
        train_epoch_loss += loss.item() * (y1.size(0)+y2.size(0))
        train_loss += loss.item() * (y1.size(0)+y2.size(0))
        loss.backward()
        optimizer.step()

        records['counter'] += 1
        no_points += (y1.size(0)+y2.size(0))

        if batch_idx % args.log_interval == 0:
            # show the interval-wise average loss:
            train_loss /= no_points
            records['train_loss'].append(train_loss)
            records['train_nodes'].append(node_idx)

            sys.stdout.flush()
            sys.stdout.write('\t      [{}/{} ({:.0f}%)]      Loss: {:.6f} \r'.
                    format(batch_idx*(len(x1)+len(x2)), NUM_TRAIN*2,
                    100. * batch_idx / (NUM_TRAIN*2), train_loss))

            train_loss = 0
            no_points = 0
    

    # compute average train loss for the epoch
    train_epoch_loss /= NUM_TRAIN*2
    records['train_epoch_loss'].append(train_epoch_loss)
    if train_epoch_loss < records['train_best_loss']:
        records['train_best_loss'] = train_epoch_loss

    print('\nTrain set: Average loss: {:.4f}'.format(train_epoch_loss))

def train_mmd(model, tree_modules, old_data_loader, data_loader, node_idx, debug=0):
    """ Train step"""
    model.train()
    train_loss = 0
    no_points = 0
    train_epoch_loss = 0
    MMD_loss = ops.MMD_loss().cuda()


    tree_modules_nn = torch.nn.ModuleList()
    for i, node in enumerate(tree_modules):
        node_modules = torch.nn.Sequential()
        node_modules.add_module('transform', node["transform"])
        node_modules.add_module('classifier', node["classifier"])
        node_modules.add_module('router', node["router"])
        tree_modules_nn.append(node_modules)

    for i, (n, p) in enumerate(model.named_parameters()):
        if n.startswith('tree_modules.0.transform'):
            p.requires_grad = True
        else:
            p.requires_grad = False
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
    )

    dataloader_iterator1 = iter(old_data_loader)
    for batch_idx, (_, x2, y2) in enumerate(data_loader):
        try:
            (_, x1, y1) = next(dataloader_iterator1)
        except StopIteration:
            dataloader_iterator1 = iter(old_data_loader)
            (_, x1, y1) = next(dataloader_iterator1)

        optimizer.zero_grad()
        if args.cuda:
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        x1, y1, x2, y2 = Variable(x1), Variable(y1), Variable(x2), Variable(y2)
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        y_pred, p_out = model(x2) # TODO 可以没有
        loss = F.nll_loss(y_pred, y2) # TODO 可以没有
        da_out = tree_modules_nn[0].transform(x1)
        da_out_new = model.forward_for_da(x2)
        min_len = min(len(x1),len(x2))
        da_old = da_out.view(da_out.size(0), -1).cuda()
        da_new = da_out_new.view(da_out_new.size(0), -1).cuda()
        mmd_loss = MMD_loss(da_old, da_new)
        loss = 0.9*loss + 0.1*mmd_loss

        # # 余弦相似度
        # similarity = F.cosine_similarity(da_out[:min_len,:, :, :], da_out_new[:min_len,:, :, :], dim=-1)
        # # loss += 0.1*(1 - similarity.mean())
        # loss += 0.1*torch.exp(-similarity.mean())
        # # loss = MSE_loss(da_out[:min_len,:, :, :], da_out_new[:min_len,:, :, :])

    
        train_epoch_loss += loss.item() * y2.size(0)
        train_loss += loss.item() * y2.size(0)
        loss.backward()
        optimizer.step()

        tree_modules = model.update_tree_modules()

        records['counter'] += 1
        no_points += y2.size(0)

        if batch_idx % args.log_interval == 0:
            # show the interval-wise average loss:
            train_loss /= no_points
            records['train_loss'].append(train_loss)
            records['train_nodes'].append(node_idx)

            sys.stdout.flush()
            sys.stdout.write('\t      [{}/{} ({:.0f}%)]      Loss: {:.6f} \r'.
                    format(batch_idx*len(x2), NUM_TRAIN,
                    100. * batch_idx / NUM_TRAIN, train_loss))

            train_loss = 0
            no_points = 0
        

    # compute average train loss for the epoch
    train_epoch_loss /= NUM_TRAIN
    records['train_epoch_loss'].append(train_epoch_loss)
    if train_epoch_loss < records['train_best_loss']:
        records['train_best_loss'] = train_epoch_loss

    print('\nTrain set: Average loss: {:.4f}'.format(train_epoch_loss))
    tree_modules = model.update_tree_modules()
    return tree_modules


def valid(model, data_loader, node_idx, struct):
    """ Validation step """
    model.eval()
    valid_epoch_loss = 0
    correct = 0
    save_ckp = False

    with torch.no_grad():
        for _, data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)

            # sum up batch loss
            valid_epoch_loss += F.nll_loss(
                output, target, size_average=False,
            ).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_epoch_loss /= len(data_loader.dataset)
    valid_epoch_accuracy = 100. * correct / len(data_loader.dataset)
    records['valid_epoch_loss'].append(valid_epoch_loss)
    records['valid_epoch_accuracy'].append(valid_epoch_accuracy.tolist())

    if valid_epoch_loss < records['valid_best_loss']:
        records['valid_best_loss'] = valid_epoch_loss

    if valid_epoch_accuracy > records['valid_best_accuracy']:
        records['valid_best_accuracy'] = valid_epoch_accuracy.tolist()

    # see if the current node is root and undergoing the initial training
    # prior to the growth phase.
    is_init_root_train = not model.split and not model.extend and node_idx == 0

    # save the best split model during node-wise training as model_tmp.pth
    if not is_init_root_train and model.split and \
            valid_epoch_loss < records['valid_best_loss_nodes_split'][node_idx]:
        records['valid_best_loss_nodes_split'][node_idx] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records, '_tmp')
        save_ckp = True

    # save the best extended model during node-wise training as model_ext.pth
    if not is_init_root_train and model.extend and \
            valid_epoch_loss < records['valid_best_loss_nodes_ext'][node_idx]:
        records['valid_best_loss_nodes_ext'][node_idx] = valid_epoch_loss
        checkpoint_model('model_ext.pth', model=model)
        checkpoint_msc(struct, records, '_ext')
        save_ckp = True

    # separately store best performance for the initial root training
    if is_init_root_train \
            and valid_epoch_loss < records['valid_best_root_nosplit']:
        records['valid_best_root_nosplit'] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records, '_tmp')
        save_ckp = True

    # saving model during the refinement (fine-tuning) phase
    if not is_init_root_train and not model.init_train:
        if valid_epoch_loss < records['valid_best_loss_nodes'][node_idx]:
            records['valid_best_loss_nodes'][node_idx] = valid_epoch_loss
            if not model.split and not model.extend:
                checkpoint_model('model_tmp.pth', model=model)
                checkpoint_msc(struct, records, '_tmp')
                save_ckp = True
    if not is_init_root_train and model.init_train:
        if valid_epoch_loss < np.inf:
            records['valid_best_loss_nodes'][node_idx] = valid_epoch_loss
            if not model.split and not model.extend:
                checkpoint_model('model_tmp.pth', model=model)
                checkpoint_msc(struct, records, '_tmp')
                save_ckp = True

    end = time.time()
    records['time'] = end - start
    print(
        'Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            valid_epoch_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), records['time'],
        ),
    )
    return valid_epoch_loss, save_ckp, valid_epoch_accuracy

def valid_class(model, data_loader, node_idx, struct):
    """ Validation step """
    model.eval()
    valid_epoch_loss = 0
    correct = 0
    save_ckp = False

    with torch.no_grad():
        for _, data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)

            # sum up batch loss
            valid_epoch_loss += F.nll_loss(
                output, target, size_average=False,
            ).item()
            if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet200' :
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            else:
                _, pred = output.topk(5, 1, True, True)
                pred = pred.t()
                temp = pred.eq(target.view(1, -1).expand_as(pred))
                correct += temp[:5].contiguous().view(-1).float().sum(0)

    valid_epoch_loss /= len(data_loader.dataset)
    valid_epoch_accuracy = 100. * correct / len(data_loader.dataset)
    records['valid_epoch_loss'].append(valid_epoch_loss)
    records['valid_epoch_accuracy'].append(valid_epoch_accuracy.tolist())

    if valid_epoch_loss < records['valid_best_loss']:
        records['valid_best_loss'] = valid_epoch_loss
        save_ckp = True

    if valid_epoch_accuracy > records['valid_best_accuracy']:
        records['valid_best_accuracy'] = valid_epoch_accuracy.tolist()
        save_ckp = True


    is_init_root_train = not model.split and not model.extend and node_idx == 0

    if is_init_root_train \
            and valid_epoch_loss < records['valid_best_root_nosplit']:
        records['valid_best_root_nosplit'] = valid_epoch_loss
        save_ckp = True

    if save_ckp and not model.extend:
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records, '_tmp')
    elif save_ckp and model.extend:
        checkpoint_model('model_ext.pth', model=model)
        checkpoint_msc(struct, records, '_ext')

    end = time.time()
    records['time'] = end - start
    print(
        'Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            valid_epoch_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), records['time'],
        ),
    )
    return valid_epoch_loss, save_ckp, valid_epoch_accuracy

def valid_replay(model, data_loader_init, data_loader, node_idx, struct):
    """ Validation step """
    model.eval()
    valid_epoch_loss = 0
    correct = 0
    save_ckp = False

    tree_da_heads = torch.nn.Sequential()
    for i, node in enumerate(tree_da_head[0]):
        if i == 0:
            tree_da_heads.add_module('transform', tree_da_head[0][node])


    with torch.no_grad():
        dataloader_iterator1 = iter(data_loader_init)
        for batch_idx, (_, x2, y2) in enumerate(data_loader): # data_loader
            try:
                (_, x1, y1) = next(dataloader_iterator1)
            except StopIteration:
                dataloader_iterator1 = iter(data_loader_init)
                (_, x1, y1) = next(dataloader_iterator1)
            if args.cuda:
                x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
            x1, y1, x2, y2 = Variable(x1), Variable(y1), Variable(x2), Variable(y2)
            inputs = tree_da_heads.transform(x1)
            output = model.forward_without_0(inputs)
            output_ = model(x2)

            # sum up batch loss
            valid_epoch_loss += F.nll_loss(
                output, y1, size_average=False,
            ).item()
            valid_epoch_loss += F.nll_loss(
                output_, y2, size_average=False,
            ).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y1.data.view_as(pred)).cpu().sum()
            pred = output_.data.max(1, keepdim=True)[1]
            correct += pred.eq(y2.data.view_as(pred)).cpu().sum()

    valid_epoch_loss /= (NUM_VALID*2)
    valid_epoch_accuracy = 100. * correct / (NUM_VALID*2)
    records['valid_epoch_loss'].append(valid_epoch_loss)
    records['valid_epoch_accuracy'].append(valid_epoch_accuracy.tolist())

    if valid_epoch_loss < records['valid_best_loss']:
        records['valid_best_loss'] = valid_epoch_loss
        save_ckp = True

    if valid_epoch_accuracy > records['valid_best_accuracy']:
        records['valid_best_accuracy'] = valid_epoch_accuracy.tolist()
        save_ckp = True


    is_init_root_train = not model.split and not model.extend and node_idx == 0

    if is_init_root_train \
            and valid_epoch_loss < records['valid_best_root_nosplit']:
        records['valid_best_root_nosplit'] = valid_epoch_loss
        save_ckp = True

    if save_ckp and not model.extend:
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records, '_tmp')
    elif save_ckp and model.extend:
        checkpoint_model('model_ext.pth', model=model)
        checkpoint_msc(struct, records, '_ext')

    end = time.time()
    records['time'] = end - start
    print(
        'Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            valid_epoch_loss, correct, NUM_VALID*2,
            100. * correct / (NUM_VALID*2), records['time'],
        ),
    )
    return valid_epoch_loss, save_ckp, valid_epoch_accuracy


def test(model, data_loader, debug=0):
    """ Test step """
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for _, data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_top1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            temp = pred.eq(target.view(1, -1).expand_as(pred))
            correct_top5 += temp[:5].contiguous().view(-1).float().sum(0)


    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct_top1 / len(data_loader.dataset)
    records['test_epoch_loss'].append(test_loss)
    records['test_epoch_accuracy'].append(test_accuracy.tolist())

    if test_loss < records['test_best_loss']:
        records['test_best_loss'] = test_loss

    if test_accuracy > records['test_best_accuracy']:
        records['test_best_accuracy'] = test_accuracy.tolist()

    end = time.time()
    print(
        'Test set: Average loss: {:.4f}, Top-1 Accuracy: {}/{} ({:.4f}%)'
        '\nTook {} seconds. '.format(
            test_loss, correct_top1, len(data_loader.dataset),
            100. * correct_top1 / len(data_loader.dataset), end - start,
        ),
    )
    print(
        'Test set: Average loss: {:.4f}, Top-5 Accuracy: {}/{} ({:.4f}%)'
        '\nTook {} seconds. '.format(
            test_loss, correct_top5, len(data_loader.dataset),
            100. * correct_top5 / len(data_loader.dataset), end - start,
        ),
    )
    return 100. * correct_top1 / len(data_loader.dataset), 100. * correct_top5 / len(data_loader.dataset)


def test_each_class(model, data_loader,nod_classes = False , debug=0):
    """ Test step """
    model.eval()
    test_loss = 0
    correct = 0
    N_CLASSES = 10
    class_correct = [0. for i in range(N_CLASSES)]
    class_total = [0. for i in range(N_CLASSES)]


    with torch.no_grad():
        for _, data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            

            c = (pred.squeeze() == target).squeeze()
            
            for i in range(len(target)):
                _label = target[i]
                class_correct[_label] += c[i].item()
                class_total[_label] += 1


    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)
    records['test_epoch_loss'].append(test_loss)
    records['test_epoch_accuracy'].append(test_accuracy.tolist())

    if test_loss < records['test_best_loss']:
        records['test_best_loss'] = test_loss

    if test_accuracy > records['test_best_accuracy']:
        records['test_best_accuracy'] = test_accuracy.tolist()

    end = time.time()
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), end - start,
        ),
    )
    if not nod_classes:
        for i in range(N_CLASSES):
            if class_total[i]:
                print('Accuracy of class %d : %.2f %%' % (i, 100 * class_correct[i] / class_total[i]))


    return 100. * correct / len(data_loader.dataset)

def test_for_conf(model, conf_data_loader,node_idxs):
    """ Test step """
    model.eval()
    conf = [0 for _ in range(len(node_idxs))]
    total_samples = 0
    
    with torch.no_grad():
        for _, data, target in conf_data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            batch_size = data.size(0)
            
            for j in range(len(node_idxs)):
                conf_tmp, _ = model.compute_routing_probabilities_uptonode(data, node_idxs[j])
                conf[j] += torch.sum(conf_tmp, dim=0)
            total_samples += batch_size
    
    # 计算平均值
    conf = [c/total_samples for c in conf]
    return conf

def dua(model, data_loader):
    """ Test step """
    model.eval()
    mom_pre = 0.1
    decay_factor = 0.94
    acc_best = 0
    mom_best = 0
    # model_best = model.copy()
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx == 100:
            break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        mom_new = (mom_pre * decay_factor)
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + 0.005
                # m.momentum = 0.01342
        mom_pre = mom_new
        _ = model(data)
        acc = test(model, data_loader)
        if acc >= acc_best:
            # model_best = model.copy()
            acc_best = acc
            mom_best = mom_pre + 0.005
            print('acc_best:'+str(acc_best)+' ----  momentum_best:'+str(mom_best))
    return model


# 测试时加载模型参数和结构
def _load_checkpoints(model_file_name, struct_file_name):
    save_dir = "./experiments/{}/{}/{}/{}".format(
        args.dataset, args.experiment, args.subexperiment, 'checkpoints',
    )
    model = torch.load(save_dir + '/' + model_file_name)
    if args.cuda:
        model.cuda()

    struct_file_fullname = save_dir + '/' + struct_file_name
    with open(struct_file_fullname, 'r') as file:
        struct = json.load(file)
    
    return model, struct


def checkpoint_model(model_file_name, struct=None, modules=None, model=None, figname='hist.png', data_loader=None):
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))
    
    # If model is not given, then build one. 
    if not(model) and modules and struct:
        model = Tree(struct, modules, cuda_on=args.cuda)
        
    # save the model:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset, args.experiment, args.subexperiment, 'checkpoints')
    model_path = save_dir + '/' + model_file_name
    # pdb.set_trace()
    torch.save(model, model_path)
    print("Model saved to {}".format(model_path))


def checkpoint_msc(struct, data_dict, name=""):
    """ Save structural information of the model and experimental results.

    Args:
        struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.
        data_dict (dict) : data about the experiment (e.g. loss, configurations)
    """
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))

    # save the tree structures as a json file:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset,args.experiment,args.subexperiment,'checkpoints')
    struct_path = save_dir + "/tree_structures"+name+".json"
    with open(struct_path, 'w') as f:
        json.dump(struct, f)
    print("Tree structure saved to {}".format(struct_path))

    # save the dictionary as jason file:
    if save_records:
        dict_path = save_dir + "/records"+name+".json"
        with open(dict_path, 'w') as f_d:
            json.dump(data_dict, f_d)
        print("Other data saved to {}".format(dict_path))


def get_decision(criteria, node_idx, tree_struct):
    """ Define the splitting criteria

    Args:
        criteria (str): Growth criteria.
        node_idx (int): Index of the current node.
        tree_struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.

    Returns:
        The function returns one of the following strings
            'split': split the node
            'extend': extend the node
            'keep': keep the node as it is
    """
    if criteria == 'always':  # always split or extend
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] > 0.0:
            return 'extend'
        else:
            return 'split'
    elif criteria == 'avg_valid_loss':
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] and \
                        tree_struct[node_idx]['valid_accuracy_gain_ext'] > dec_thre:
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_ext']))
            return 'extend'

        elif tree_struct[node_idx]['valid_accuracy_gain_split'] > dec_thre: 
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_split']))
            return 'split'

        else:
            print("Average valid loss is aggravated by split/extension."
                  " Keep the node as it is.")
            return 'keep'
    else:
        raise NotImplementedError(
            "specified growth criteria is not available. ",
        )


def optimize_fixed_tree(
        model, tree_struct, new_train_loader,
        new_valid_loader, new_test_loader, no_epochs, node_idx, class_aware=False, grow_class=False
):
    """ grow_class - grow CDTree
        class_aware - finetune CDTree """
    # get if the model is growing or fixed
    grow = (model.split or model.extend)
    load_model = False
    print(grow)
    print(model.split)
    print(model.extend)

    # define optimizer and trainable parameters
    if class_aware:
        params, names = get_params_node(grow, node_idx,  model, class_aware, grow_class)
        node_idx = node_idx[-1]
    else:
        params, names = get_params_node(grow, node_idx,  model, class_aware, grow_class)
        if not isinstance(node_idx, int):
            node_idx = node_idx[-1]
    for i, (n, p) in enumerate(model.named_parameters()):
        if not(n in names):
            # print('(Fix)   ' + n)
            p.requires_grad = False
        else:
            # print('(Optimize)     ' + n)
            p.requires_grad = True

    for i, p in enumerate(params):
        if not(p.requires_grad):
            print("(Grad not required)" + names[i])

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params), lr=args.lr,
    )
    # optimizer = optim.SGD(
    #     filter(lambda p: p.requires_grad, params), lr=args.lr,
    # )
    
    if args.scheduler:
        scheduler = get_scheduler(args.scheduler, optimizer, grow)

    # monitor nodewise best valid loss:
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes']) == node_idx:
        records['valid_best_loss_nodes'].append(np.inf)
    
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_split']) == node_idx:
        records['valid_best_loss_nodes_split'].append(np.inf)

    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_ext']) == node_idx:
        records['valid_best_loss_nodes_ext'].append(np.inf)

    # start training
    min_improvement = 0.0  # acceptable improvement in loss for early stopping
    valid_loss = np.inf
    patience_cnt = 1

    for epoch in range(1, no_epochs + 1):
        print("\n----- Layer {}, Node {}, Epoch {}/{}, Patience {}/{}---------".
              format(tree_struct[node_idx]['level'], node_idx, 
                     epoch, no_epochs, patience_cnt, args.epochs_patience))
        if class_aware: #finetune CDTree tree part with replay
            train_replay(model, train_loader_replay, new_train_loader, optimizer, node_idx)
            valid_loss_new, save_ckp, valid_epoch_accuracy = valid_class(model, new_valid_loader, node_idx, tree_struct)
        else:
            train(model, new_train_loader, optimizer, node_idx)
            valid_loss_new, save_ckp, valid_epoch_accuracy = valid_class(model, new_valid_loader, node_idx, tree_struct)

 
        
        # learning rate scheduling:
        if args.scheduler == 'plateau':
            scheduler.step(valid_loss_new)
        elif args.scheduler == 'step_lr':
            scheduler.step()
        
        test(model, new_test_loader)

        if not((valid_loss-valid_loss_new) > min_improvement):  #and grow
            patience_cnt += 1
        valid_loss = valid_loss_new*1.0
        
        if patience_cnt > args.epochs_patience > 0:
            print('Early stopping by patience!')
            break

        if save_ckp:
            load_model = True

 
    # load the node-wise best model based on validation accuracy:
    if no_epochs > 0 and grow:
        if model.extend:
            if load_model:
                print('return the node-wise best extended model')
                model, tree_struct = _load_checkpoints('model_ext.pth', 'tree_structures_ext.json')
        else:
            if load_model:
                print('return the node-wise best split model')
                model, tree_struct = _load_checkpoints('model_tmp.pth', 'tree_structures_tmp.json')
    if class_aware:  #finetune
        if load_model:
            print('return the class-wise best tmp model')
            model, tree_struct = _load_checkpoints('model_tmp.pth', 'tree_structures_tmp.json')

    # return the updated models:
    tree_modules = model.update_tree_modules()
    if model.split:
        child_left, child_right = model.update_children()
        return model, tree_modules, child_left, child_right
    elif model.extend:
        child_extension = model.update_children()
        return model, tree_modules, child_extension
    else:
        return model, tree_modules




def grow_ant_nodewise(tree_struct, tree_modules, layer_n):
    """The main function for optimising an ANT """
    

    # train classifier on root node (no split no extension):
    model = Tree(
        tree_struct, tree_modules, split=False, extend=False, cuda_on=args.cuda, init_train=True
    )
    if args.cuda:
        model.cuda()
    
    # optimise
    model, tree_modules = optimize_fixed_tree(
        model, tree_struct,
        train_loader, valid_loader, test_loader, args.epochs_node, node_idx=len(tree_struct)-1,
    )
    checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules)
    checkpoint_msc(tree_struct, records)



    if args.cuda:
        model.cuda()

    # ######################## 1: Growth phase starts ########################
    nextind = 1
    last_node = 0
    for lyr in range(layer_n, args.maxdepth):
        print("---------------------------------------------------------------")
        print("\nAt layer " + str(lyr))
        for node_idx in range(len(tree_struct)):
            change = False
            if tree_struct[node_idx]['level']==lyr and tree_struct[node_idx]['is_leaf'] and not(tree_struct[node_idx]['visited']):

                print("\nProcessing node " + str(node_idx))

                # -------------- Define children candidate nodes --------------
                # ---------------------- (1) Split ----------------------------
                # left child
                identity = True
                meta_l, node_l = define_node(
                    args,
                    node_index=nextind, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # right child
                meta_r, node_r = define_node(
                    args,
                    node_index=nextind+1, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # inheriting solver modules to facilitate optimization:
                if args.solver_inherit and meta_l['identity'] and meta_r['identity'] and not(node_idx == 0):
                    node_l['classifier'] = tree_modules[node_idx]['classifier']
                    node_r['classifier'] = tree_modules[node_idx]['classifier']

                # define a tree with a new split by adding two children nodes:
                model_split = Tree(tree_struct, tree_modules,
                                   split=True, node_split=node_idx,
                                   child_left=node_l, child_right=node_r,
                                   extend=False,
                                   cuda_on=args.cuda)

                # -------------------- (2) Extend ----------------------------
                # define a tree with node extension
                meta_e, node_e = define_node(
                    args,
                    node_index=nextind,
                    level=lyr+1,
                    parent_index=node_idx,
                    tree_struct=tree_struct,
                    identity=False,
                )
                # Set the router at the current node as one-sided One().
                # TODO: this is not ideal as it changes tree_modules
                tree_modules[node_idx]['router'] = One()

                # define a tree with an extended edge by adding a node
                model_ext = Tree(tree_struct, tree_modules,
                                 split=False,
                                 extend=True, node_extend=node_idx,
                                 child_extension=node_e,
                                 cuda_on=args.cuda)

                # ---------------------- Optimise -----------------------------
                best_tr_loss = records['train_best_loss']
                best_va_loss = records['valid_best_loss']
                best_te_loss = records['test_best_loss']
                best_va_acc = records['valid_best_accuracy']

                print("\n---------- Optimizing a binary split ------------")
                if args.cuda:
                    model_split.cuda()

                # split and optimise
                model_split, tree_modules_split, node_l, node_r \
                    = optimize_fixed_tree(model_split, tree_struct,
                                          train_loader, valid_loader, test_loader,
                                          args.epochs_node,
                                          node_idx)

                best_tr_loss_after_split = records['train_best_loss']
                # best_va_loss_adter_split = records['valid_best_loss_nodes_split'][node_idx]
                best_va_loss_after_split = records['valid_best_loss']
                best_te_loss_after_split = records['test_best_loss']
                best_va_acc_after_split = records['valid_best_accuracy']
                tree_struct[node_idx]['train_accuracy_gain_split'] \
                    = best_tr_loss - best_tr_loss_after_split
                tree_struct[node_idx]['valid_accuracy_gain_split'] \
                        = best_va_acc_after_split - best_va_acc
                tree_struct[node_idx]['test_accuracy_gain_split'] \
                    = best_te_loss - best_te_loss_after_split

                print("\n----------- Optimizing an extension --------------")
                if not(meta_e['identity']):
                    if args.cuda:
                        model_ext.cuda()

                    # make deeper and optimise
                    model_ext, tree_modules_ext, node_e \
                        = optimize_fixed_tree(model_ext, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_node,
                                              node_idx)

                    best_tr_loss_after_ext = records['train_best_loss']
                    best_va_loss_adter_ext = records['valid_best_loss']
                    best_te_loss_after_ext = records['test_best_loss']
                    best_va_acc_after_ext = records['valid_best_accuracy']

                    tree_struct[node_idx]['train_accuracy_gain_ext'] \
                        = best_tr_loss - best_tr_loss_after_ext
                    tree_struct[node_idx]['valid_accuracy_gain_ext'] \
                        = best_va_acc_after_ext - best_va_acc
                    tree_struct[node_idx]['test_accuracy_gain_ext'] \
                        = best_te_loss - best_te_loss_after_ext
                else:
                    print('No extension as '
                          'the transformer is an identity function.')
                
                # ---------- Decide whether to split, extend or keep -----------
                criteria = get_decision(args.criteria, node_idx, tree_struct)

                if criteria == 'split':
                    print("\nSplitting node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['right_child'] = nextind+1
                    tree_struct[node_idx]['split'] = True

                    # add the children nodes
                    tree_struct.append(meta_l)
                    tree_modules_split.append(node_l)
                    tree_struct.append(meta_r)
                    tree_modules_split.append(node_r)

                    # update tree_modules:
                    tree_modules = tree_modules_split
                    nextind += 2
                    change = True
                elif criteria == 'extend':
                    print("\nExtending node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['extended'] = True

                    # add the children nodes
                    tree_struct.append(meta_e)
                    tree_modules_ext.append(node_e)

                    # update tree_modules:
                    tree_modules = tree_modules_ext
                    nextind += 1
                    change = True
                else:
                    # revert weights back to state before split
                    print("No splitting at node " + str(node_idx))
                    print("Revert the weights to the pre-split state.")
                    # model = _load_checkpoint('model.pth')
                    model, tree_struct = _load_checkpoints('model.pth', 'tree_structures.json')
                    tree_modules = model.update_tree_modules()

                # record the visit to the node
                tree_struct[node_idx]['visited'] = True

                # save the model and tree structures:
                checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                                 data_loader=test_loader,
                                 figname='hist_split_node_tmp.png')#figname='hist_split_node_{:03d}.png'.format(node_idx)
                checkpoint_msc(tree_struct, records)
                last_node = node_idx

                # global refinement prior to the next growth
                # NOTE: this is an option not included in the paper.
                if args.finetune_during_growth and (criteria == 1 or criteria == 2):
                    print("\n-------------- Global refinement --------------")   
                    model = Tree(tree_struct, tree_modules,
                                 split=False, node_split=last_node,
                                 extend=False, node_extend=last_node,
                                 cuda_on=args.cuda)
                    if args.cuda: 
                        model.cuda()

                    model, tree_modules = optimize_fixed_tree(
                        model, tree_struct,
                        train_loader, valid_loader, test_loader,
                        args.epochs_finetune_node, node_idx,
                    )
            # break # TODO 要删掉
        # terminate the tree growth if no split or extend in the final layer
        if not change: break

    # ############### 2: Refinement (finetuning) phase starts #################
    print("\n\n------------------- Fine-tuning the tree --------------------")
    best_valid_accuracy_before = records['valid_best_accuracy']
    model = Tree(tree_struct, tree_modules,
                 split=False,
                 node_split=last_node,
                 child_left=None, child_right=None,
                 extend=False,
                 node_extend=last_node, child_extension=None,
                 cuda_on=args.cuda)
    test(model, test_loader)
    # model = _load_checkpoint('model.pth')
    # model, tree_struct = _load_checkpoints('model.pth', 'tree_structures.json')
    tree_modules = model.update_tree_modules()
    if args.cuda: 
        model.cuda()

    model, tree_modules = optimize_fixed_tree(model, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_finetune,
                                              last_node)

    best_valid_accuracy_after = records['valid_best_accuracy']

    # only save if fine-tuning improves validation accuracy
    if best_valid_accuracy_after - best_valid_accuracy_before > 0:
        # model = _load_checkpoint('model_tmp.pth')
        model, tree_struct = _load_checkpoints('model_tmp.pth', 'tree_structures_tmp.json')
        tree_modules = model.update_tree_modules()
        checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                         data_loader=test_loader,
                         figname='hist_split_node_finetune.png')
        checkpoint_msc(tree_struct, records)

    return tree_struct, tree_modules, lyr





def grow_ant_classwise(tree_struct, tree_modules, layer_n, grow_class=True):
    """The main function for optimising CDTree """
    """ grow_class - grow CDTree"""
    class_aware = True # finetune
    model = Tree(
        tree_struct, tree_modules, split=False, extend=False, cuda_on=args.cuda, init_train=True
    )
    if args.cuda:
        model.cuda()

    finetune_nodes=[]

    leaf_nodes = model.leaves_list
    print(' ------ leaf nodes and corresponding confidence:')
    print(leaf_nodes)
    conf = test_for_conf(model, train_loader_conf, leaf_nodes)[-1]
    print(conf)
    sorted_indices = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True) # false 反排
    sorted_leaf_nodes=[]
    sorted_conf = []
    for i in sorted_indices:
        sorted_leaf_nodes.append(leaf_nodes[i])
        sorted_conf.append(conf[i])
    print(sorted_leaf_nodes)


    ######################## 1: Growth phase starts ########################
    for n in range(len(sorted_leaf_nodes)):
        if grow_class and (sorted_conf[n] > tree_thre or n == 0):
            nextind = len(tree_struct)
            node_idx = sorted_leaf_nodes[n]
            for lyr in range(args.maxdepth):
                change = False
                grown = False
                if tree_struct[node_idx]['level']==lyr:
                    grown = True
                    print("---------------------------------------------------------------")
                    print("\nAt layer " + str(lyr))
                    print("\nProcessing node " + str(node_idx))

                    # -------------- Define children candidate nodes --------------
                    # ---------------------- (1) Split ----------------------------
                    # left child
                    identity = True
                    meta_l, node_l = define_node(
                        args,
                        node_index=nextind, level=lyr+1,
                        parent_index=node_idx, tree_struct=tree_struct,
                        identity=identity,
                    )
                    # right child
                    meta_r, node_r = define_node(
                        args,
                        node_index=nextind+1, level=lyr+1,
                        parent_index=node_idx, tree_struct=tree_struct,
                        identity=identity,
                    )
                    # inheriting solver modules to facilitate optimization:
                    if args.solver_inherit and meta_l['identity'] and meta_r['identity'] and not(node_idx == 0):
                        node_l['classifier'] = tree_modules[node_idx]['classifier']
                        node_r['classifier'] = tree_modules[node_idx]['classifier']

                    # define a tree with a new split by adding two children nodes:
                    model_split = Tree(tree_struct, tree_modules,
                                    split=True, node_split=node_idx,
                                    child_left=node_l, child_right=node_r,
                                    extend=False,
                                    cuda_on=args.cuda)

                    # -------------------- (2) Extend ----------------------------
                    # define a tree with node extension
                    meta_e, node_e = define_node(
                        args,
                        node_index=nextind,
                        level=lyr+1,
                        parent_index=node_idx,
                        tree_struct=tree_struct,
                        identity=False,
                    )
                    # Set the router at the current node as one-sided One().
                    # TODO: this is not ideal as it changes tree_modules
                    tree_modules[node_idx]['router'] = One()

                    # define a tree with an extended edge by adding a node
                    model_ext = Tree(tree_struct, tree_modules,
                                    split=False,
                                    extend=True, node_extend=node_idx,
                                    child_extension=node_e,
                                    cuda_on=args.cuda)

                    # ---------------------- Optimise -----------------------------
                    best_tr_loss = records['train_best_loss']
                    best_va_loss = records['valid_best_loss']
                    best_te_loss = records['test_best_loss']
                    best_va_acc = records['valid_best_accuracy']

                    print("\n---------- Optimizing a binary split ------------")
                    if args.cuda:
                        model_split.cuda()

                    # split and optimise
                    model_split, tree_modules_split, node_l, node_r \
                        = optimize_fixed_tree(model_split, tree_struct,
                                            train_loader, valid_loader, test_loader,
                                            args.epochs_node,
                                            node_idx, grow_class=grow_class)

                    best_tr_loss_after_split = records['train_best_loss']
                    best_va_loss_after_split = records['valid_best_loss']
                    best_te_loss_after_split = records['test_best_loss']
                    best_va_acc_after_split = records['valid_best_accuracy']
                    tree_struct[node_idx]['train_accuracy_gain_split'] \
                        = best_tr_loss - best_tr_loss_after_split
                    tree_struct[node_idx]['valid_accuracy_gain_split'] \
                        = best_va_acc_after_split - best_va_acc
                    # if (best_va_loss - best_va_loss_after_split)<=0 and (best_va_acc_after_split-best_va_acc)>0:
                    #     tree_struct[node_idx]['valid_accuracy_gain_split'] \
                    #         = best_va_acc_after_split - best_va_acc
                    # else:
                    #     tree_struct[node_idx]['valid_accuracy_gain_split'] \
                    #         = best_va_loss - best_va_loss_after_split
                    # tree_struct[node_idx]['valid_accuracy_gain_split'] \
                    #     = best_va_loss - best_va_loss_adter_split
                    tree_struct[node_idx]['test_accuracy_gain_split'] \
                        = best_te_loss - best_te_loss_after_split

                    print("\n----------- Optimizing an extension --------------")
                    if not(meta_e['identity']):
                        if args.cuda:
                            model_ext.cuda()

                        # make deeper and optimise
                        model_ext, tree_modules_ext, node_e \
                            = optimize_fixed_tree(model_ext, tree_struct,
                                                train_loader, valid_loader, test_loader,
                                                args.epochs_node,
                                                node_idx, grow_class=grow_class)

                        best_tr_loss_after_ext = records['train_best_loss']
                        best_va_loss_after_ext = records['valid_best_loss']
                        best_te_loss_after_ext = records['test_best_loss']
                        best_va_acc_after_ext = records['valid_best_accuracy']

                        tree_struct[node_idx]['train_accuracy_gain_ext'] \
                            = best_tr_loss - best_tr_loss_after_ext
                        tree_struct[node_idx]['valid_accuracy_gain_ext'] \
                                = best_va_acc_after_ext - best_va_acc
                        tree_struct[node_idx]['test_accuracy_gain_ext'] \
                            = best_te_loss - best_te_loss_after_ext
                    else:
                        print('No extension as '
                            'the transformer is an identity function.')
                    
                    # ---------- Decide whether to split, extend or keep -----------
                    criteria = get_decision(args.criteria, node_idx, tree_struct)

                    if criteria == 'split':
                        print("\nSplitting node " + str(node_idx))
                        # update the parent node
                        tree_struct[node_idx]['is_leaf'] = False
                        tree_struct[node_idx]['left_child'] = nextind
                        tree_struct[node_idx]['right_child'] = nextind+1
                        tree_struct[node_idx]['split'] = True

                        # add the children nodes
                        tree_struct.append(meta_l)
                        tree_modules_split.append(node_l)
                        tree_struct.append(meta_r)
                        tree_modules_split.append(node_r)

                        # update tree_modules:
                        tree_modules = tree_modules_split
                        nextind += 2
                        change = True
                        finetune_nodes.append(node_idx)
                    elif criteria == 'extend':
                        print("\nExtending node " + str(node_idx))
                        # update the parent node
                        tree_struct[node_idx]['is_leaf'] = False
                        tree_struct[node_idx]['left_child'] = nextind
                        tree_struct[node_idx]['extended'] = True

                        # add the children nodes
                        tree_struct.append(meta_e)
                        tree_modules_ext.append(node_e)

                        # update tree_modules:
                        tree_modules = tree_modules_ext
                        nextind += 1
                        change = True
                        finetune_nodes.append(node_idx)
                    else:
                        # revert weights back to state before split
                        print("No splitting at node " + str(node_idx))
                        print("Revert the weights to the pre-split state.")
                        # model_files = "./experiments/{}/{}/{}/{}/{}".format(
                        #     args.dataset, args.experiment, args.subexperiment, 'checkpoints', 'model_temp.pth'
                        # )
                        # if os.path.exists(model_files):
                        # model = _load_checkpoint('model_temp.pth')
                        model, tree_struct = _load_checkpoints('model_temp.pth', 'tree_structures_temp.json')
                        # test_each_class(model, test_loader)
                        # else:
                        #     model = _load_checkpoint('model.pth')
                        tree_modules = model.update_tree_modules()
                        break

                    # record the visit to the node
                    tree_struct[node_idx]['visited'] = True

                    # save the model and tree structures:
                    checkpoint_model('model_temp.pth', struct=tree_struct, modules=tree_modules,
                                    data_loader=test_loader,
                                    figname='hist_split_node_tmp.png')
                    model = Tree(tree_struct, tree_modules, cuda_on=args.cuda)
                    checkpoint_msc(tree_struct, records, '_temp')
                    # model_temp = _load_checkpoint('model_temp.pth')
                    model, tree_struct = _load_checkpoints('model_temp.pth', 'tree_structures_temp.json')
                    # pdb.set_trace()
                    tree_modules = model.update_tree_modules()
                    # test_each_class(model, test_loader)


                    if args.finetune_during_growth and (criteria == 1 or criteria == 2):
                        print("\n-------------- Global refinement --------------")   
                        model = Tree(tree_struct, tree_modules,
                                    split=False, node_split=last_node,
                                    extend=False, node_extend=last_node,
                                    cuda_on=args.cuda)
                        if args.cuda: 
                            model.cuda()

                        model, tree_modules = optimize_fixed_tree(
                            model, tree_struct,
                            train_loader, valid_loader, test_loader,
                            args.epochs_finetune_node, node_idx,
                        )
            # terminate the tree growth if no split or extend in the final layer
            if not change and grown: break


    last_node = len(tree_struct)-2
    finetune_nodes.append(last_node)
    # ############### 2: Refinement (finetuning) phase starts #################
    print("\n\n------------------- Fine-tuning the tree --------------------")
    records['valid_best_accuracy'] = 0

    model = Tree(tree_struct, tree_modules,
                 split=False,
                 node_split=last_node,
                 args = args,
                 child_left=None, child_right=None,
                 extend=False,
                 node_extend=last_node, child_extension=None,
                 cuda_on=args.cuda,
                 class_aware=class_aware)
    
    tree_modules = model.update_tree_modules()
    if args.cuda: 
        model.cuda()
    if args.testeachclass and args.dataset=='cifar10':
        test_each_class(model, test_loader)
    else:
        test(model, test_loader)
    if args.finetune or args.optimal:
        class_aware = False # no replay
    
    if args.optimal:
        model, tree_modules = optimize_fixed_tree(model, tree_struct,
                                              all_train_loader, all_test_loader, all_test_loader,
                                              args.epochs_finetune,
                                              finetune_nodes, class_aware, grow_class)
    else:
        model, tree_modules = optimize_fixed_tree(model, tree_struct,
                                              train_loader, all_test_loader, test_loader,
                                              args.epochs_finetune,
                                              finetune_nodes, class_aware, grow_class)
        
    checkpoint_model('model_cft.pth', struct=tree_struct, modules=tree_modules,
            data_loader=test_loader,
            figname='hist_split_node_finetune.png')
    checkpoint_msc(tree_struct, records, name="_cft")


    return tree_struct, tree_modules, layer_n




#  ----------- init records ------------
records = init_records()

#  ----------- data processing -----------
# domain_types = ["None", "RandomHorizontalFlip", "ColorJitter", "RandomRotation",  "RandomAffine"] # 1234
domain_types = ["None", "RandomHorizontalFlip", "ColorJitter", "RandomRotation",  "RandomAffine", "RandomGrayscale", "RandomVerticalFlip", "RandomHorizontalFlip",  "ColorJitter",  "None"] # 1234
if args.dataset == "cifar10":
    ingroup_n = 1 
    tasks_num = 5
    NUM_TRAIN,NUM_VALID,NUM_TEST = 30000,6000,6000
elif args.dataset == "cifar100":
    ingroup_n = 10
    tasks_num = 5
    NUM_TRAIN,NUM_VALID,NUM_TEST = 30000,6000,6000
elif args.dataset == "imagenet200":
    ingroup_n = 20
    tasks_num = 5
    NUM_TRAIN,NUM_VALID,NUM_TEST = 60000,6000,6000
else:
    print('unknown datasets')
data_manager = DataManager(args.dataset , True, args.seed, 10, 10)
args.classes = ()
save_records = True
tree_da_head = []
# load init/task0 datasets
train_set_init = data_manager.get_dataset(np.arange(0*ingroup_n, 6*ingroup_n), domain_type=domain_types[0], source="train", mode="train", domainTrans=True)
test_set_init = data_manager.get_dataset(np.arange(0*ingroup_n, 6*ingroup_n), domain_type=domain_types[0], source="test", mode="test", domainTrans=True)
train_loader_init = torch.utils.data.DataLoader(train_set_init, batch_size=args.batch_size, shuffle=True)
test_loader_init = torch.utils.data.DataLoader(test_set_init, batch_size=args.batch_size, shuffle=False)
all_train_sets = []
all_test_sets = []

# ablation study setting
if args.finetune or args.finetune_replay or args.optimal:
    only_tree = True
    only_da = True
elif args.onlyda:
    only_tree = False
    only_da = True
elif args.onlytree:
    only_tree = True
    only_da = False
else: # cdtree
    only_tree = False      # only grow the tree
    only_da = False       # only domain adaptation

# threshold setting
da_thre = 0.5
if args.traininit:
    tree_thre = 0
    dec_thre=0
else:
    tree_thre = 0.35
    dec_thre=0.2
    if args.dataset == "cifar10":
        start_tree_acc = 60 #over 60% no grow tree
    elif args.dataset == "cifar100":
        start_tree_acc = 30 #over 30% no grow tree
    else:
        start_tree_acc = 15 #over 15% no grow tree
acc_final = [[],[]]
tasks_num_id = 2 # 1 = 5 tasks，2=10 tasks



if args.testonly:
    start = time.time()
    model_file = 'model.pth'
    tree_struct_file = 'tree_structures.json'
    model, tree_structs = _load_checkpoints(model_file, tree_struct_file)
    tree_modules = model.update_tree_modules()
    tree_module_init = copy.deepcopy(tree_modules[:])
    tree_da_head.append(copy.deepcopy(tree_modules[0]))
    print('============== Performance on task 0 =================')
    model = Tree(tree_structs, tree_modules, split=False, extend=False, cuda_on=args.cuda)
    top1, top5 = test(model, test_loader_init)
    if args.testeachclass and args.dataset=='cifar10':
        acc_total = test_each_class(model, test_loader_init)
    acc_final[0].append(top1.tolist())
    acc_final[1].append(top5.cpu().tolist())
    print('\n----- Top1 acc:', acc_final[0])
    print('----- Top5 acc:', acc_final[1])

else:
    tree_structs = []  # stores graph information for each node
    tree_modules = []  # stores modules for each node
    tree_da_head = []
    tree_module_init = []

    # --------------------------- Start growing CDTree! ---------------------------
    start = time.time()
    layer_n = 0
    if args.dataset:
        for i in range(tasks_num * tasks_num_id):
            dn = i % tasks_num
            if i == 0:
                total_classes = 6 * ingroup_n
                known_classes = total_classes
            elif i >= tasks_num:
                total_classes = 10 * ingroup_n
                known_classes = total_classes
            else:
                total_classes = (dn+6)*ingroup_n
                known_classes = total_classes-ingroup_n
            records['valid_best_loss'] = np.inf
            records['valid_best_accuracy'] = 0.0
            records['test_best_loss'] = np.inf
            records['test_best_accuracy'] = 0.0

            # overlap classes in comming tasks (for domain adaptation)
            train_set_init_da = data_manager.get_dataset(np.arange(dn*ingroup_n, 6*ingroup_n), domain_type=domain_types[i], source="train", mode="train", domainTrans=True)
            test_set_init_da = data_manager.get_dataset(np.arange(dn*ingroup_n, 6*ingroup_n), domain_type=domain_types[i], source="test", mode="test", domainTrans=True)
            train_loader_init_da = torch.utils.data.DataLoader(train_set_init_da, batch_size=args.batch_size, shuffle=False)
            test_loader_init_da = torch.utils.data.DataLoader(test_set_init_da, batch_size=args.batch_size, shuffle=False)
            train_set_init_old = data_manager.get_dataset(np.arange(dn*ingroup_n, 6*ingroup_n), domain_type=domain_types[0], source="train", mode="train", domainTrans=True)
            test_set_init_old = data_manager.get_dataset(np.arange(dn*ingroup_n, 6*ingroup_n), domain_type=domain_types[0], source="test", mode="test", domainTrans=True)
            train_loader_init_old = torch.utils.data.DataLoader(train_set_init_old, batch_size=args.batch_size, shuffle=False)
            test_loader_init_old = torch.utils.data.DataLoader(test_set_init_old, batch_size=args.batch_size, shuffle=False)


            # old classes in comming tasks
            train_set_old = data_manager.get_dataset(np.arange(dn*ingroup_n, (dn+5)*ingroup_n), domain_type=domain_types[i], source="train", mode="train", domainTrans=True)
            test_set_old = data_manager.get_dataset(np.arange(dn*ingroup_n, (dn+5)*ingroup_n), domain_type=domain_types[i], source="test", mode="test", domainTrans=True)
            train_loader_old = torch.utils.data.DataLoader(train_set_old, batch_size=args.batch_size, shuffle=True)
            train_loader_da = torch.utils.data.DataLoader(train_set_old, batch_size=128, shuffle=True)
            test_loader_old = torch.utils.data.DataLoader(test_set_old, batch_size=args.batch_size, shuffle=False)
            # new classes in comming tasks
            train_set_conf = data_manager.get_dataset(np.arange((dn+5)*ingroup_n, (dn+6)*ingroup_n), domain_type=domain_types[i], source="train", mode="train", domainTrans=True)
            test_set_conf = data_manager.get_dataset(np.arange((dn+5)*ingroup_n, (dn+6)*ingroup_n), domain_type=domain_types[i], source="test", mode="test", domainTrans=True)
            train_loader_conf = torch.utils.data.DataLoader(train_set_conf, batch_size=args.batch_size, shuffle=True)
            test_loader_conf = torch.utils.data.DataLoader(test_set_conf, batch_size=args.batch_size, shuffle=False)
            # combine new and old classes  in comming tasks
            train_set = ConcatDataset([train_set_old, train_set_conf])
            test_set = ConcatDataset([test_set_old, test_set_conf])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
            valid_loader = test_loader

            # combine current task and old data
            train_set_remaining = data_manager.get_dataset(np.arange(0, known_classes), domain_type=domain_types[0], source="train", mode="train", domainTrans=True)
            # train_set_replay_old = ConcatDataset([train_set_remaining, train_set_old, train_set_conf])
            train_loader_replay = torch.utils.data.DataLoader(train_set_remaining, batch_size=args.batch_size, shuffle=True)


            
            if i > 0:


                # all dataset except current task
                all_train_set_except_current = ConcatDataset(all_train_sets)
                all_test_set_except_current = ConcatDataset(all_test_sets)
                all_train_loader_except_current = torch.utils.data.DataLoader(all_train_set_except_current, batch_size=args.batch_size, shuffle=True)
                all_test_loader_except_current = torch.utils.data.DataLoader(all_test_set_except_current, batch_size=args.batch_size, shuffle=False)

            all_train_sets.append(train_set)
            all_test_sets.append(test_set)
            if args.optimal:
                # all dataset
                all_train_set = ConcatDataset(all_train_sets)
                all_train_loader = torch.utils.data.DataLoader(all_train_set, batch_size=args.batch_size, shuffle=True)
            all_test_set = ConcatDataset(all_test_sets)
            all_test_loader = torch.utils.data.DataLoader(all_test_set, batch_size=args.batch_size, shuffle=False)

            

            args.input_nc, args.input_width, args.input_height, classes = data_manager.get_dataset_details(task=i)
            list_union = list(args.classes) + [x for x in classes if x not in args.classes]
            args.classes = tuple(list_union)
            args.no_classes = total_classes
            # ----------- learning on task0 ------------
            if i == 0: 
                # ==== train init model
                if args.traininit:
                    save_records = True
                    # define the root node
                    root_meta, root_module = define_node(
                        args, node_index=0, level=0, parent_index=-1, tree_struct=tree_structs,
                    )
                    tree_structs.append(root_meta)
                    tree_modules.append(root_module)
                    tree_structs, tree_modules, layer_n = grow_ant_nodewise(tree_structs, tree_modules, layer_n)
                    # grow tree threshold
                    dec_thre=0.2
                
                # ==== load model
                model_file = 'model.pth'
                tree_struct_file = 'tree_structures.json'
                model, tree_structs = _load_checkpoints(model_file, tree_struct_file)
                tree_modules = model.update_tree_modules()
                tree_module_init = copy.deepcopy(tree_modules[:])
                tree_da_head.append(copy.deepcopy(tree_modules[0]))
                
            save_records = False
            # load model (without layer_n) -> get layer_n
            for node_idx in range(len(tree_structs)):
                if tree_structs[node_idx]['level'] > layer_n :
                    layer_n = tree_structs[node_idx]['level']

            for node_idx in range(len(tree_structs)):
                if tree_structs[node_idx]['level'] == layer_n :
                    tree_structs[node_idx]['visited'] = False
            
        
            # ----------- continual learning on task i  ------------
            if i > 0:
                print('============== Continual learning on task '+str(i)+' =================')
                print('\n[before adaption and growth]')
                tree_modules = copy.deepcopy(tree_module_init[:])
                model = Tree(tree_structs, tree_modules, args = args, split=False, extend=False, cuda_on=args.cuda, class_aware=True)
                tree_modules = model.update_tree_modules()
                if args.testeachclass and args.dataset=='cifar10':
                    da_acc = test_each_class(model, test_loader)
                else:
                    da_acc, _ = test(model, test_loader)

                if not (only_da and only_tree):
                    MMD_loss = ops.MMD_loss()
                    
                    # 从train_loader_init_old收集数据
                    old_data_list = []
                    for _, data, _ in train_loader_init_old:
                        old_data_list.append(data)
                    old_data = torch.cat(old_data_list, dim=0)
                    old_data = old_data.view(old_data.size(0), -1)  # 将数据展平为2D张量
                    print("old_data shape:", old_data.shape)
                    
                    # 从train_loader_init_da收集数据
                    new_data_list = []
                    for _, data, _ in train_loader_init_da:
                        new_data_list.append(data)
                    new_data_1 = torch.cat(new_data_list, dim=0)
                    new_data_1 = new_data_1.view(new_data_1.size(0), -1)  # 将数据展平为2D张量
                    print("new_data_1 shape:", new_data_1.shape)
                    
                    loss = MMD_loss(old_data.cuda(), new_data_1.cuda())
                    print('MMD loss:' + str(loss))
                    # import utils.domain_analysis as domain_analysis
                    # 计算域差异分数
                    # domain_gap = domain_analysis.compute_domain_gap(old_data, new_data_1)
                    # print(f"域差异分数: {domain_gap:.4f}")
                    # 计算域偏移
                    # domain_shift = domain_analysis.compute_domain_shift(old_data.cpu().numpy(), new_data_1.cpu().numpy())
                    # print(f"Wasserstein距离: {domain_shift['wasserstein_distance']:.4f}")
                    # print(f"CDF距离: {domain_shift['cdf_distance']:.4f}")

                
                # ablation study - only da module
                if not only_tree and loss > da_thre:
                    save_da = False
                    # =========== start adaptation ============ 
                    print('\n ------------ start adaptation .............')
                    for j in range(args.epochs_da):
                        print(f"--------- Task {i}, DA Epoch {j+1}/{args.epochs_da} ---------")
                        tree_modules = train_mmd(model, tree_modules, train_loader_init_old, train_loader_init_da, node_idx,1)
                        model = Tree(tree_structs, tree_modules, args = args, split=False, extend=False, cuda_on=args.cuda, class_aware=True)
                        if args.testeachclass and args.dataset=='cifar10':
                            da_acc_new = test_each_class(model, test_loader)
                        elif args.dataset=='cifar10':
                            da_acc_new, _ = test(model, test_loader)
                        else:
                            # _ , da_acc_new = test(model, test_loader)
                            da_acc_new, _ = test(model, test_loader)
                        if da_acc_new > da_acc:
                            da_acc = da_acc_new
                            checkpoint_model('model_temp.pth', model=model)
                            checkpoint_msc(tree_structs, records, '_temp')
                            save_da = True

                    print('[after adaption and growth]')
                    if save_da:
                        model, tree_structs = _load_checkpoints('model_temp.pth', 'tree_structures_temp.json')
                    else:
                        tree_modules = copy.deepcopy(tree_module_init[:])
                        model = Tree(tree_structs, tree_modules, args = args, split=False, extend=False, cuda_on=args.cuda, class_aware=True)
                    tree_modules = model.update_tree_modules()
                    if args.testeachclass and args.dataset=='cifar10':
                        da_acc = test_each_class(model, test_loader)
                    else:
                        da_acc, _ = test(model, test_loader)

                # ablation study - only grow the tree
                if only_da or da_acc > start_tree_acc or i >= tasks_num:
                    grow_class = False
                else:
                    grow_class = True
                tree_structs, tree_modules, layer_n = grow_ant_classwise(tree_structs, tree_modules, layer_n, grow_class)
                print('after growth:')
                model_file = 'model_cft.pth'
                tree_struct_file = 'tree_structures_cft.json'
                model, tree_structs = _load_checkpoints(model_file, tree_struct_file)
                tree_modules = model.update_tree_modules()
                if args.testeachclass and args.dataset=='cifar10':
                    test_each_class(model, test_loader)
                else:
                    test(model, test_loader)

                tree_da_head.append(copy.deepcopy(tree_modules[0]))

                # test 
                print(' ========================== test all task models ==========================')
                acc_total_1 = 0
                acc_total_5 = 0
                for k in range(i+1):
                    dn_k = k % tasks_num
                    if not only_tree:
                        tree_modules[0] = copy.deepcopy(tree_da_head[k])
                        model = Tree(tree_structs, tree_modules, args = args, split=False, extend=False, cuda_on=args.cuda, class_aware=False)
                    test_set = data_manager.get_dataset(np.arange(dn_k*ingroup_n, (dn_k+6)*ingroup_n), domain_type=domain_types[k], source="test", mode="test", domainTrans=True)
                    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
                    top1, top5 = test(model, test_loader)
                    if args.testeachclass and args.dataset=='cifar10':
                        acc_total = test_each_class(model, test_loader)
                    acc_total_1 += top1
                    acc_total_5 += top5

                print('\nAverage Top-1 accuracy: '+str(acc_total_1/(i+1)))
                acc_final[0].append((acc_total_1/(i+1)).cpu().tolist())
                acc_final[1].append((acc_total_5/(i+1)).cpu().tolist())
                print('\n----- Top1 acc:', acc_final[0])
                print('----- Top5 acc:', acc_final[1])
                    
                if only_tree:
                    tree_module_init = tree_modules
                else:
                    tree_module_init = [tree_da_head[i]] + tree_modules[1:]
            
            else:  # test task0
                print('============== Performance on task 0 =================')
                model = Tree(tree_structs, tree_modules, split=False, extend=False, cuda_on=args.cuda)
                top1, top5 = test(model, test_loader)
                if args.testeachclass and args.dataset=='cifar10':
                    acc_total = test_each_class(model, test_loader)
                acc_final[0].append(top1.tolist())
                acc_final[1].append(top5.cpu().tolist())
                print('\n----- Top1 acc:', acc_final[0])
                print('----- Top5 acc:', acc_final[1])