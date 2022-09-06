'''
Training script
Compared to train.py this training script update all the attributes
simultaneously, instead of one at a time
'''
import argparse
import datetime
import os
import sys
import json
import numpy as np
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from resnet import resnet50 as backbone
from models import ProxyNet, CondNet, ProxyLoss, MultiProxyNet
from utils import pairwise_distances, perform_faiss, AverageMeter, to_numpy, get_recall
sys.path.append('..')


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DML')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=456, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--learned_proxies', action='store_true', default=False,
                    help='Learn proxies during training (default: False)')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Finetuning resnet')
parser.add_argument('--lambdaI', type=float, default=1.,
                    help='Contribution of ITEM loss')
parser.add_argument('--lambdaC', type=float, default=1.,
                    help='Contribution of CAT loss')
parser.add_argument('--lambdaA', type=float, default=1.,
                    help='Contribution of ATTR loss')
parser.add_argument('--lambdaE', type=float, default=0.001,
                    help='Regularization on embedding')
parser.add_argument('--factor_lower', type=float, default=.1,
                    help='Multiplicative factor of the LR for lower layers')
parser.add_argument('--factor_upper', type=float, default=1.,
                    help='Multiplicative factor of the LR for upper layers')
parser.add_argument('--factor_proxies', type=float, default=10.,
                    help='Multiplicative factor of the LR for proxies')
parser.add_argument('--data_dir', type=str, metavar='DD',
                    default='/home/wthong1/data/cars_dataset/CompCars/data/cropped256',
                    help='data folder path')
parser.add_argument('--exp_dir', type=str, default='../exp', metavar='ED',
                    help='folder for saving exp')
parser.add_argument('--dataset', type=str, required=True,
                    help='DF|DIV')
parser.add_argument('--mode', type=str, required=True,
                    help='partial|full')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='enables multi gpu training')
parser.add_argument('--m', type=str, default='SUB', metavar='M',
                    help='message')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# create experiment folder
date = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
dname = date + '_' + args.dataset + '_' + args.mode + '_' + args.m
BASE_PATH = os.path.join(args.exp_dir, dname)
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

# data normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


if args.dataset == 'DIV':
    from data import create_div_splits, create_cc_csm, CCDataset

    # get data splits
    df_train, df_valid, df_test = create_div_splits(fname='../data/diverse/all.hdf5')

    ATTRIBUTES = ['door_number', 'seat_number', 'type']
    attr_for_loader = list(ATTRIBUTES)

    CAT_ARG = 'maker_name'
    INS_ARG = 'model_name'

    att_counts = {'door_number': 4, 'seat_number': 4, 'type': 9}
    create_csm = create_cc_csm
    MyDataset = CCDataset

elif args.dataset == 'DF':
    from data import create_df_splits, create_df_csm, DFDataset

    # get data splits
    df_train, df_valid, df_query, df_test = create_df_splits(
        fname='../data/deepfashion/all.hdf5')

    # basic info
    ATTRIBUTES = ['fabric', 'front', 'hem', 'neck', 'print',
                  'shoulder', 'silhouette', 'sleeve']
    attr_for_loader = list(ATTRIBUTES)
    CAT_ARG = 'clothes_category'
    INS_ARG = 'id'

    att_counts = {'fabric': 6, 'front': 7, 'hem': 6, 'neck': 13,
                  'print': 15, 'shoulder': 4, 'silhouette': 2, 'sleeve': 6}


    create_csm = create_df_csm
    MyDataset = DFDataset


args.n_attributes = len(ATTRIBUTES)
args.n_classes = len(df_train[INS_ARG].unique())
args.n_metaclasses = len(df_train[CAT_ARG].unique())

if args.mode in ['partial']:
    ATTRIBUTES = ATTRIBUTES + [CAT_ARG]
    att_counts[CAT_ARG] = args.n_metaclasses

n_concepts = len(ATTRIBUTES)
args.step = int(args.dim_embed / n_concepts)
args.sample_mode = 'full'

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

# saving logs
with open(os.path.join(BASE_PATH, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)

with open(os.path.join(BASE_PATH, 'logs.txt'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n\nExperiment with train %s\n' % args.dataset)


def write_logs(txt, logpath=os.path.join(BASE_PATH, 'logs.txt')):
    with open(logpath, 'a') as f:
        f.write('\n')
        f.write(txt)


def main(args, path):
    print '\n'
    print('# of classes: %d, # of images: %d, # of dims: %d' %
          (args.n_classes, len(df_train), args.dim_embed))

    # data loaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MyDataset(df_train,
                  transform=train_transforms,
                  data_dir=args.data_dir,
                  attributes=attr_for_loader,
                  mode=args.sample_mode),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        MyDataset(df_valid,
                  transform=test_transforms,
                  data_dir=args.data_dir,
                  attributes=attr_for_loader,
                  mode=args.sample_mode),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MyDataset(df_test,
                  transform=test_transforms,
                  data_dir=args.data_dir,
                  attributes=attr_for_loader,
                  mode=args.sample_mode),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # instanciate the models
    model = backbone(pretrained=True, embedding_size=args.dim_embed)

    if args.mode == 'full':
        proxy_noun_model = ProxyNet(args.n_classes, args.dim_embed,
                                    learned_proxies=args.learned_proxies)
        # get class-superclass-matrix
        csm_train = np.asarray(create_csm(df_train), dtype=np.int64)
        csm_train = torch.from_numpy(csm_train)
        if args.cuda:
            csm_train = csm_train.cuda()
    elif args.mode == 'partial':
        proxy_noun_model = ProxyNet(args.n_classes, args.dim_embed - args.step,
                                    learned_proxies=args.learned_proxies)
        csm_train = None
    else:
        proxy_noun_model = None
        csm_train = None

    proxy_adjs = [ProxyNet(att_counts[a], args.dim_embed,
                           learned_proxies=args.learned_proxies)
                  for a in ATTRIBUTES]

    if args.cuda:
        [pa.cuda() for pa in proxy_adjs]
    proxy_adj_models = MultiProxyNet(proxy_adjs)

    cond_model = CondNet(n_concepts, args.dim_embed, n_concepts)

    criterion_model = ProxyLoss(
        model, cond_model,
        proxy_adj_models, proxy_noun_model,
        csm_train,
        multigpu=args.multi_gpu,
        mode=args.mode)

    if args.cuda:
        model.cuda()
        if args.mode in ['full', 'partial']:
            proxy_noun_model.cuda()
        cond_model.cuda()
        criterion_model.cuda()

    parameters_set = []

    # fine-tuning resnet
    if args.finetune:
        low_layers = []
        upper_layers = []

        for c in model.children():
            if isinstance(c, nn.Linear):
                upper_layers.extend(list(c.parameters()))
            else:
                low_layers.extend(list(c.parameters()))

        parameters_set.append({'params': low_layers,
                               'lr': args.lr * args.factor_lower})
        parameters_set.append({'params': upper_layers,
                               'lr': args.lr * args.factor_upper})
    else:
        # first freeze parameters in resnet
        for p in criterion_model.parameters():
            p.requires_grad = False
        for p in model.fc_embed.parameters():
            p.requires_grad = True

        parameters_nn = filter(lambda p: p.requires_grad,
                               model.parameters())
        parameters_set.append({'params': parameters_nn})

    # learn proxies
    if args.learned_proxies:
        if args.mode in ['full', 'partial']:
            parameters_pn = filter(lambda p: p.requires_grad,
                                   proxy_noun_model.parameters())
            parameters_set.append({'params': parameters_pn,
                                   'lr': args.lr  * args.factor_proxies})

        parameters_pa = filter(lambda p: p.requires_grad,
                               proxy_adj_models.parameters())
        parameters_set.append({'params': parameters_pa,
                               'lr': args.lr  * args.factor_proxies})

    optimizer = optim.Adam(parameters_set, lr=args.lr, weight_decay=5e-5)

    n_parameters = sum([p.data.nelement()
                        for p in criterion_model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    # scheduler = ExponentialLR(optimizer, 0.94)
    scheduler = CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), eta_min=3e-6)

    best_acc = 0

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        # scheduler.step()

        # train for one epoch
        train(train_loader, criterion_model, optimizer, epoch, scheduler)

        # evaluate on validation set
        acc = validate(valid_loader, criterion_model, catacc=True)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if epoch % 5 == 0:
            tmp = 'checkpoint_%04d.pth.tar' % epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': criterion_model.state_dict(),
                'best_prec1': best_acc,
            }, is_best, filename=tmp)
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': criterion_model.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

    print('\nResults on test set (end of training)')
    write_logs('\nResults on test set (end of training)',
               logpath=os.path.join(path, 'logs.txt'))
    test_acc = validate(test_loader, criterion_model)

    print('\nResults on test set (best model)')
    write_logs('\nResults on test set (best model)',
               logpath=os.path.join(path, 'logs.txt'))
    test_best_model(test_loader, criterion_model, folder=path)


def train(train_loader, criterion_model, optimizer, epoch, scheduler):
    """Training loop for one epoch
    Arguments:
        train_loader (Dataset loader): data generator
        criterion_model (nn.Module): neural network that outputs
            the mse losses
        optimizers (optim): list of optimizers: (1) nn weights,
        (2) centers
        epoch (int): epoch no
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()

    val_acc = {key: AverageMeter()
               for key in ['items', 'cats', 'attrs']}
    val_loss = {key: AverageMeter()
                for key in ['overall', 'items', 'cats', 'attrs']}
    val_reg = {key: AverageMeter()
               for key in ['emb']}

    # switch to train mode
    criterion_model.train()

    end = time.time()

    for i, (imgs, labels, masks) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            imgs = imgs.cuda()
            labels = [l.cuda() for l in labels]
            masks = [m.cuda() for m in masks]

        imgs = Variable(imgs)
        masks = [Variable(m) for m in masks]

        # compute losses
        ll, aa, rr = criterion_model(imgs, labels, masks)
        loss_ip, loss_cp, loss_m = ll
        loss = args.lambdaI * loss_ip + args.lambdaC * loss_cp + args.lambdaA * loss_m + args.lambdaE * rr[0]

        val_loss['overall'].update(to_numpy(loss)[0])
        for k, key in enumerate(['items', 'cats', 'attrs']):
            val_acc[key].update(aa[k], imgs.size(0))
        for k, key in enumerate(['items', 'cats', 'attrs']):
            val_loss[key].update(to_numpy(ll[k])[0], imgs.size(0))
        for k, key in enumerate(['emb']):
            val_reg[key].update(to_numpy(rr[k])[0], imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Printing stuff
    txt = ('Epoch [%d]:\t Time %.3f\t Data %.3f\t' % \
           (epoch, batch_time.avg * i, data_time.avg * i))

    tmp = '\t'.join('%s: %.4f' % (key, val_loss[key].avg)
                    for key in ['overall', 'items', 'cats', 'attrs'])
    txt += '\n.. LOSS\t' + tmp

    tmp = '\t'.join('%s: %.4f' % (key, val_reg[key].avg)
                    for key in ['emb'])
    txt += '\n.. REG\t' + tmp

    tmp = '\t'.join('%s: %.4f' % (key, val_acc[key].avg * 100)
                    for key in ['items', 'cats', 'attrs'])
    txt += '\n.. PREC\t' + tmp

    print(txt)
    write_logs(txt)


def validate(val_loader, model, catacc=False, v=True):
    nouns = []
    items = []
    adjs = []
    mymasks = []
    outputs = []

    # switch to evaluate mode
    model.eval()

    for i, (imgs, labels, masks) in enumerate(val_loader):
        if args.cuda:
            imgs = imgs.cuda()
        # volatile for inference
        imgs = Variable(imgs, volatile=True)

        # compute output
        output = model.model(imgs)
        outputs.extend(output.cpu().data.numpy())

        item = labels[0]
        noun = labels[1]
        adj = labels[2:]

        m_cat = masks[0]
        m_attr = masks[1:]

        items.extend(item.numpy())
        nouns.extend(noun.numpy())
        adjs.extend(np.vstack([a.numpy() for a in adj]).T)
        mymasks.extend(np.vstack([m.numpy() for m in m_attr]).T)

    if catacc:
        if args.mode == 'full':
            proxies_cat = model.create_supercenters()
            tmp_proxies_cat = proxies_cat.cpu().data.numpy()
            prec1_c, per_class_prec1_c, loss_c = per_class_accuracy(
                np.asarray(outputs),
                np.asarray(nouns),
                tmp_proxies_cat)
        elif args.mode == 'partial':
            start = -args.step
            tmp_proxies_cat = model.proxy_adj_nets.embs[-1].proxies.weight
            tmp_proxies_cat = tmp_proxies_cat.data.cpu().numpy()[:, start:]
            prec1_c, per_class_prec1_c, loss_c = per_class_accuracy(
                np.asarray(outputs)[:, start:],
                np.asarray(nouns),
                tmp_proxies_cat)

        txt = ('.. Test (cat): Per-class Prec@1 %.4f\t Prec@1 %.4f\t' %
               (per_class_prec1_c * 100, prec1_c * 100))
        print(txt)
        if v:
            write_logs(txt)

    # measure adj accuracy and record loss
    adjs = np.vstack(adjs)
    masks = np.vstack(mymasks)
    prec1_mean = []
    for i, a in enumerate(ATTRIBUTES):
        if a not in [CAT_ARG, INS_ARG]:
            proxies_adj = model.proxy_adj_nets.embs[i].proxies.weight
            proxies_adj = proxies_adj.cpu().data.numpy()

            start = i * args.step
            prec1_a, per_class_prec1_a, loss_a = per_class_accuracy(
                np.asarray(outputs)[:, start:start + args.step],
                adjs[:, i],
                proxies_adj[:, start:start + args.step],
                masks[:, i])
            prec1_mean.append(prec1_a)
            txt = ('.. Test (%s): Loss %.4f\t Per-class Prec@1 %.4f\t Prec@1 %.4f\t' %
                   (a, loss_a, per_class_prec1_a * 100, prec1_a * 100))
            print(txt)
            if v:
                write_logs(txt)

    overall_acc = np.mean(prec1_mean)

    # perform recall
    if args.mode in 'full':
        gallery = np.asarray(outputs)
        idx = perform_faiss(gallery, gallery,
                            K=5, d=args.dim_embed, gpu=True)[:, 1:]
    elif args.mode == 'partial':
        end = - args.step
        gallery = np.ascontiguousarray(np.asarray(outputs)[:, :end])
        d = gallery.shape[1]
        idx = perform_faiss(gallery, gallery,
                            K=5, d=d, gpu=True)[:, 1:]

    r1 = get_recall(idx, np.asarray(items), K=1)
    txt = ('.. Recall@%d: %.02f' % (1, r1))
    print(txt)
    if v:
        write_logs(txt)

    # adjective accuracy
    txt = ('.... Overall test acc: %.4f' % overall_acc)
    print(txt)
    if v:
        write_logs(txt)
    return overall_acc


def test_best_model(test_loader, model,
                    folder=BASE_PATH, filename='model_best.pth.tar'):
    txt = "=> loading checkpoint '{}'".format(os.path.join(folder, filename))
    print(txt)
    write_logs(txt)
    checkpoint = torch.load(os.path.join(folder, filename))
    model.load_state_dict(checkpoint['state_dict'])
    txt = "=> loaded checkpoint '{}' (epoch {})".format(
        os.path.join(folder, filename), checkpoint['epoch'])
    print(txt)
    write_logs(txt)
    _ = validate(test_loader, model)


def count_unique(y):
    y_unique, inv, counts = np.unique(
        y, return_inverse=True, return_counts=True)
    y_counts = counts[inv]
    return y_unique, y_counts


def per_class_accuracy(output, target, proxies, mask=None):
    """Computes the precision@1 per class and the loss on CPU"""

    # copy on cpu
    my_output = output.copy()
    my_proxies = proxies.copy()
    my_target = target.copy()

    n_samples = my_target.size
    unique_labels, proxies_count = count_unique(my_target)

    # distances
    distances = cdist(my_output, my_proxies, 'sqeuclidean')

    # loss
    dist = distances[np.arange(n_samples), my_target]
    loss_val = 0.5 * np.sum(dist / proxies_count)

    # accuracy
    y_pred = np.argmin(distances, axis=1)

    if mask is None:
        n = np.count_nonzero(my_target == y_pred)
        acc = n / float(n_samples)
    elif mask is not None:
        cond = mask.astype(bool)
        n = np.count_nonzero(my_target[cond] == y_pred[cond])
        n_samples = np.count_nonzero(cond)
        acc = n / float(n_samples)
    prec = np.mean(acc)

    # per-class accuracy
    acc = []
    for label in unique_labels:
        if mask is None:
            n = np.count_nonzero(my_target == label)
            correct_preds = np.count_nonzero(
                np.logical_and(my_target == label, y_pred == label))
        elif mask is not None:
            n = np.count_nonzero(my_target[cond] == label)
            correct_preds = np.count_nonzero(
                np.logical_and(my_target[cond] == label, y_pred[cond] == label))
        acc.append(correct_preds / float(n))
    per_class_prec = np.mean(acc)

    return prec, per_class_prec, loss_val


def adjust_learning_rate(optimizer, epoch, lr=args.lr, milestone=15):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    local_lr = lr * (0.1 ** (epoch // milestone))
    for param_group in optimizer.param_groups:
        param_group['lr'] = local_lr


def save_checkpoint(state, is_best, folder=BASE_PATH, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


if __name__ == '__main__':
    main(args, BASE_PATH)
