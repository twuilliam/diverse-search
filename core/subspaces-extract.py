import argparse
import datetime
import os
import sys
import json
import numpy as np
import shutil
import time
import torch
from torch.autograd import Variable
from torchvision import transforms
from resnet import resnet50 as backbone
from models import ProxyNet, CondNet, ProxyLoss, MultiProxyNet


# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model-path', type=str, default='exp', metavar='ED',
                    help='model path')
parser.add_argument('--new-data-path', type=str, default='', metavar='ED',
                    help='overwrite data path')

args = parser.parse_args()
exp_path = os.path.dirname(args.model_path)

with open(os.path.join(exp_path, 'config.json')) as f:
    tmp = json.load(f)

tmp['model_path'] = args.model_path
tmp['new_data_path'] = args.new_data_path
args = type('parser', (object,), tmp)

if not args.new_data_path == '':
    args.data_dir = args.new_data_path

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


if args.mode == 'concat':
    ATTRIBUTES = ATTRIBUTES + [CAT_ARG] + [INS_ARG]
    att_counts[INS_ARG] = args.n_classes
    att_counts[CAT_ARG] = args.n_metaclasses
elif args.mode in ['partial']:
    ATTRIBUTES = ATTRIBUTES + [CAT_ARG]
    att_counts[CAT_ARG] = args.n_metaclasses

n_concepts = len(ATTRIBUTES)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])


def main(args):
    # data loaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MyDataset(df_train,
                  transform=test_transforms,
                  data_dir=args.data_dir,
                  attributes=attr_for_loader,
                  mode='full'),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MyDataset(df_test,
                  transform=test_transforms,
                  data_dir=args.data_dir,
                  attributes=attr_for_loader,
                  mode='full'),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    if args.dataset == 'DF':
        query_loader = torch.utils.data.DataLoader(
            MyDataset(df_query,
                      transform=test_transforms,
                      data_dir=args.data_dir,
                      attributes=attr_for_loader,
                      mode='full'),
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

    print('Loading checkpoint %s' % args.model_path)
    checkpoint = torch.load(args.model_path)
    criterion_model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        model.cuda()
        if args.mode in ['full', 'partial']:
            proxy_noun_model.cuda()
        cond_model.cuda()
        criterion_model.cuda()

    print('Extracting training set...')
    extract(train_loader, criterion_model, fname='features_train.npz')
    print('Extracting testing set...')
    extract(test_loader, criterion_model, fname='features_test.npz')
    if args.dataset == 'DF':
        print('Extracting query set...')
        extract(query_loader, criterion_model, fname='features_query.npz')
    print('Done!')


def extract(val_loader, model, fname):
    # switch to evaluate mode
    model.eval()
    np.random.seed(args.seed)

    all_feats = []
    all_labels = []
    all_masks = []
    for i, (img, labels, masks) in enumerate(val_loader):
        all_labels.append(torch.stack(labels).numpy().transpose())
        all_masks.append(torch.stack(masks).numpy().transpose())
        if args.cuda:
            img = img.cuda(async=True)
            labels = [l.cuda() for l in labels]
            masks = [m.cuda() for m in masks]

        img = Variable(img, volatile=True)
        labels = [Variable(l, volatile=True) for l in labels]
        masks = [Variable(m, volatile=True) for m in masks]

        # predict
        feat = model.model(img)
        all_feats.append(feat.data.cpu().numpy())

    data = {'feat': np.concatenate(all_feats),
            'label': np.concatenate(all_labels),
            'mask': np.concatenate(all_masks),
            'order': ['ins', 'cat'] + attr_for_loader}
    np.savez(os.path.join(exp_path, fname), **data)


if __name__ == '__main__':
    main(args)
