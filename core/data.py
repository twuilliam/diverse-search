import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
sys.path.append('..')


CC_ATT = ['door_number', 'seat_number', 'type']
DF_ATT = ['fabric', 'front', 'hem', 'neck', 'print',
          'shoulder', 'silhouette', 'sleeve']


def get_car_brand(df, car_model, previous=[]):
    '''Given a car model, get its car brand and how many models this brand has'''
    car_brand = df.loc[df['model_name'] == car_model, 'maker_name'].unique()[0]
    car_models_in_brand = df.loc[df['maker_name'] == car_brand, 'model_name'].unique()
    if len(previous) > 1:
        car_models_in_brand = [cm for cm in car_models_in_brand if cm not in previous]
    n_models = len(car_models_in_brand)
    return car_brand, n_models


def sample_by_car_type(df, valid=30, seed=2345):
    '''Sample car models to be in the val set by their type and brand'''
    np.random.seed(seed)

    vv, cc = np.unique(df.groupby('model_name').first()['type'], return_counts=True)
    n_items = float(cc.sum())
    n_to_take = np.ceil(cc / n_items * valid)

    print('Val set: (initial) %d car models, (final) %d car models' %
          (valid, n_to_take.sum()))

    to_take = []
    for i, n in enumerate(n_to_take):
        # get car models corresponding to the intended type
        items = df.loc[df['type'] == i + 1, 'model_name'].unique()
        # for selected car models, check if the corresponding car brand
        # has multiple models
        items = [i for i in items if get_car_brand(df, i, to_take)[1] > 1]
        choice = np.random.choice(items, size=int(n), replace=False)
        to_take.extend(choice)
    return to_take


def create_div_splits(fname='data/diverse/all.hdf5', attributes=CC_ATT):
    '''Create Train, Val and Test splits'''
    df = pd.read_hdf(fname)

    # transform into int columns
    keys = ['x1', 'y1', 'x2', 'y2', 'door_number', 'seat_number']
    df[keys] = df[keys].astype(int)

    # transform into categorical columns
    for key in attributes:
        df[key] = df[key].astype('category')

    cond = df['status'] == 'train'
    df_train = df.loc[cond]
    df_train = df_train.assign(model_name=df_train['model_name'].astype('category'))
    df_train = df_train.assign(maker_name=df_train['maker_name'].astype('category'))

    cond = df['status'] == 'val'
    df_valid = df.loc[cond]
    df_valid = df_valid.assign(model_name=df_valid['model_name'].astype('category'))
    df_valid = df_valid.assign(maker_name=df_valid['maker_name'].astype('category'))

    cond = df['status'] == 'test'
    df_test = df.loc[cond]
    df_test = df_test.assign(model_name=df_test['model_name'].astype('category'))
    df_test = df_test.assign(maker_name=df_test['maker_name'].astype('category'))

    return df_train, df_valid, df_test


def create_cc_csm(df):
    '''Class-SuperClass matrix
    Rows: car models
    Columns: car brands
    '''
    maker_names = df['maker_name'].cat.categories.unique()
    model_names = df['model_name'].cat.categories.unique()

    csm = np.zeros((len(model_names), len(maker_names)), dtype=bool)
    for i, car_model in enumerate(model_names):
        cond = df['model_name'] == car_model
        car_brand = df.loc[cond, 'maker_name'].unique()[0]
        car_brand_code = df.loc[cond, 'maker_name'].cat.codes.unique()[0]

        csm[i, car_brand_code] = True
    return csm


def sample_one_item_per_value(df, seed=2345, attributes=DF_ATT):
    '''Sample one clothes id per attribute value'''
    np.random.seed(seed)

    df_unique_id = df.drop_duplicates(subset='id')
    to_select = []

    for key in DF_ATT:
        for v in df_unique_id[key].cat.categories.unique().values[1:]:
            vec = df_unique_id.loc[df_unique_id[key] == v, 'id'].values
            tmp = np.random.choice(vec, size=1)
            to_select.extend(tmp)
    return to_select


def create_df_splits(fname='data/deepfashion/all.hdf5',
                     attributes=DF_ATT):
    '''Create Train, Val and Test splits'''
    df = pd.read_hdf(fname)

    # transform into categorical columns
    keys = ['status', 'clothes_category', 'clothes_type']
    keys = keys + attributes
    for key in keys:
        df[key] = df[key].astype('category')

    # transform into int columns
    keys = ['x1', 'y1', 'x2', 'y2']
    df[keys] = df[keys].astype(int)
    df['idx'] = np.arange(len(df))

    # Quick note
    # train split, query split, gallery (test) split
    df_tmp = df.loc[df['status'] == 'train']
    val_ids = sample_one_item_per_value(df_tmp)

    df_train = df_tmp.loc[~df_tmp['id'].isin(val_ids)]

    df_train = df_train.assign(id=df_train['id'].astype('category'))
    df_valid = df_tmp.loc[df_tmp['id'].isin(val_ids)]
    df_valid = df_valid.assign(id=df_valid['id'].astype('category'))
    df_query = df.loc[df['status'] == 'query']
    df_query = df_query.assign(id=df_query['id'].astype('category'))
    df_gallery = df.loc[df['status'] == 'gallery']
    df_gallery = df_gallery.assign(id=df_gallery['id'].astype('category'))

    return df_train, df_valid, df_query, df_gallery


def create_df_csm(df, key='clothes_category'):
    '''Class-SuperClass matrix
    Columns: clothes category
    Rows: clothes instance id
    '''
    tmp = df.drop_duplicates(subset='id')
    csm = np.asarray(pd.get_dummies(tmp[key]).values, dtype=np.int64)
    return csm


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, split, transform, data_dir,
                 attributes=CC_ATT,
                 loader=default_image_loader,
                 mode='test',
                 ins_arg='model_name',
                 cat_arg='maker_name'):
        self.split = split
        self.split_ins = self.split.drop_duplicates(subset=ins_arg)
        self.loader = loader
        self.transform = transform
        self.data_dir = data_dir
        self.mode = mode
        self.attributes = attributes
        self.to_get = [cat_arg] + self.attributes
        self.ins_arg = ins_arg
        self.cat_arg = cat_arg

    def remove_empty(self, df, offset=0):
        '''remove items with no attribute labels'''
        to_remove = df[self.attributes].values == 'None'
        cond = np.sum(to_remove, axis=1) < (len(self.attributes) - offset)
        return df[cond]

    def get_ins(self, fname):
        '''Get instance name from a file name'''
        return self.split.loc[self.split.index == fname, self.ins_arg][0]

    def get_coded_labels(self, model_name, offset=0):
        '''Get attribute code (defined by pandas DF)'''

        tmp = self.split_ins.loc[self.split_ins[self.ins_arg] == model_name]

        # get labels, offset is for DF dataset
        labels = []
        for a in self.to_get:
            # category column
            if a in self.cat_arg:
                labels.append(tmp[a].cat.codes[0].astype('int64'))
            # attribute columns
            else:
                labels.append(tmp[a].cat.codes[0].astype('int64') - offset)

        # get mask
        masks = [np.float32(0.) if l < 0 else np.float32(1.)
                 for l in labels]

        # hack to remove the negative values when no attribute label is present
        labels = [np.maximum(0, l) for l in labels]
        return labels, masks

    def read_img(self, fname):
        # read img and apply transformations
        img = self.loader(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        return img

    def transform_fname(self, fname):
        return fname

    def __getitem__(self, index):
        """ Read img, transform img and return idx in long int """
        # basic info
        fname = self.split.iloc[index].name
        ins = self.split.iloc[index][self.ins_arg]

        # read img and apply transformations
        img = self.read_img(self.transform_fname(fname))

        # get labels
        labels, masks = self.get_coded_labels(ins, offset=self.offset)

        if self.mode == 'cond':
            # add label about the intance item
            item = self.split[self.ins_arg].cat.codes.iloc[index].astype('int64')

            # rearrange for ACI instead of CA
            labels_ica = labels[1:] + [labels[0]] + [item]
            masks_ica = masks[1:] + [masks[0]] + [np.float32(1.)]

            # sample which condition to select
            cond = np.random.choice(np.squeeze(np.argwhere(np.asarray(masks_ica) == 1.)))
            return img, labels_ica[cond], cond

        elif self.mode in ['full', 'test']:
            # add label about the intance item
            item = self.split[self.ins_arg].cat.codes.iloc[index].astype('int64')

            # return ICA
            labels_ica = [item] + labels

            return img, labels_ica, masks

    def __len__(self):
        return self.split.shape[0]


class CCDataset(MyDataset):
    def __init__(self, split, transform, data_dir,
                 attributes=CC_ATT,
                 loader=default_image_loader,
                 mode='full',
                 ins_arg='model_name',
                 cat_arg=['maker_name']):
        # original split
        self.orig_split = split
        self.split = split

        # split with only unique items
        self.split_ins = self.split.drop_duplicates(subset=ins_arg)

        # image loading stuff
        self.loader = loader
        self.transform = transform
        self.data_dir = data_dir

        # handling the pandas df
        self.mode = mode
        self.attributes = attributes
        self.to_get = cat_arg + self.attributes
        self.ins_arg = ins_arg
        self.cat_arg = cat_arg
        self.offset = 0


class DFDataset(MyDataset):
    def __init__(self, split, transform, data_dir,
                 attributes=DF_ATT,
                 loader=default_image_loader,
                 mode='full',
                 ins_arg='id',
                 cat_arg=['clothes_category']):

        # original split
        self.orig_split = split
        self.attributes = attributes

        # handling the pandas df
        self.mode = mode
        self.to_get = cat_arg + self.attributes
        self.ins_arg = ins_arg
        self.cat_arg = cat_arg

        self.split = split

        # split with only unique items
        self.split_ins = self.split.drop_duplicates(subset=ins_arg)

        # image loading stuff
        self.loader = loader
        self.transform = transform
        self.data_dir = data_dir

        self.offset = 1

    def transform_fname(self, fname, mode='remove'):
        if mode == 'remove':
            return '/'.join(fname.split('/')[1:])
        elif mode == 'add':
            return os.path.join('img', fname)
