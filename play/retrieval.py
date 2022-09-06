import argparse
import sys
import os
import ipdb
import faiss
import time
import collections
import numpy as np
import pandas as pd
import itertools
from metrics import apk, preck, reck
sys.path.append('..')


def config():
    parser = argparse.ArgumentParser(description='PyTorch DML')
    parser.add_argument('--dataset', type=str, required=True,
                        help='DIV or DF')
    parser.add_argument('--features', type=str, required=True,
                        help='features dir')
    parser.add_argument('--l2', action='store_true', default=False,
                        help='l2 normalization of the features')
    parser.add_argument('--entire', action='store_true', default=False,
                        help='l2 normalization on the entire embedding space')
    parser.add_argument('--mode', type=str, default='ICA',
                        help='features dir')
    args = parser.parse_args()
    return args


def get_essential(args):
    '''Return dataframes, features, keys for category and attribute values'''
    if args.dataset == 'DIV':
        from core.data import create_div_splits

        # get data splits
        df_train, df_valid, df_test = create_div_splits(
            fname='../data/diverse/all.hdf5')

        # load features
        train = np.load(os.path.join(args.features, 'features_train.npz'))
        gallery = np.load(os.path.join(args.features, 'features_test.npz'))

        feat_train = train['feat']
        feat_test = gallery['feat']

        fpath = '../data/diverse/queries.npz'
        queries = np.load(fpath)['queries'].item()

        ins_arg = 'model_name'
        cat_arg = 'maker_name'
        attributes = ['door_number', 'seat_number', 'type']

    elif args.dataset == 'DF':
        from core.data import create_df_splits

        # get data splits
        df_train, df_valid, df_query, df_gallery = create_df_splits(
            fname='../data/deepfashion/all.hdf5')

        df_test = pd.concat([df_query, df_gallery])

        # load features
        train = np.load(os.path.join(args.features, 'features_train.npz'))
        query = np.load(os.path.join(args.features, 'features_query.npz'))
        gallery = np.load(os.path.join(args.features, 'features_test.npz'))

        feat_train = train['feat']
        feat_query = query['feat']
        feat_gallery = gallery['feat']
        feat_test = np.vstack((feat_query, feat_gallery))

        fpath = '../data/deepfashion/queries.npz'
        queries = np.load(fpath)['queries'].item()

        ins_arg = 'id'
        cat_arg = 'clothes_category'
        attributes = ['fabric', 'front', 'hem', 'neck', 'print',
                      'shoulder', 'silhouette', 'sleeve']

    df = {'train': df_train, 'test': df_test}
    feat = {'train': feat_train, 'test': feat_test}
    keys = {'id': ins_arg, 'cls': cat_arg, 'attr': attributes}

    return df, feat, queries, keys


def normalize(x):
    if len(x.shape) == 1:
        return x / np.linalg.norm(x)
    elif len(x.shape) == 2:
        return x / np.linalg.norm(x, axis=1)[:, None]


def normalize_per_subspace(x, step=50):
    n_attr = x.shape[1] // step
    for i in range(n_attr):
        start = i * step
        stop = start + step
        x[:, start:stop] = normalize(x[:, start:stop])
    return x


def query_trans(cls, attr, keys):
    '''Query transformer
    Transform the output of query_gen (list) to be fed to query_builder (dict)
    '''
    query = {}
    query[keys['cls']] = cls
    for v, a in zip(attr, keys['attr']):
        if v == 'None':
            pass
        else:
            query[a] = v
    return query


def query_builder(query, feat, df, mode='and'):
    '''Query builder
    Build the real value query based on the semantic query
    '''
    cond = [df[k] == v for k, v in zip(query.keys(), query.values())]
    if mode == 'and':
        new_cond = reduce(np.logical_and, cond, True)
        return np.mean(feat[new_cond, :], axis=0)
    elif mode == 'or':
        # note: or and wmeta are similar but are computed differently
        idx = [np.argwhere(c.values) for c in cond]
        # idx = np.unique(np.concatenate(idx))
        idx = np.squeeze(np.concatenate(idx))
        return np.mean(feat[idx, :], axis=0)
    elif mode == 'meta':
        sub = [np.mean(feat[c, :], axis=0) for c in cond]
        return np.mean(sub, axis=0)
    elif mode == 'wmeta':
        sub = [np.sum(feat[c, :], axis=0) for c in cond]
        size = sum([float(np.sum(c)) for c in cond])
        return np.sum(sub, axis=0) / float(size)


def retrieve(query, gallery):
    mse = np.sum((query - gallery) ** 2, axis=1)
    idx = np.argsort(mse)
    return idx


def score(query, idx, df):
    cond = [df[k] == v for k, v in zip(query.keys(), query.values())]
    cond = reduce(np.logical_and, cond, True)

    true_id = list(df[cond].index.values)
    retrieved_id = list(df.iloc[idx].index.values)

    res = {}

    res['mAP'] = apk(true_id, retrieved_id, k=len(df))
    res['mAP100'] = apk(true_id, retrieved_id, k=100)

    res['prec10'] = preck(true_id, retrieved_id, k=10)
    res['prec20'] = preck(true_id, retrieved_id, k=20)
    res['prec30'] = preck(true_id, retrieved_id, k=30)

    cond = [df.iloc[idx][k] == v for k, v in zip(query.keys(), query.values())]
    cond = reduce(np.logical_and, cond, True).values
    res['rec1'] = reck(cond, k=1)
    res['rec5'] = reck(cond, k=5)
    res['rec10'] = reck(cond, k=10)

    return res


def subspace(x, start, end):
    if len(x.shape) == 1:
        return np.ascontiguousarray(x[start:end])
    elif len(x.shape) == 2:
        return np.ascontiguousarray(x[:, start:end])


def prune(q, q_feat, db_feat, keys, step=50):
    aoi = []
    for k in q.keys():
        if k in keys['attr']:
            aoi.append(keys['attr'].index(k))

    tmp_q_feat = []
    tmp_db_feat = []
    for no in aoi:
        start = no * step
        end = start + step

        tmp_q_feat.append(subspace(q_feat, start, end))
        tmp_db_feat.append(subspace(db_feat, start, end))

    return np.hstack(tmp_q_feat), np.hstack(tmp_db_feat)


class DataBase(object):
    def __init__(self, keys, df, feat):
        self.keys = keys
        self.df = df
        self.feat = feat
        self.categories = df['test'][keys['cls']].cat.categories

        d = feat['test'].shape[1]
        self.index_flat = faiss.IndexFlatL2(d)
        self.index_flat.add(feat['test'])

    def retrieve_faiss(self, query):
        K = self.feat['test'].shape[0]
        D, I = self.index_flat.search(query, K)
        return I

    def init_scores(self):
        self.scores = {}
        for i in ['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']:
            self.scores[i] = []

        self.best = {}
        self.best['mAP'] = []
        self.best['q'] = []

    def update_scores(self, res):
        for i in ['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']:
            self.scores[i].append(res[i])

    def update_best(self, res, q):
        self.best['mAP'].append(res['mAP'])
        self.best['q'].append(q)

    def get_best(self):
        return self.best

    def retrieve_single(self, q, mode):
        qf = query_builder(q, self.feat['train'], self.df['train'], mode=mode)
        idx = self.retrieve_faiss(qf[None, :])
        return np.squeeze(idx)

    def get_retrieve_labels(self, q):
        cond = [self.df['test'][k] == v for k, v in zip(q.keys(), q.values())]
        cond = reduce(np.logical_and, cond, True)
        return cond

    def get_score(self, queries, mode='and', pruning=False):
        self.init_scores()
        mAP = []
        mAP100 = []
        rec1 = []
        rec10 = []
        for category in self.categories:
            if len(queries[category]) == 0:
                # some categories have no queries
                pass
            else:
                # the gallery stays the same for all queries
                # first, get the queries
                q = []
                q_feat = []
                for i in range(len(queries[category])):
                    q.append(query_trans(category, queries[category][i], self.keys))
                    q_feat.append(query_builder(
                        q[-1], self.feat['train'], self.df['train'], mode=mode))

                # second, perform the distance measurements
                idx = self.retrieve_faiss(np.asarray(q_feat))

                # third, perform the retrieval scoring
                for i in range(len(queries[category])):
                    res = score(q[i], idx[i, :], self.df['test'])
                    self.update_scores(res)
                    self.update_best(res, q[i])
        return self.scores


def hmean(x, y):
    return (2*x*y)/(x+y)


def compile_scores(seen, unseen, idx=[1, 2, 3],
                   keys=['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']):
    seen_scores = {k: [] for k in keys}
    unseen_scores = {k: [] for k in keys}
    for i in idx:
        for k in keys:
            seen_scores[k].extend(seen[i][k])
            unseen_scores[k].extend(unseen[i][k])

    final = collections.defaultdict(dict)

    for k in keys:
        final[k]['seen'] = np.mean(seen_scores[k])
        final[k]['unseen'] = np.mean(unseen_scores[k])
        final[k]['all'] = np.mean(seen_scores[k] + unseen_scores[k])
        final[k]['hm'] = hmean(np.mean(seen_scores[k]),
                               np.mean(unseen_scores[k]))
    return final


def print_all_scores(scores, keys=['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']):
    return ['[%s] %05.2f' % (k, np.mean(scores[k])*100) for k in keys]


def main():
    args = config()
    df, feat, queries, keys = get_essential(args)

    if args.mode in ['I', 'A', 'IA']:
        feat['train'][:, -50:] = np.finfo(np.float32).eps
        feat['test'][:, -50:] = np.finfo(np.float32).eps
    elif args.mode in ['C']:
        feat['train'][:, :-50] = np.finfo(np.float32).eps
        feat['test'][:, :-50] = np.finfo(np.float32).eps

    if args.entire:
        # normalize the space
        feat['train'] = normalize(feat['train'])
        feat['test'] = normalize(feat['test'])
    else:
        # normalize per subspace
        feat['train'] = normalize_per_subspace(feat['train'])
        feat['test'] = normalize_per_subspace(feat['test'])

    db = DataBase(keys, df, feat)

    print('Performing retrieval')

    # seen and unseen
    mode = 'meta'
    print('\n[Query building mode -- %s]' % mode)

    seen = {}
    unseen = {}
    nq_seen = []
    nq_unseen = []
    to_save = {}
    to_save['seen'] = {}
    to_save['unseen'] = {}
    to_select = 30

    for i in [1, 2, 3]:
        start = time.time()
        seen[i] = db.get_score(queries[i]['seen'], mode=mode)
        unseen[i] = db.get_score(queries[i]['unseen'], mode=mode)

        nq_seen.append(len(queries[i]['seen']['count_img_gal']))
        nq_unseen.append(len(queries[i]['unseen']['count_img_gal']))

        inter = compile_scores(seen, unseen, idx=[i])

        print('** Category + %d attribute(s) **' % i)
        for k in ['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']:
            txt = '\t'.join(print_all_scores(inter[k],
                             keys=['seen', 'unseen', 'all', 'hm']))
            print(k + '\t' + txt)
        print('.. %d seen, %d unseen queries' % (nq_seen[-1], nq_unseen[-1]))
        print('.. Took %.2f sec' % (time.time()-start))

    print('\n** FINAL **')
    final = compile_scores(seen, unseen)
    for k in ['mAP', 'mAP100', 'rec1', 'rec5', 'rec10']:
        txt = '\t'.join(print_all_scores(final[k],
                         keys=['seen', 'unseen', 'all', 'hm']))
        print(k + '\t' + txt)
    print('.. %d seen, %d unseen queries' % (sum(nq_seen), sum(nq_unseen)))


if __name__ == '__main__':
    main()
