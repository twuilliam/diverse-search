import argparse
import sys
import os
import ipdb
import numpy as np
import pandas as pd
import itertools
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('..')


def config():
    parser = argparse.ArgumentParser(description='PyTorch DML')
    parser.add_argument('--dataset', type=str, required=True,
                        help='DF|DIV')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing file')
    args = parser.parse_args()
    return args


def get_essential(args):
    '''Return dataframes, features, keys for category and attribute values'''
    if args.dataset == 'DF':
        from core.data import create_df_splits

        # get data splits
        df_train, df_valid, df_query, df_gallery = create_df_splits(
            fname='../data/deepfashion/all.hdf5')

        df_test = pd.concat([df_query, df_gallery])

        ins_arg = 'id'
        cat_arg = 'clothes_category'
        attributes = ['fabric', 'front', 'hem', 'neck', 'print',
                      'shoulder', 'silhouette', 'sleeve']

    elif args.dataset == 'DIV':
        from core.data import create_div_splits

        # get data splits
        df_train, df_valid, df_test = create_div_splits(
            fname='../data/diverse/all.hdf5')

        ins_arg = 'model_name'
        cat_arg = 'maker_name'
        attributes = ['door_number', 'seat_number', 'type']

    df = {'train': df_train, 'test': df_test}
    keys = {'id': ins_arg, 'cls': cat_arg, 'attr': attributes}

    return df, keys


def query_gen(df_train, df_test, keys, extra=1):
    ''' Semantic query generator
    Generate all possible attribute value combinations
    (wih max number of combinations) given a category
    a bit ugly but does the job
    '''
    # keep labels at the instance level
    df_test_id = df_test.drop_duplicates(subset=keys['id'])

    n_cat = len(df_test[keys['cls']].unique())
    n_attr = len(keys['attr'])

    categories = df_test[keys['cls']].cat.categories
    train_categories = df_train[keys['cls']].cat.categories

    queries = {}
    queries['seen'] = {}
    queries['unseen'] = {}
    queries['seen']['count_img_gal'] = []
    queries['unseen']['count_img_gal'] = []
    queries['seen']['count_ins_gal'] = []
    queries['unseen']['count_ins_gal'] = []

    # iterate over categories
    for coi in categories:
        queries['seen'][coi] = []
        queries['unseen'][coi] = []

        # get dataframe for the coi, with unique combinations for attr
        df_coi = df_test_id.loc[df_test_id[keys['cls']] == coi]
        df_coi = df_coi.drop_duplicates(subset=keys['attr'])

        # generate attribute combinations
        for aoi in itertools.combinations(keys['attr'], extra):
            aoi = list(aoi)
            df_comb = df_coi.drop_duplicates(subset=aoi)[aoi]

            for item in df_comb.values:
                if 'None' in list(item) or coi not in train_categories:
                    # ignore this category+attribute(s) combination
                    pass
                else:
                    # create composite query
                    query = ['None'] * n_attr
                    for i, a in enumerate(aoi):
                        pos = keys['attr'].index(a)
                        query[pos] = item[i]

                    status = checker(df_train, keys, coi, aoi, query)
                    c_img, c_ins = counting(df_test, keys, coi, aoi, query)
                    if status == 'seen':
                        queries['seen'][coi].append(query)
                        queries['seen']['count_img_gal'].append(c_img)
                        queries['seen']['count_ins_gal'].append(c_ins)
                    else:
                        queries['unseen'][coi].append(query)
                        queries['unseen']['count_img_gal'].append(c_img)
                        queries['unseen']['count_ins_gal'].append(c_ins)
    return queries


def checker(df_train, keys, coi, aoi, query):
    '''Checks if the query is present in the training set'''
    count, _ = counting(df_train, keys, coi, aoi, query)
    if count > 0:
        return 'seen'
    else:
        return 'unseen'


def counting(df, keys, coi, aoi, query):
    '''Count how many images and instances exhibit this query'''
    cond = df[keys['cls']] == coi
    df_coi = df.loc[cond]

    cond = []
    for a in aoi:
        pos = keys['attr'].index(a)
        cond.append(df_coi[a] == query[pos])
    cond = reduce(np.logical_and, cond, True)
    return np.count_nonzero(cond), len(df_coi.loc[cond, keys['id']].unique())


def statistics(queries):
    print('[Statistics]')

    count = []

    for i in queries.keys():
        sq = queries[i]['seen']
        uq = queries[i]['unseen']

        count.append(len(sq['count_img_gal']) + len(uq['count_img_gal']))

        print('%d attribute(s)' % i)
        print('.. [seen] %6d [unseen] %6d \tqueries (Total: %d)' %
              (len(sq['count_img_gal']), len(uq['count_img_gal']), count[-1]))
        print('.. [seen] %6.2f [unseen] %6.2f \timages per query' %
              (np.mean(sq['count_img_gal']), np.mean(uq['count_img_gal'])))
        print('.. [seen] %6.2f [unseen] %6.2f \tinstances per query' %
              (np.mean(sq['count_ins_gal']), np.mean(uq['count_ins_gal'])))
    print('-> Total: %d queries' % sum(count))


if __name__ == '__main__':
    args = config()

    if args.dataset == 'DIV':
        fpath = '../data/diverse/queries.npz'
    elif args.dataset == 'DF':
        fpath = '../data/deepfashion/queries.npz'

    if os.path.isfile(fpath) and not args.overwrite:
        queries = np.load(fpath)['queries'].item()
    else:
        df, keys = get_essential(args)
        queries = {}
        print('Generating queries for %s dataset' % args.dataset)
        for i in [1, 2, 3]:
            print('.. %d attribute(s)' % i)
            queries[i] = query_gen(df['train'], df['test'], keys, i)
        np.savez(fpath, queries=queries)

    statistics(queries)
