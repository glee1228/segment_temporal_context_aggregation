import glob
import os
import pickle as pk

import h5py
import numpy as np
from tqdm import tqdm

import torch
from pca import PCA
from multiprocessing import Pool

def get_feature_list(root='/mldisk/nfs_shared_/dh/datasets/', dataset='vcdb', feat='imac'):
    print(root + dataset + '/features/' + feat)
    if dataset == 'vcdb' or dataset == 'vcdb-mocov2' or dataset == 'vcdb-byol' :
        dataset = 'vcdb'
    elif dataset == 'vcdb-segment':
        dataset = 'vcdb-segment'
    elif dataset == 'ccweb':
        dataset = 'ccweb'
    elif dataset == 'fivr' or dataset == 'fivr-byol':
        dataset = 'fivr'
    elif dataset == 'fivr-segment':
        dataset = 'fivr-segment'
    if dataset == 'vcdb':
        return sorted(glob.glob(root + dataset + '/features/' + feat +  '/core/*.npy')) + sorted(glob.glob(root + dataset + '/features/' + feat +  '/[0123456789]*/*.npy'))
    return sorted(glob.glob(root + dataset + '/features/' + feat +  '/*/*.npy'))

def export_feature_list(feature_list, out_path):
    with open(out_path, 'w') as f:
        for path in feature_list:
            f.write(path.split('/')[-1].split('.')[-2] + '\t' + path + '\n')

def npy2h5py(feature_list_path, h5path, pca=None):
    paths = [l.split('\t')[1].strip() for l in open(feature_list_path, 'r').readlines()]
    with h5py.File(h5path, 'w') as f:
        for path in tqdm(paths):
            vid = path.split('/')[-1].split('.')[-2]
            if pca:
                f.create_dataset(vid, data=pca.infer(torch.from_numpy(np.load(path)).cuda()).cpu())
            else:
                f.create_dataset(vid, data=np.load(path))   

if __name__ == "__main__":
    datasets = ['fivr-segment']
    feature_types = ['l2norm']
    n_list = [1024]
    for dataset, feature_type, n_components in zip(datasets, feature_types, n_list):
        feature_list = get_feature_list(dataset=dataset, feat=feature_type)
        raw_feature_txt = f'/mldisk/nfs_shared_/dh/datasets/{dataset}/{dataset}{len(feature_list)}_resnet50_{feature_type}_pca_{n_components}.txt'
        export_feature_list(feature_list, out_path=raw_feature_txt)


        def pipe(a):
            a = np.load(a)
            a = a[np.random.choice(len(a), 10), :]
            return a

        paths = [l.split('\t')[1].strip() for l in open(raw_feature_txt, 'r').readlines()]

        pool = Pool(16)
        features = []
        t = tqdm(total=500)
        for path in paths:
            features += [pool.apply_async(pipe,
                                          args=[path],
                                          callback=(lambda *a: t.update()))]
        pool.close()
        pool.join()

        feat_array = []
        for feat in tqdm(features):
            feat_array.append(feat.get())


        feats = np.concatenate(feat_array)
        feats = torch.from_numpy(feats)

        pca = PCA(parameters_path=f'/mldisk/nfs_shared_/dh/datasets/{dataset}/pca_params_{dataset}{len(feature_list)}_resnet50_{feature_type}_3840_{n_components}.npz',n_components=n_components)
        pca.train(feats)

        pca = PCA(parameters_path=f'/mldisk/nfs_shared_/dh/datasets/{dataset}/pca_params_{dataset}{len(feature_list)}_resnet50_{feature_type}_3840_{n_components}.npz',n_components=n_components)
        pca.load()

        paths = get_feature_list(dataset=dataset, feat=feature_type)

        h5path = f'/mldisk/nfs_shared_/dh/datasets/{dataset}/{dataset}_{feature_type}_{len(features)}_{n_components}.hdf5'
        npy2h5py(raw_feature_txt, h5path, pca=pca)

