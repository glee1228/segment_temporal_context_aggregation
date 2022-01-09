import io
import pickle as pk
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dataloader import KVReader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, utils

from utils import resize_axis
import wandb


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self,
                 vid2features,
                 videos,
                 padding_size=300,
                 random_sampling=False):
        super(FeatureDataset, self).__init__()
        self.vid2features = vid2features
        self.padding_size = padding_size
        self.random_sampling = random_sampling
        self.videos = videos
        self.keys = set(self.vid2features.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        if self.videos[index] in self.keys:
            feat = self.vid2features[self.videos[index]][:]
            len_feat = len(feat)
            return resize_axis(feat, axis=0,
                               new_size=self.padding_size, fill_value=0,
                               random_sampling=self.random_sampling).transpose(-1, -2), len_feat, self.videos[index]
        else:
            return torch.Tensor([]), 0, 'None'

class VCDBPairDataset(Dataset):
    def __init__(self,
                 annotation_path,
                 feature_path='/mldisk/nfs_shared_/dh/datasets/vcdb/vcdb_imac.hdf5',
                 padding_size=300,
                 random_sampling=False,
                 neg_num=1):
        self.feature_path = feature_path
        self.padding_size = padding_size
        self.random_sampling = random_sampling
        self.neg_num = neg_num
        self.features = h5py.File(self.feature_path, 'r', swmr=True)
        self.pairs = []
        self.vcdb = pk.load(open(annotation_path, 'rb'))
        for pair in self.vcdb['video_pairs']:
            vid1, vid2 = pair['videos'][0], pair['videos'][1]
            self.pairs.append([vid1, vid2])
        self.negs = self.vcdb['negs']
        self.negs = [n for n in self.negs if n in self.features.keys()]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ns = random.sample(self.negs, self.neg_num)
        feat_a, feat_p, feat_n = self.features[self.pairs[index][0]][:], self.features[self.pairs[index][1]][:], [self.features[item][:] for item in ns]

        len_a, len_p, len_n = torch.Tensor([len(feat_a)]), torch.Tensor([len(feat_p)]), torch.Tensor([len(item) for item in feat_n])
        a = resize_axis(feat_a, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)
        p = resize_axis(feat_p, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)
        n = torch.stack([resize_axis(item, axis=0, new_size=self.padding_size, fill_value=0,
                                     random_sampling=self.random_sampling).transpose(-1, -2) for item in feat_n])
        return a, p, n, len_a, len_p, len_n

class FSAVCDBPairDataset(Dataset):
    def __init__(self,
                 annotation_path,
                 frame_feature_path='/mldisk/nfs_shared_/dh/datasets/vcdb/vcdb_imac.hdf5',
                 segment_feature_path = '/mldisk/nfs_shared_/dh/datasets/vcdb/vcdb_imac.hdf5',
                 padding_size=300,
                 random_sampling=False,
                 neg_num=1,
                 augmentation=False):
        self.frame_feature_path = frame_feature_path
        self.segment_feature_path = segment_feature_path
        self.padding_size = padding_size
        self.random_sampling = random_sampling
        self.neg_num = neg_num
        self.frame_features = h5py.File(self.frame_feature_path, 'r', swmr=True)
        self.segment_features = h5py.File(self.segment_feature_path, 'r', swmr=True)

        self.pairs = []
        self.vcdb = pk.load(open(annotation_path, 'rb'))
        for pair in self.vcdb['video_pairs']:
            vid1, vid2 = pair['videos'][0], pair['videos'][1]
            self.pairs.append([vid1, vid2])
        self.negs = self.vcdb['negs']
        self.diffs = ['uOGksc5CoUQ','ZtzjFEgIi6E','6H5LXIRow50','3tvH9u9jllQ']
        self.negs = [n for n in self.negs if n in self.segment_features.keys() and n not in self.diffs]
        self.detect_frame_segment_diff = False
        self.augmentation = augmentation
        # breakpoint()

    def temporal_cutout(self, video):
        # video = torch.Tensor(video)
        l = video.shape[0]
        drops = np.random.rand(l) < 0.2
        # Assume we always fill with the previous frame
        drops[0] = False
        drops[-1] = False
        aug_idx = list(range(l))
        for i, drop in enumerate(drops):
            if drop:
                aug_idx[i] = aug_idx[i - 1]
        video = video[aug_idx]
        return video

    def augment(self, video):
        if video.shape[0] > 8:
            rnd = np.random.uniform()

            if rnd < 0.1:
                mask = np.random.rand(video.shape[0]) > 0.3
                # print(video.shape[0])
                # print(mask.shape)
                if np.sum(mask):
                    video = video[mask.nonzero()]
            elif rnd < 0.2:
                video = video[::2]
            if rnd < 0.3:
                self.temporal_cutout(video)

        return video

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ns = random.sample(self.negs, self.neg_num)
        frame_feat_a, frame_feat_p, frame_feat_n = self.frame_features[self.pairs[index][0]][:], self.frame_features[self.pairs[index][1]][:], [self.frame_features[item][:] for item in ns]
        segment_feat_a, segment_feat_p, segment_feat_n = self.segment_features[self.pairs[index][0]][:], self.segment_features[self.pairs[index][1]][:], [self.segment_features[item][:] for item in ns]


        feat_a = np.concatenate((frame_feat_a,segment_feat_a),axis=1)
        feat_p = np.concatenate((frame_feat_p,segment_feat_p),axis=1)

        if self.augmentation:
            feat_a = self.augment(feat_a)
            feat_a = self.augment(feat_a)

        if self.detect_frame_segment_diff :
            equal_mat = np.equal([ f.shape for f in frame_feat_n],[ s.shape for s in segment_feat_n] )
            print(equal_mat.shape)
            for k,eq in enumerate(equal_mat):
                print(eq)
                if eq[0] == False or eq[1] == False:
                    print(ns)
                    print([(f.shape,s.shape) for f, s in zip(frame_feat_n, segment_feat_n)])
        if self.augmentation:
            feat_n = [self.augment(np.concatenate((f, s), axis=1)) for f, s in zip(frame_feat_n, segment_feat_n)]
        else:
            feat_n = [np.concatenate((f,s),axis=1) for f, s in zip(frame_feat_n,segment_feat_n)]

        # print(feat_a.shape,feat_p.shape,len(feat_n),feat_n[0].shape)

        len_a, len_p, len_n = torch.Tensor([len(feat_a)]), torch.Tensor([len(feat_p)]), torch.Tensor([len(item) for item in feat_n])
        a = resize_axis(feat_a, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)

        p = resize_axis(feat_p, axis=0, new_size=self.padding_size, fill_value=0,
                        random_sampling=self.random_sampling).transpose(-1, -2)
        n = torch.stack([resize_axis(item, axis=0, new_size=self.padding_size, fill_value=0,
                                     random_sampling=self.random_sampling).transpose(-1, -2) for item in feat_n])
        return a, p, n, len_a, len_p, len_n

class CC_WEB_VIDEO(object):

    def __init__(self):
        with open('datasets/cc_web_video.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.database = dataset['vid2index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(map(str, self.database.keys()))

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 25)
        print('All dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=False),
            self.calculate_mAP(similarities, all_videos=True, clean=False)))

        print('Clean dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(
            self.calculate_mAP(similarities, all_videos=False, clean=True),
            self.calculate_mAP(similarities, all_videos=True, clean=True)))


class VCDB(object):

    def __init__(self):
        with open('datasets/vcdb.pickle', 'rb') as f:
            dataset = pk.load(f, encoding='latin1')
        self.database = dataset['index']
        self.queries = dataset['index'][:528]
        self.ground_truth = dict({query: set() for query in self.queries})
        for query in self.queries:
            self.ground_truth[query].add(query)
        for pair in dataset['video_pairs']:
            self.ground_truth[pair['videos'][0]].add(pair['videos'][1])
            self.ground_truth[pair['videos'][1]].add(pair['videos'][0])

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db):

        query_gt = self.ground_truth[query]
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
            # if video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
                    # if (i+1)==len(query_gt):
                    #     print(f'query/db : {query[:10]}/{video[:10]} | recall : {i/len(query_gt):.4f} | precision : {s/i:.4f} '
                    #       f' | video count : {int(i)}/{len(query_gt)} ')


        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        ans = []
        for query, res in similarities.items():
            ans.append(self.calculate_mAP(query, res, all_db))
        # import pdb;pdb.set_trace()
        print('=' * 5, 'VCDB Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('VCDB mAP: {:.4f}'.format(np.mean(ans)))

        return ans

class FIVR(object):

    def __init__(self, version='5k'):
        self.version = version
        with open('datasets/fivr.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.annotation = dataset['annotation']
        self.queries = dataset[self.version]['queries']
        self.database = dataset[self.version]['database']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = []
        for label in relevant_labels:
            if label in gt_sets:
                query_gt.append(gt_sets[label])

        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)

        i, ri, s = 0.0, 0, 0.0

        if len(query_gt) == 0:   # empty set check
            return None
        for video in sorted(res.keys(), key=lambda x: res[x], reverse=True):
            if video != query and video in all_db:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri

        return s / len(query_gt)

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        DSVR, CSVR, ISVR = [], [], []
        dsvr_ap, csvr_ap, isvr_ap = 0.0, 0.0, 0.0
        for query, res in similarities.items():
            dsvr_ap = self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS'])
            csvr_ap = self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS', 'CS'])
            isvr_ap = self.calculate_mAP(query, res, all_db,
                                           relevant_labels=['ND', 'DS', 'CS', 'IS'])

            if dsvr_ap is not None:
                DSVR.append(dsvr_ap)
            if csvr_ap is not None:
                CSVR.append(csvr_ap)
            if isvr_ap is not None:
                ISVR.append(isvr_ap)

        print('=' * 5, 'FIVR-{} Dataset'.format(self.version.upper()), '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 16)
        print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
        print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
        print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))

        return DSVR, CSVR, ISVR

class EVVE(object):

    def __init__(self):
        with open('datasets/evve.pickle', 'rb') as f:
            dataset = pk.load(f)
        self.events = dataset['annotation']
        self.queries = dataset['queries']
        self.database = dataset['database']
        self.query_to_event = {qname: evname
                               for evname, (queries, _, _) in self.events.items()
                               for qname in queries}

    def get_queries(self):
        return list(self.queries)

    def get_database(self):
        return list(self.database)

    def score_ap_from_ranks_1(self, ranks, nres):
        """ Compute the average precision of one search.
        ranks = ordered list of ranks of true positives (best rank = 0)
        nres  = total number of positives in dataset
        """
        if nres == 0 or ranks == []:
            return 0.0

        ap = 0.0

        # accumulate trapezoids in PR-plot. All have an x-size of:
        recall_step = 1.0 / nres

        for ntp, rank in enumerate(ranks):
            # ntp = nb of true positives so far
            # rank = nb of retrieved items so far

            # y-size on left side of trapezoid:
            if rank == 0:
                precision_0 = 1.0
            else:
                precision_0 = ntp / float(rank)
            # y-size on right side of trapezoid:
            precision_1 = (ntp + 1) / float(rank + 1)
            ap += (precision_1 + precision_0) * recall_step / 2.0
        return ap

    def evaluate(self, similarities, all_db=None):
        results = {e: [] for e in self.events}
        if all_db is None:
            all_db = set(self.database).union(set(self.queries))

        not_found = 0
        for query in self.queries:
            if query not in similarities:
                not_found += 1
            else:
                res = similarities[query]
                evname = self.query_to_event[query]
                _, pos, null = self.events[evname]
                if all_db:
                    pos = pos.intersection(all_db)
                pos_ranks = []

                ri, n_ext = 0.0, 0.0
                for ri, dbname in enumerate(sorted(res.keys(), key=lambda x: res[x], reverse=True)):
                    if dbname in pos:
                        pos_ranks.append(ri - n_ext)
                    if dbname not in all_db:
                        n_ext += 1

                ap = self.score_ap_from_ranks_1(pos_ranks, len(pos))
                results[evname].append(ap)

        print('=' * 18, 'EVVE Dataset', '=' * 18)

        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(
                not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos\n'.format(len(all_db - set(self.queries))))
        print('-' * 50)
        ap = []
        for evname in sorted(self.events):
            queries, _, _ = self.events[evname]
            nq = len(queries.intersection(all_db))
            ap.extend(results[evname])
            print('{0: <36} '.format(evname), 'mAP = {:.4f}'.format(
                np.sum(results[evname]) / nq))

        print('=' * 50)
        print('overall mAP = {:.4f}'.format(np.mean(ap)))

if __name__ == '__main__':
    import horovod.torch as hvd
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from model import TCA, MoCo

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    kwargs = {'num_workers': 12,
              'pin_memory': True}

    # train_dataset = VCDBPairDataset(annotation_path='/workspace/TCA/datasets/vcdb.pickle', feature_path='/mldisk/nfs_shared_/dh/datasets/vcdb/vcdb_imac_all.hdf5',
    #                                 padding_size=300, random_sampling=True, neg_num=5)
    train_dataset = FSAVCDBPairDataset(annotation_path='/workspace/TCA/datasets/vcdb.pickle',
                                    frame_feature_path='/workspace/TCA/pre_processing/vcdb-byol_rmac_89325.hdf5',
                                    segment_feature_path='/workspace/TCA/pre_processing/vcdb-segment_l2norm_89325.hdf5',
                                    padding_size=300, random_sampling=True, neg_num=16)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)


    train_loader = DataLoader(train_dataset, batch_size=256,
                              sampler=train_sampler, drop_last=True, **kwargs)
    model = TCA(feature_size=2048, nlayers=1, dropout=0.2)

    model = MoCo(model, dim=2048, K=65536, m=0.999, T=0.07)
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.

    lr_scaler = hvd.local_size()
    for batch_idx, (a, p, n, len_a, len_p, len_n) in enumerate(train_loader):
        a, p, n = a.cuda(), p.cuda(), n.cuda()
        len_a, len_p, len_n = len_a.cuda(), len_p.cuda(), len_n.cuda()
        # breakpoint()
        output, target = model(a, p, n, len_a, len_p, len_n)

    # import glob
    # root = '/mldisk/nfs_shared_/MLVD/FIVR'
    # with open('datasets/fivr.pickle', 'rb') as f:
    #     dataset = pk.load(f)
    # annotation = dataset['annotation']
    # queries = dataset['5k']['queries']
    # database = dataset['5k']['database']
    # paths = glob.glob(root + 'frames/core/*.mp4')
    # # paths += glob.glob(root + 'frames/background_dataset/*/*.mp4')
    #
    # import shutil
    # q = queries[1]
    # print(q)
    # ds=annotation[q]['DS']
    # # cs = annotation[q]['CS']
    #
    #
    # import os
    # for q in queries:
    #     query_path = f'{root}/frames/core/{q}.mp4'
    #     try:
    #         shutil.copytree(query_path,f'/workspace/TCA/fivr_sample/{q}.mp4')
    #     except :
    #         pass
    # for d in ds:
    #     origin_path = os.path.join(root,f'frames/core/{d}.mp4')
    #     save_path = f'/workspace/TCA/fivr_sample/ds'
    #     os.makedirs(save_path, exist_ok=True)
    #     try:
    #         shutil.copytree(origin_path, f'{save_path}/{d}.mp4')
    #     except:
    #         pass
    # import pdb;pdb.set_trace()