import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import horovod.torch as hvd
import utils
from data import VCDBPairDataset
from model import NetVLAD, MoCo, NeXtVLAD, LSTMModule, GRUModule, TCA
import wandb
from scipy.spatial.distance import cdist
import h5py
from data import FeatureDataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.multiprocessing.set_sharing_strategy('file_system')


def train(args):
    # Horovod: initialize library.
    hvd.init()
    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    # Horovod: limit # of CPU threads to be used per worker.
    # torch.set_num_threads(1)
    kwargs = {'num_workers': args.num_workers,
              'pin_memory': True} if args.cuda else {}

    # import h5py
    # features = h5py.File(args.feature_path, 'r', swmr=True)
    # import pdb;pdb.set_trace()

    train_dataset = VCDBPairDataset(annotation_path=args.annotation_path, feature_path=args.feature_path,
                                    padding_size=args.padding_size, random_sampling=args.random_sampling, neg_num=args.neg_num)

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_sz,
                              sampler=train_sampler, drop_last=True, **kwargs)

    model = TCA(feature_size=args.pca_components, nlayers=args.num_layers, dropout=0.2)
    # model = NeXtVLAD(feature_size=args.pca_components)
    model = MoCo(model, dim=args.output_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t,mlp=args.mlp)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = utils.CircleLoss(m=0.25, gamma=256).cuda()

    # Horovod: scale learning rate by lr_scaler.
    if False:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate * lr_scaler,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate * lr_scaler,
                                     weight_decay=args.weight_decay)
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Wandb Initialization
    if args.wandb:
        run = wandb.init(project= args.dataset + '_' + str(args.pca_components) + '_train' , notes='')
        wandb.config.update(args)

    start = datetime.now()
    model.train()
    for epoch in range(1, args.epochs + 1):
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        # import pdb;pdb.set_trace()
        train_loss = 0

        for batch_idx, (a, p, n, len_a, len_p, len_n) in enumerate(train_loader):
            if args.cuda:
                a, p, n = a.cuda(), p.cuda(), n.cuda()
                len_a, len_p, len_n = len_a.cuda(), len_p.cuda(), len_n.cuda()

            output, target = model(a, p, n, len_a, len_p, len_n)

            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % args.print_freq == 0 and hvd.rank() == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(a), len(train_sampler),
                    100. * (batch_idx + 1) * len(a) / len(train_sampler), loss.item()))

        scheduler.step()

        if args.wandb:
            wandb.log({'loss': train_loss / len(train_loader),'lr': scheduler.get_lr()[0]}, step=epoch)

        if hvd.rank() == 0 and epoch % 2 == 0:
            print("Epoch complete in: " + str(datetime.now() - start))
            print("Saving model...")
            os.makedirs(args.model_path,exist_ok=True)
            torch.save(model.encoder_q.state_dict(), os.path.join(args.model_path,f'model_{epoch}.pth'))

        if epoch == 10:
            break
    if args.wandb:
        run.finish()
    del model
    del train_dataset
    del train_loader

def calculate_similarities(query_features, target_feature, metric='euclidean', comparator=None):
    """
      Args:
        query_features: global features of the query videos
        target_feature: global features of the target video
        metric: distance metric of features
      Returns:
        similarities: the similarities of each query with the videos in the dataset
    """
    similarities = []
    if metric == 'euclidean':
        dist = np.nan_to_num(
            cdist(query_features, target_feature, metric='euclidean'))
        for i, v in enumerate(query_features):
            sim = np.round(1 - dist[i] / dist.max(), decimals=6)
            similarities.append(sim.item())
    elif metric == 'cosine':
        # import pdb;pdb.set_trace()x
        dist = np.nan_to_num(cdist(np.squeeze(query_features), target_feature, metric='cosine'))
        for i, v in enumerate(query_features):
            sim = 1 - dist[i]
            similarities.append(sim.item())
    elif metric == 'chamfer':
        global debug_num
        # if debug_num == 1:
        #     import pdb;
        #     pdb.set_trace()
        for q_idx, query in enumerate(query_features):
            sim = chamfer(query, target_feature, comparator)
            # if q_idx == debug_num:
            #     print()
            #     print(query[0]==target_feature[0], sim,' ', q_idx,' ',len(query)==len(target_feature),' ',len(query))
            similarities.append(sim)
        debug_num += 1
    elif metric == 'symm_chamfer':
        for query in query_features:
            sim1 = chamfer(query, target_feature, comparator)
            sim2 = chamfer(target_feature, query, comparator)
            similarities.append((sim1 + sim2) / 2.0)
    else:
        for query in query_features:
            sim1 = chamfer(query, target_feature, comparator)
            sim2 = chamfer(target_feature, query, comparator)
            similarities.append((sim1 + sim2) / 2.0)
    return similarities



def chamfer(query, target_feature, comparator=False):
    query = torch.Tensor(query).cuda()
    target_feature = torch.Tensor(target_feature).cuda()
    simmatrix = torch.einsum('ik,jk->ij', [query, target_feature])

    if comparator:
        simmatrix = comparator(simmatrix).detach()
    sim = simmatrix.max(dim=1)[0].sum().cpu().item() / simmatrix.shape[0]

    return sim



def query_vs_database(model, dataset, args):
    print('loading features...')

    vid2features = h5py.File(args.eval_feature_path, 'r')
    print('...features loaded')
    # Wandb Initialization
    run = None
    if args.wandb:
        run = wandb.init(project= args.dataset + '_' + str(args.pca_components) + '_eval' , notes='')
        wandb.config.update(args)

    model_list = os.listdir(args.model_path)
    model_epochs = sorted([int(model_filename.split('.')[0].split('_')[1])for model_filename in model_list])

    all_refer_ids = []
    for q_id, v in dataset.annotation.items():
        for fivr_task in dataset.annotation[q_id].keys():
            all_refer_ids += dataset.annotation[q_id][fivr_task]

    unique_all_refer_ids = list(set(all_refer_ids))

    with torch.no_grad():  # no gradient to keys
        for model_epoch in model_epochs:
            model_path = os.path.join(args.model_path,f'model_{model_epoch}.pth')
            print(f'{model_epoch}th loading weights...')
            model.load_state_dict(torch.load(model_path))
            model = model.eval()
            if args.cuda:
                model = model.cuda()
            print(f'...{model_epoch}th weights loaded')
            test_loader = DataLoader(
                FeatureDataset(vid2features, dataset.get_queries(),
                               padding_size=args.eval_padding_size, random_sampling=args.eval_random_sampling),
                batch_size=1, shuffle=False)
            # Extract features of the queries
            all_db, queries, queries_ids = set(), [], []

            refers_ids = []
            refers = []
            referes_query_ids = []

            for feature, feature_len, query_id in tqdm(test_loader):
                # import pdb;pdb.set_trace()
                query_id = query_id[0]
                if feature.shape[1] > 0:
                    if args.cuda:
                        feature = feature.cuda()
                        feature_len = feature_len.cuda()
                    # queries.append(model(feature, feature_len).detach().cpu().numpy()[0])
                    if args.metric == 'cosine':
                        queries.append(model(feature, feature_len).detach().cpu().numpy())
                    else:
                        queries.append(model.encode(feature, feature_len).detach().cpu().numpy()[0])
                    queries_ids.append(query_id)
                    all_db.add(query_id)
            queries = np.array(queries)

            test_loader = DataLoader(
                FeatureDataset(vid2features, dataset.get_database(),
                               padding_size=args.eval_padding_size, random_sampling=args.eval_random_sampling),
                batch_size=1, shuffle=False)

            # Calculate similarities between the queries and the database videos
            similarities = dict({query: dict() for query in queries_ids})
            for feature, feature_len, video_id in tqdm(test_loader):
                video_id = video_id[0]
                # print('current video : {} {}'.format(video_id, feature.shape))
                if feature.shape[1] > 0:
                    if args.cuda:
                        feature = feature.cuda()
                        feature_len = feature_len.cuda()
                    if args.metric == 'cosine':
                        embedding = model(feature, feature_len).detach().cpu().numpy()
                    else:
                        embedding = model.encode(
                            feature, feature_len).detach().cpu().numpy()[0]
                    all_db.add(video_id)
                    refers_ids.append(video_id)
                    q_in = False
                    if video_id in unique_all_refer_ids:
                        for q_id, v in dataset.annotation.items():
                            if q_in:
                                break
                            for fivr_task in dataset.annotation[q_id].keys():
                                if video_id in dataset.annotation[q_id][fivr_task] and q_in == False and q_id in queries_ids:
                                    referes_query_ids.append(q_id)
                                    refers.append(embedding)
                                    q_in = True
                                    break
                    else:
                        referes_query_ids.append('back')
                        refers.append(embedding)
                    sims = calculate_similarities(queries, embedding, args.metric, None)

                    for i, s in enumerate(sims):
                        similarities[queries_ids[i]][video_id] = float(s)

            if args.wandb:
                if 'VCDB' in args.dataset:
                    avg_precs = dataset.evaluate(similarities, all_db)


                if 'FIVR' in args.dataset:
                    DSVR, CSVR, ISVR = dataset.evaluate(similarities, all_db)
                    wandb.log({'DSVR': np.mean(DSVR), 'epoch': model_epoch})
                    wandb.log({'CSVR': np.mean(CSVR), 'epoch': model_epoch})
                    wandb.log({'ISVR': np.mean(ISVR), 'epoch': model_epoch})

                    q_ids = queries_ids + referes_query_ids
                    q_nums = []
                    q_dict = {}
                    for idx, q_id in enumerate(queries_ids):
                        q_dict[q_id] = idx + 1
                    q_dict['back'] = 0
                    for idx, id in enumerate(q_ids):
                        try:
                            q_nums.append(q_dict[id])
                        except:
                            breakpoint()

                    model = TSNE(n_components=2)
                    sns.set(style='white', rc={'figure.figsize': (14, 10)})
                    ts = model.fit_transform(np.squeeze(np.concatenate((queries, refers), axis=0)))

                    new_ts = []
                    new_nums = []
                    back_ts = []
                    back_nums = []
                    for t, num in zip(ts, q_nums):
                        if num == 0:
                            back_ts.append(t)
                            back_nums.append(num)
                        else:
                            new_ts.append(t)
                            new_nums.append(num)
                    new_ts = np.array(new_ts)
                    back_ts = np.array(back_ts)

                    new_ts = new_ts[:, :]
                    back_ts = back_ts[:, :]
                    new_nums = new_nums[:]

                    plt.scatter(new_ts[:, 0], new_ts[:, 1], s=10, c=new_nums, cmap='Spectral')
                    plt.scatter(back_ts[:, 0], back_ts[:, 1], s=10, alpha=0.3, c='grey')
                    plt.gca().set_aspect('equal', 'datalim')
                    # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
                    plt.axis('off')
                    # plt.set_xticks(())
                    # plt.set_yticks(())
                    plt.show()
                    plt.savefig(f'tsne/TCA_{np.mean(DSVR)}_{np.mean(CSVR)}_{np.mean(ISVR)}_{args.dataset}_{model_epoch}.png',dpi=300)

                if 'CC_WEB' in args.dataset:
                    dataset.evaluate(similarities, all_db)

                if 'CC_WEB' in args.dataset:
                    dataset.evaluate(similarities, all_db)

            del similarities
            del all_db
        if args.wandb:
            run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--annotation_path', type=str, default='/workspace/TCA/datasets/vcdb.pickle',
                        help='Path to the .pk file that contains the annotations of the train set')
    parser.add_argument('-fp', '--feature_path', type=str, default='/workspace/TCA/pre_processing/vcdb-byol_rmac_89325.hdf5',
                        help='Path to the kv dataset that contains the features of the train set')

    parser.add_argument('-mp', '--model_path', type=str, default='/mldisk/nfs_shared_/dh/weights/vcdb_rmac_89325_TCA_no_momentum',
                        help='Directory where the generated files will be stored')

    # parser.add_argument('-nc', '--num_clusters', type=int, default=256,
    #                     help='Number of clusters of the NetVLAD model')
    parser.add_argument('-od', '--output_dim', type=int, default=1024,
                        help='Dimention of the output embedding of the NetVLAD model')
    parser.add_argument('-nl', '--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('-ni', '--normalize_input', action='store_true',
                        help='If true, descriptor-wise L2 normalization is applied to input')
    parser.add_argument('-nn', '--neg_num', type=int, default=16,
                        help='Number of negative samples of each batch')

    parser.add_argument('-e', '--epochs', type=int, default=81,
                        help='Number of epochs to train the DML network. Default: 5')
    parser.add_argument('-bs', '--batch_sz', type=int, default=64,
                        help='Number of triplets fed every training iteration. '
                             'Default: 256')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5,
                        help='Learning rate of the DML network. Default: 10^-4')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4,
                        help='Regularization parameter of the DML network. Default: 10^-4')

    parser.add_argument('-pc', '--pca_components', type=int, default=1024,
                        help='Number of components of the PCA module.')
    parser.add_argument('-ps', '--padding_size', type=int, default=64,
                        help='Padding size of the input data at temporal axis.')
    parser.add_argument('-rs', '--random_sampling', action='store_true',
                        help='Flag that indicates that the frames in a video are random sampled if max frame limit is exceeded')
    parser.add_argument('-nr', '--num_readers', type=int, default=16,
                        help='Number of readers for reading data')
    parser.add_argument('-nw', '--num_workers', type=int, default=8,
                        help='Number of workers of dataloader')

    # moco specific configs:
    parser.add_argument('-mk','--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('-mm','--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('-mt','--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', default=False,
                        help='mlp layer after encoder')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')

    # eval configs:
    parser.add_argument('-d', '--dataset', type=str, default='FIVR-200K',
                        help='Name of evaluation dataset. Options: CC_WEB_VIDEO, VCDB, '
                             '\"FIVR-200K\", \"FIVR-5K\", \"EVVE\"')
    parser.add_argument('-efp', '--eval_feature_path', type=str, default='/workspace/TCA/pre_processing/fivr-byol_rmac_187563.hdf5',
                        help='Path to the .hdf5 file that contains the features of the dataset')
    parser.add_argument('-eps', '--eval_padding_size', type=int, default=300,
                        help='Padding size of the input data at temporal axis')
    parser.add_argument('-ers', '--eval_random_sampling', action='store_true',
                        help='Flag that indicates that the frames in a video are random sampled if max frame limit is exceeded')
    parser.add_argument('-m', '--metric', type=str, default='cosine',
                        help='Metric that will be used for similarity calculation')

    # log config:
    parser.add_argument('--wandb', default=False,
                        help='wandb'
                        )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # clear model path
    import shutil
    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)

    train(args)



    if 'CC_WEB' in args.dataset:
        from data import CC_WEB_VIDEO
        dataset = CC_WEB_VIDEO()
        eval_function = query_vs_database
    elif 'VCDB' in args.dataset:
        from data import VCDB
        dataset = VCDB()
        eval_function = query_vs_database
    elif 'FIVR' in args.dataset:
        from data import FIVR
        dataset = FIVR(version=args.dataset.split('-')[1].lower())
        eval_function = query_vs_database
    elif 'EVVE' in args.dataset:
        from data import EVVE
        dataset = EVVE()
        eval_function = query_vs_database
    else:
        raise Exception('[ERROR] Not supported evaluation dataset. '
                        'Supported options: \"CC_WEB_VIDEO\", \"VCDB\", \"FIVR-200K\", \"FIVR-5K\", \"EVVE\"')

    model = TCA(feature_size=args.pca_components, nlayers=args.num_layers)

    eval_function(model, dataset, args)
    if args.dataset.split('-')[1].lower() == '200k':
        dataset = FIVR(version='5k')
        args.dataset = 'FIVR-5K'
        eval_function(model, dataset, args)

if __name__ == '__main__':
    main()
