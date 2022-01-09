import os
import torch
from model import NetVLAD, MoCo, NeXtVLAD, LSTMModule, GRUModule, TCA, CTCA
import h5py
from data import FIVR, FeatureDataset
from torch.utils.data import DataLoader, BatchSampler

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import resize_axis
from sklearn.preprocessing import normalize, MinMaxScaler
import seaborn as sns
import numpy as np

def chamfer(query, target_feature, comparator=False):
    query = torch.Tensor(query).cuda()
    target_feature = torch.Tensor(target_feature).cuda()
    simmatrix = torch.einsum('ik,jk->ij', [query, target_feature])

    if comparator:
        simmatrix = comparator(simmatrix).detach()
    sim = simmatrix.max(dim=1)[0].sum().cpu().item() / simmatrix.shape[0]

    return sim


# eval_feature_path = '/workspace/CTCA/pre_processing/fivr-byol_rmac_187563.hdf5'
# model_dir =  '/mldisk/nfs_shared_/dh/weights/vcdb_rmac_89325_TCA_tsne'
# feature_size_list = [1024]

eval_feature_path = '/workspace/CTCA/pre_processing/fivr-byol_rmac_segment_l2norm.hdf5'
model_dir =  '/mldisk/nfs_shared_/dh/weights/vcdb-byol_rmac-segment_89325_CTCA_tsne'
feature_size_list = [2048]

vid2features = h5py.File(eval_feature_path, 'r')
print('...features loaded')

model_list = os.listdir(model_dir)
model_epochs = sorted([int(model_filename.split('.')[0].split('_')[1])for model_filename in model_list])

dataset = FIVR(version='5k')
test_loader = DataLoader(FeatureDataset(vid2features, dataset.get_queries(),
                               padding_size=300, random_sampling=True),
                batch_size=1, shuffle=False)

for feature_size in feature_size_list:
    model = TCA(feature_size=feature_size, nlayers=1)
    with torch.no_grad():  # no gradient to keys
        for model_epoch in [10]:
            model_path = os.path.join(model_dir, f'model_{model_epoch}.pth')
            print(f'{model_epoch}th loading weights...')
            model.load_state_dict(torch.load(model_path))
            model = model.eval()
            model = model.cuda()
            print(f'...{model_epoch}th weights loaded')
            for q_feature, q_len, q_id in tqdm(test_loader):
                # import pdb;pdb.set_trace()
                q_id = q_id[0]
                if q_len[0].item() >= 1:

                    for type, rs in dataset.annotation[q_id].items():
                        if type=='DS':
                            for r in rs:
                                if r in vid2features.keys():
                                    q1 = q_feature.cuda()[0]
                                    r1 = torch.tensor(vid2features[r]).cuda()
                                    q1 = q1.transpose(0,1)

                                    # q1_minmax_in = np.array(min_max_scaler.fit_transform(q1.clone().cpu()))
                                    # r1_minmax_in = np.array(min_max_scaler.fit_transform(r1.clone().cpu()))
                                    # q1_norm_in = normalize(q1.clone().cpu(), axis=1, norm='l1').cuda()
                                    # r1_norm_in = normalize(r1.clone().cpu(), axis=1, norm='l1').cuda()

                                    # q1_minmax_in = torch.tensor(q1_minmax_in).cuda()
                                    # r1_minmax_in = torch.tensor(r1_minmax_in).cuda()

                                    simmatrix = torch.einsum('ik,jk->ij', [q1[:q_len, :], r1]).detach().cpu()
                                    mn, mx = simmatrix.min(), simmatrix.max()
                                    simmatrix_minmax = ((simmatrix - mn)*2 / (mx - mn))-1
                                    plt.clf()
                                    fig, ax = plt.subplots(figsize=(20, 20))
                                    cax = ax.matshow(simmatrix_minmax, interpolation='nearest',cmap='jet')
                                    plt.axis('off')
                                    # plt.xticks(range(33), rotation=90)
                                    # plt.yticks(range(33))
                                    # fig.colorbar(cax)
                                    plt.savefig(
                                        f'simmatrix/{model_epoch}_{q_id}_{r}_TCA_{feature_size}_in.png',
                                        dpi=300)
                                    plt.show()
                                    # breakpoint()
                                    r1 = r1.cpu()
                                    r_len = torch.tensor([r1.shape[0]])
                                    r1 = resize_axis(r1, axis=0, new_size=300, fill_value=0, random_sampling=True).transpose(-1, -2)
                                    q1 = q1.transpose(0, 1)
                                    q1 = torch.unsqueeze(q1, 0)
                                    r1 = torch.unsqueeze(r1, 0).cuda()

                                    q1_out = model.encode(q1, q_len.cuda())[0]
                                    r1_out = model.encode(r1, r_len.cuda())[0]
                                    # breakpoint()

                                    simmatrix = torch.einsum('ik,jk->ij',[q1_out, r1_out]).detach().cpu()

                                    plt.clf()
                                    fig, ax = plt.subplots(figsize=(20, 20))
                                    cax = ax.matshow(simmatrix, interpolation='nearest',cmap='jet')
                                    plt.axis('off')
                                    plt.savefig(
                                        f'simmatrix/{model_epoch}_{q_id}_{r}_TCA_{feature_size}_out.png',
                                        dpi=300)
                                    plt.show()



                                    # a = chamfer(feature.detach().cpu().numpy()[0].transpose(0, 1), prev_feature.detach().cpu().numpy()[0].transpose(0, 1),
                                    #             False)
                                    # b = chamfer(model.encode(feature, feature_len).detach().cpu().numpy()[0],
                                    #             model.encode(prev_feature, prev_feature_len).detach().cpu().numpy()[0], False)
                                    # print(a, b)