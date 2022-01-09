import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torchvision.transforms._transforms_video as transforms_video
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_video_paths_fivr,VideoDataset
from utils import c2d_r50, i3d_r50, slow_r50, slow_r50_4x16, slow_r50_detection,slowfast_r101,slowfast_r50_detection

from utils import VideoVisualizer
import json
import itertools
import torch.nn.functional as F
from tqdm import tqdm

import cv2
import pickle as pk

class VCDBFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/vcdb/'):
        self.paths = glob.glob(root + 'frames/core/*/*.npy')
        self.paths += glob.glob(root + 'frames/background_dataset/*/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]

class VCDBSegments(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/vcdb/'):
        self.paths = glob.glob(root + 'frames_fps_4/core/*/*.npy')
        self.paths += glob.glob(root + 'frames_fps_4/background_dataset/*/*.npy')
        # self.paths = [path for path in self.paths if 'david_beckham_lights_the_olympic_torch' in path]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        suffix =  self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        if os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb-segment/' + 'features/' + suffix):
            return torch.Tensor([]), suffix
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for segments in frames for img in segments]

        if len(imgs) == 0:
            return torch.Tensor([]), suffix
        return torch.stack(imgs), suffix

class CCWEBFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/ccweb/'):
        self.paths = glob.glob(root + 'frames/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]


class FIVRSegments(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/fivr/'):
        self.paths = glob.glob(root + 'frames_fps_4/core/*.npy')
        self.paths += glob.glob(root + 'frames_fps_4/background_dataset/*/*.npy')
        self.paths = sorted(self.paths)
        self.paths2 = sorted(['/'.join(i.split('/')[-2:]) for i in self.paths])
        self.feature_paths = sorted(glob.glob('/mldisk/nfs_shared_/dh/datasets/' + 'fivr-segment' + '/features/' + 'l2norm' + '/*/*.npy'))
        self.feature_paths2 = sorted(['/'.join(i.split('/')[-2:]) for i in self.feature_paths])
        self.yet_paths = []
        for i, (s, f) in enumerate(zip(self.paths2, self.feature_paths2)):
            if s != f:
                breakpoint()
        self.paths = self.paths[100008:100009]
        self.transform = transform
        # breakpoint()
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        suffix = self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        try:
            frames = np.load(self.paths[idx])
            imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for segments in frames for img in segments]
            if len(imgs) == 0:
                return torch.Tensor([]), suffix

            return torch.stack(imgs), suffix
        except:
            return torch.Tensor([]), suffix
class FIVRFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/fivr/'):
        self.paths = glob.glob(root + 'frames/core/*.npy')
        self.paths += glob.glob(root + 'frames/background_dataset/*/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        # print(len(imgs))
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-2] + '/' + self.paths[idx].split('/')[-1]

class EVVEFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/evve/'):
        self.paths = glob.glob(root + 'frames/*.npy')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        frames = np.load(self.paths[idx])
        imgs = [self.transform(Image.fromarray(img.astype('uint8'))) for img in frames]
        if len(imgs) == 0:
            return torch.Tensor([]), self.paths[idx].split('/')[-1]
        return torch.stack(imgs), self.paths[idx].split('/')[-1]


class Video2Vec():
    def __init__(self, dataset='evve', model='resnet-50', layers=['layer1', 'layer2', 'layer3', 'layer4'], feat='rmac',
                 num_workers=4):
        """ Video2Vec
        :param model: String name of requested model
        :param layer: List of strings depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.model_name = model
        self.feat = feat
        self.low_fps_core_vids = {'SbDubvQyTMg': 12.439516129032258, 'uQCJQ5mNkfo': 15.0, 'aMYeLmS1wHA': 15.0, 'iQkOIYvuqLs': 12.0, '2Dj9SLgeKRU': 18.0, 's8Tse-TqdB8': 15.0, '5XK8WwhCQNI': 19.35, 'WzZYnUVFVyY': 9.878048780487806, 'TPxQok2JvJI': 12.5, 'ovdL2ZU65M8': 9.879414298018949, 'Lv3x1_ZoU6M': 10.0, 'ZCES3NwCOLk': 15.0, 'aE6OhOCfxX4': 10.015, 'RyuPqwjweWM': 10.0, 'ztLARHW4yNY': 15.0, 'RiBHXdqT6fg': 10.0, 'jea8VS-fioY': 10.0, 'EMVc7kFpjZM': 15.0, '2C1AxBwkl2U': 15.0, 'cBqdRKb5i-E': 14.0, 'PQNte_M4dRc': 15.817903212130695, 'N4_9TVLbCX4': 14.083333333333334, 'fHECEP_ETxo': 12.0, 'AsTp3ee5frg': 17.07436334996062, 'hfNNdjGtZoQ': 19.012156110044785, 'Dj-vcU-p1WI': 15.0, 'MKosYhqwbzA': 14.0, '7JFDT6rQRdQ': 12.446275071633238, 'R3DJxjE4cok': 9.873563218390805, 'kY0pMlXqJaQ': 15.0, 'kkHFcX70BOQ': 9.99000999000999, '8opAGQ-lttE': 14.298, 'HXvnGGobDH8': 15.0, 'ipU4NN7qI8I': 15.039052078287305, 'QhRHxPGZU7E': 15.0, 'A1EHe3Xguko': 15.0, 'heEgm5bs6Sg': 15.0, 'RCoHqFGUn6M': 13.524, 'zWz-gKyzNT8': 10.0, '9pBqrpNRetY': 10.0, 'eV-tD30z9Is': 15.0, 'lcG31QONEPo': 9.0, 'yvL_jmXmDDs': 14.0, 'kYBKaDMeGBI': 14.0, '-6eckeV6K_g': 12.0, 'y9dZKz_RD4s': 10.0, 'vYn_yrALfaM': 14.0, 'CvD4K2hydbA': 9.947, 'dN42Q3mO2JM': 15.0, '9RUTwXyJkl4': 15.0, 'CkIl3bdjPG8': 14.0, 'fIXzWB1MSs4': 12.57401413604737, 'RnHXYh_YNGY': 15.0, 'jRu4sbrMSLM': 7.437, '_MbiSHauRV0': 14.913964861438146, 'P_hpppBV7m4': 10.0, 'EvqPkmEGXJA': 15.0, 'rTv7zhsz1xA': 5.871778755499686, '5LkzEXdXeNY': 14.917, 'on-M7z_xFBE': 15.0, 'g0dfaJAC4cY': 14.985014985014985, 'uTuDCzXSErk': 15.0, '5FzMIcuU8og': 14.916666666666666, 'FVpsuxyIy88': 15.0, 'yYnvXgqyGL4': 15.0, '52fn5Yx8IWY': 14.985014985014985, 'dNSQMnXepDw': 14.883990719257541, 'APUnaH1oyz0': 15.0, 'yh__bD5vYeA': 11.882, 'Y5yeJE_NbgU': 14.0, 'SJpw26YkYPI': 6.0, 'ruybuU46WPQ': 16.811195565298604, 'tmhovdRacOw': 15.0, 'Ksda9gmVbTA': 14.947, 'ZTQyl_b4kKY': 13.0, 'NxyV6xutvwA': 14.0, 'CuKyxnlH3qo': 12.061, 'mkgax5aY8MI': 10.0, 'ExbIxv6V77M': 14.083333333333334, '5qQELeFACCQ': 6.0, '-aLyMlc8-9g': 10.0, 'Z9g3nwAysBQ': 14.916666666666666, 'PNqfQ9yptdM': 17.758, 'O7W0deNqj3w': 14.0, 'aZqWnEAwnBI': 15.0, 'qGWt0Ftcii8': 14.89051094890511, 'vP_EN3EZaQA': 15.0, '7eCPb0VnJeE': 10.0, 'N7I2gTCMIUI': 9.87551867219917, 'kmDaHqE_nVQ': 18.0, 'aMRm3xUo8yk': 10.416666666666666, 'V-hMpO-cY_k': 15.0, 'tQDq_gr2qTY': 14.0, 'XS6q13CXm0Y': 15.0, 'yp8945F0LvI': 14.0, '7DMwL_LyXLE': 15.0, 'ARrp5C6tuUI': 14.0, 'Lamg3Q8b8B8': 14.0, 'yhBYfCws4Sc': 15.0, 'XElUlXcpYT0': 10.0, 'PK3sYwEYvvk': 12.0, 'xtKMX1jCZHk': 15.0, 'GYfgTxiixG8': 15.0, 'BDSrmCsdwCQ': 15.0, 'bq7QOVFC0gE': 9.878971255673223, 'jCOOXbt4zuk': 16.0, 'HiOndWvMzC4': 14.89778534923339, 'hHhpgWuEJ2A': 15.0, '7LT2UFKJEKo': 10.0, '5eKgnb5vojE': 10.0, 'tDRjyXZj2fs': 15.0, 'MMy5SJCzvDw': 19.87022350396539, 'HTEjok-Ijk4': 6.0, '8Xwyreo7LiI': 15.0, 'sj8PiVj1iRw': 14.892857142857142, 'FQlbMXqgrCU': 15.0, 'Xhd6kReV_Dc': 19.879, 'ZMpN0YyATMA': 18.0, 'noXErVVyU_0': 10.0, 'qrFAFPZs9i0': 7.5, 'dB9kFhffnmc': 15.0, 'nzTS89LcWPg': 14.876579203109815, 'wQdnRvyLqJc': 14.0, 'f-i1uertbYQ': 15.0, 'YuOjAwUQT10': 10.0, 'K4TGcaQAYOI': 14.916897506925208, 'XtqeKEwjv2o': 19.330779329003093, 'eixULimDf4s': 6.0, 'WmjkFQLJSYE': 12.468, 'QeQH3W4xg30': 14.985014985014985, 'uKAI7q1xMIE': 16.071, 'nkI_VEWw4wM': 10.0, 'kuzq_UsgLOU': 14.0, 'Kk8zr5IiO88': 14.0, 'AJ-euzVL92Q': 15.0, 'RcAjQJlg8Yk': 7.44, 'eGge5vFvc9U': 15.0, 'LUfv-kNOGIA': 7.589567211235311, 'd-FjUwm7U0E': 19.889349930843707, 'Y6kPy6tIMWM': 14.0, 'Srb8H1drzfI': 15.0, 'rnwIyl3qUBU': 15.0, 'UMuGOPuhnKU': 10.015, 'UMDzTnucUs0': 12.447698744769875, 'J7YJAl5wKYk': 11.833333333333334, '3ycX-PMBcGw': 10.0, 'bTcTKjqiviI': 15.0, 'NgGfTQ6aWR0': 14.883333333333333, 'BQ1XJfXMQGM': 14.985014985014985, 'dMd-grj9UYE': 14.0, 'u3aQCICUuTU': 10.0, 'bxUYIOtsJ1c': 6.0, '7VqbayGBWdM': 15.0, 'VZpCVPjHxBI': 15.0, 'iTTn8XzVwcA': 15.0, 'WgCdbJtABTI': 6.0, 'pWb9lE91_AU': 12.0, '9SolJ4r6iSs': 9.889380530973451, 'tM-m7walunw': 14.0, 'rN0Z9TI2FhY': 6.0, '1_YFQvSa1Cs': 15.0, 's9IRR47rB_s': 15.0, 'Dd1ikCshPKA': 19.05451576132888, 'Lx9NmNL1S1c': 16.0, 'H-8d990H4uk': 14.0, 'D3k0FQhHy9g': 16.67407840842598, '9V1UcFl-0xI': 15.0, 'ql8dsiV7I-0': 18.0, '2XJdbJzws60': 14.89655172413793, 'HLfeFrcALDk': 15.0, 'pEcFRp54UDw': 6.0, 'UZbT_BFqusw': 15.0, 'TRANkIjnICY': 12.54, 'erhvwBYIQnA': 15.0, 'QUbUWW-EOIM': 15.0, 'uolqTsATsvQ': 10.0, 'cQF-tVAtzVk': 16.811195565298604, '0xHhpXjJfgY': 10.0, '4n-T-PrtB1s': 15.0, 'a0alIfQWAbg': 15.0, 'ekrh698ZndA': 15.0, 'FELB-TYaklI': 15.01687762730688, 'dd-4KB0bHiw': 15.0, 'qfCiyJJQ5dg': 10.0, 'LbRJseo5pIU': 15.0, 'PApEyX2MUQc': 12.5, 'jBddOh-gZuQ': 15.0, 'Epd0HnH6d-k': 15.0, 'k38flCfK6co': 10.0, 'vrYr5RAqfbI': 10.0, 'LTM8iD3ffAQ': 15.017, 'jGK2ycDokqM': 6.0, 'c_1Eojr-B8k': 6.0, '-1-Ll5KspdE': 14.927360774818402, 'ys16kF18nqQ': 14.0, 'uQGllY3r3ro': 10.0, 'jpOO1-M_918': 15.0, 'spB31FIwP7w': 6.0, '1-aY9DUOSDI': 10.0, 'S74xLujZsUU': 10.0, 'bTSaEKdGeRk': 15.0, 'daR6fjsNcqI': 10.0, 'wXrmQ3xIp0s': 10.0, '8Et3sJrL4Ak': 15.0, 'uknpNOVZvag': 14.0, 'eXGBipLubnA': 12.0, 'BO3Kl4Gn7ZA': 10.787451984635084, 'VKrfH8gLUCg': 10.0, 'yZnhhTV65tg': 14.0, '2ISOFVSyAFw': 15.0, 'GE5ZMuXwCZY': 14.916666666666666, 'NE0diJY_JKU': 15.0, 'rZdb4DVETPQ': 10.0, 'yiwyJUFKYck': 15.0, 'aCnRukjYPBE': 10.0, 'fgVHeK2hIVI': 15.0, 'rCJbsOhOs7A': 19.713, 'fKw1mJ1kzVk': 15.0, 'eqCjpbgtyx0': 14.916666666666666, 'uZcebP5zBbo': 10.0, '-RuW92lwdv4': 9.957, 'g7lkW_7WOPM': 9.877777777777778, 'wnHeSroVuQs': 6.0, 'fQJrbtPM8Qw': 15.0, 's0DDC2Sl71M': 8.07, 'EO_y5BbLvDw': 10.0, 'Ne6LqTZzsWE': 12.87893982808023, 'AqAE2bZMYz0': 14.0, 'NnctBQz9YAU': 14.0, '9iWO5s_dhjk': 12.453874538745387, '-TPmO7dGVNQ': 15.0, 'HeMK3U9Dqng': 12.44646680942184, '7PDnFx6uYoY': 15.0, 'Etq3Te4VZ6U': 14.0, 'aUlIzZxZEbg': 15.610405394876944, 'yu9QI8OhuFg': 15.1, 'sPVBhD16-9M': 15.0, 'cWFl55niptM': 10.0, 'TYqvNYSRmr0': 12.450396825396826, 'hccJUcAEG-8': 14.0, '1CgI7tSNN5o': 17.0, 'N1G1jpnhgF0': 12.0, '02Ly_cxmzcE': 6.0, 'HXNkB79ad_k': 15.0, '9TJF46wOp38': 14.0, '6qHxuuQPCbs': 15.0, 'L8AzN1VEC0c': 15.0, 'liIi-QGTZOg': 14.913970721127779, 'Qnycodzxz_M': 8.333, 'rDXa0e41HrI': 6.0, '9Wc_tWIx3G4': 11.867787369377556, 'R0ZQ-s7gy34': 14.985014985014985, 'gfNPUk6gINY': 15.0, 'nfV-w4B2OXs': 14.872773536895675, 'gkOT9UVNRIg': 15.0, 'lZZnW4-KpV8': 15.0, 'SIpw-spgn20': 15.0, 'uU4B9UitocQ': 15.0, '8tZwFwnYU10': 15.0, 'LVxS6tgHRus': 14.0, '-tL42UZPmrs': 11.926153846153847, 'XglribWzDRk': 15.0, '1yXIWQnrtuc': 16.675352112676055, 'B1cYbSL6bsg': 15.0, '-izRIM00JFc': 10.0, 'FczIP2IuNTc': 15.0, 'aPC87JAvWB4': 10.0, 'jxMl3xhqQ0A': 12.5, 'TZrbrfNGrbI': 14.0, 'ImXsKAhxno8': 14.0, 'HlYa7nWaIlA': 15.0, 'kyORYfea9gQ': 10.0, 'a9U8lGJ2kXc': 14.889705882352942, 'NEh3F9ASQ3I': 14.875074360499703, 'xpSS4CgpyTw': 14.0, 'HjywBV2eeyc': 10.0, '7tIUsIra22Q': 14.876644736842104, 'NMdbiiBL0S0': 6.0, 'RjF09NrwxNk': 14.8828125, '7x614r6wXz8': 15.0, '4PlUahwabiQ': 12.444196428571429, 'mPA8XPcOL3k': 14.0, 'VCHxOe0Mfvw': 14.985014985014985, 'gK9QVhGQSeA': 14.913964861438146}

        self.model, self.extraction_layers = self._get_model_and_layers(model, layers)
        if self.model_name =='slow-resnet-50':
            self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.model.eval()
        if model == 'slow-resnet-50':
            self.transformer = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transformer = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if dataset == 'vcdb':
            self.data_loader = DataLoader(VCDBFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'vcdb-segment':
            self.data_loader = DataLoader(VCDBSegments(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'ccweb':
            self.data_loader = DataLoader(CCWEBFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'fivr':
            self.data_loader = DataLoader(FIVRFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'fivr-segment':
            self.data_loader = DataLoader(FIVRSegments(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        elif dataset == 'evve':
            self.data_loader = DataLoader(EVVEFrames(transform=self.transformer), batch_size=1, shuffle=False,
                                          num_workers=num_workers)
        else:
            raise KeyError('Dataset %s was not found' % dataset)

    def get_vec(self, path):
        """ Get vector embedding from a video(frames)
        :returns: Numpy ndarray
        """

        buffer = []

        # hook function
        def imac(module, input, output):
            logits = F.normalize(output.detach())
            # print(logits.shape)
            logits = F.max_pool2d(logits, kernel_size=logits.size()[2:])
            # print(logits.shape)
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())

        def rmac(module, input, output):
            logits = F.normalize(output.detach())
            # print(logits.shape)
            p = int(min(logits.size()[-2], logits.size()[-1]) * 2 / 7)  # 28->8 14->4 7->2
            logits = F.max_pool2d(logits, kernel_size=(int(p + p / 2), int(p + p / 2)), stride=p)  # (n, c, 3, 3)
            # print(logits.shape)
            logits = logits.view(logits.size()[0], logits.size()[1], -1)  # (n, c, 9)
            # print(logits.shape)
            logits = F.normalize(logits)
            logits = torch.sum(logits, dim=-1)
            # print(logits.shape)
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())


        with torch.no_grad():
            for data, suffix in tqdm(self.data_loader):
                data = data[0]  # batch_size == 1, current shape (frames, c, h, w)
                suffix = self.feat + '/' + suffix[0]

                if data.shape[0] == 0 or os.path.exists(path + 'features/' + suffix):
                    continue

                if self.model_name == 'slow-resnet-50':
                    len_segment = data.shape[0] / 4
                    if not len_segment.is_integer():
                        print(f'segment length is not correct : {len_segment}')
                    data = data.reshape(int(len_segment), 4, 3, data.shape[-2], data.shape[-1]).transpose(1,2)


                if data.shape[0] >= 2400:
                    datas = torch.split(data, data.shape[0] // 8, dim=0)
                elif data.shape[0] >= 2100:
                    datas = torch.split(data, data.shape[0] // 7, dim=0)
                elif data.shape[0] >= 1800:
                    datas = torch.split(data, data.shape[0] // 6, dim=0)
                elif data.shape[0] >= 1500:
                    datas = torch.split(data, data.shape[0] // 5, dim=0)
                elif data.shape[0] >= 1200:
                    datas = torch.split(data, data.shape[0] // 4, dim=0)
                elif data.shape[0] >= 900:
                    datas = torch.split(data, data.shape[0] // 3, dim=0)
                elif data.shape[0] >= 600:
                    datas = torch.split(data, data.shape[0] // 2, dim=0)
                else:
                    datas = [data]

                features = []

                for data in datas:
                    buffer = []
                    handles = []

                    data = data.cuda()
                    if not self.model_name == 'slow-resnet-50':
                        for layer in self.extraction_layers:
                            if self.feat == 'imac':
                                h = layer.register_forward_hook(imac)
                            else:
                                h = layer.register_forward_hook(rmac)
                            handles.append(h)


                    h_x = self.model(data)

                    if self.model_name == 'slow-resnet-50':
                        h_x = F.normalize(h_x.detach())
                        h_x = h_x.detach().cpu().numpy()
                        features.append(h_x)

                    for h in handles:
                        h.remove()

                    del h_x
                    del data

                    if not self.model_name == 'slow-resnet-50':
                        features.append(np.concatenate(buffer, axis=1))

                os.makedirs(path + 'features/' + '/'.join(suffix.split('/')[:2]), exist_ok=True)
                np.save(path + 'features/' + suffix, np.squeeze(np.concatenate(features, axis=0)))


    def load_pretrained(self, model, pretrained_model, logger=print):
        ckpt = torch.load(pretrained_model, map_location='cpu')
        if len(ckpt) == 3:  # moco initialization
            ckpt = {k[17:]: v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder_q')}

            # for fc in ('fc_inter', 'fc_intra', 'fc_order', 'fc_tsn'):
            #     ckpt[fc + '.0.weight'] = ckpt['fc.0.weight']
            #     ckpt[fc + '.0.bias'] = ckpt['fc.0.bias']
            #     ckpt[fc + '.2.weight'] = ckpt['fc.2.weight']
            #     ckpt[fc + '.2.bias'] = ckpt['fc.2.bias']
        elif 'byol' in pretrained_model:
            ckpt = ckpt
        else:
            ckpt = ckpt['model']
        [misskeys, unexpkeys] = model.load_state_dict(ckpt, strict=False)
        logger('Missing keys: {}'.format(misskeys))
        logger('Unexpect keys: {}'.format(unexpkeys))
        logger("==> loaded checkpoint '{}'".format(pretrained_model))

    def _get_model_and_layers(self, model_name, layers):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-50'
        :param layers: layers as a string for resnet-50 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-50':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))
            # breakpoint()
            return model, model_layers
        elif model_name == 'resnet-50-mocov2':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))
            self.load_pretrained(model,
                                 pretrained_model='/mldisk/nfs_shared_/dh/weights/mocov2/moco_v2_200ep_pretrain.pth.tar')

            return model, model_layers
        elif model_name == 'resnet-50-byol':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))

            self.load_pretrained(model,
                                 pretrained_model='/mldisk/nfs_shared_/dh/weights/byol/byol-pretrain_res50x1.pth.tar')

            return model, model_layers
        elif model_name == 'slow-resnet-50':
            model = slow_r50_4x16(True)
            model.blocks[5].dropout = torch.nn.Identity()
            model.blocks[5].proj = torch.nn.Identity()

            model_layers = []

            for layer in [5]:
                model_layers.append(model._modules.get('blocks')[layer])

            return model, model_layers
        else:
            raise KeyError('Model %s was not found' % model_name)


if __name__ == "__main__":
    feature_num = 2
    if feature_num == 1:
        v2v = Video2Vec(dataset='fivr', model='resnet-50', layers=['layer1','layer2','layer3','layer4'], num_workers=12, feat='rmac')
        v2v.get_vec(path='/mldisk/nfs_shared_/dh/datasets/fivr/')
    elif feature_num == 2:
        v2v = Video2Vec(dataset='fivr-segment', model='slow-resnet-50', num_workers=12, feat='l2norm')
        v2v.get_vec(path='/mldisk/nfs_shared_/dh/datasets/fivr-segment/')

