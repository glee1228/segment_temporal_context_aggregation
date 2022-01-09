import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

import numpy as np
from PIL import Image
from tqdm import tqdm

class VCDBFrames(Dataset):
    def __init__(self, transform, root='/mldisk/nfs_shared_/dh/datasets/vcdb/'):
        self.paths = glob.glob(root + 'frames/core/*.npy')
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
    def __init__(self, dataset='evve', model='resnet-50', layers=['layer1', 'layer2', 'layer3', 'layer4'], feat='rmac', num_workers=4):
        """ Video2Vec
        :param model: String name of requested model
        :param layer: List of strings depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.model_name = model
        self.feat = feat
        
        self.model, self.extraction_layers = self._get_model_and_layers(model, layers)

        self.model = self.model.cuda()

        self.model.eval()

        self.transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if dataset == 'vcdb':
            self.data_loader = DataLoader(VCDBFrames(transform=self.transformer), batch_size=1, shuffle=False, num_workers=num_workers)
        elif dataset == 'ccweb':
            self.data_loader = DataLoader(CCWEBFrames(transform=self.transformer), batch_size=1, shuffle=False, num_workers=num_workers)
        elif dataset == 'fivr':
            self.data_loader = DataLoader(FIVRFrames(transform=self.transformer), batch_size=1, shuffle=False, num_workers=num_workers)
        elif dataset == 'evve':
            self.data_loader = DataLoader(EVVEFrames(transform=self.transformer), batch_size=1, shuffle=False, num_workers=num_workers)
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
            p = int(min(logits.size()[-2], logits.size()[-1]) * 2 / 7) # 28->8 14->4 7->2
            logits = F.max_pool2d(logits, kernel_size=(int(p + p / 2), int(p + p / 2)), stride=p) # (n, c, 3, 3)
            # print(logits.shape)
            logits = logits.view(logits.size()[0], logits.size()[1], -1) # (n, c, 9)
            # print(logits.shape)
            logits = F.normalize(logits)
            logits = torch.sum(logits, dim=-1)
            # print(logits.shape)
            logits = F.normalize(logits)
            buffer.append(logits.detach().cpu().numpy())

        with torch.no_grad():
            for data, suffix in tqdm(self.data_loader):
                data = data[0] # batch_size == 1, current shape (frames, c, h, w)
                suffix = self.feat + '/' + suffix[0]

                if data.shape[0] == 0 or os.path.exists(path + 'features/' + suffix):
                    continue
                if data.shape[0] >= 600:
                    datas = torch.split(data, data.shape[0]//2, dim=0)
                else:
                    datas = [data]   
                
                features = []

                for data in datas:
                    buffer = []
                    handles = []
                    data = data.cuda()
                    for layer in self.extraction_layers:
                        if self.feat == 'imac':
                            h = layer.register_forward_hook(imac)
                        else:
                            h = layer.register_forward_hook(rmac)
                        handles.append(h)
                    h_x = self.model(data)

                    for h in handles:
                        h.remove()
                    del h_x
                    del data
                    features.append(np.concatenate(buffer, axis=1))
                # breakpoint()
                os.makedirs(path + 'features/' + '/'.join(suffix.split('/')[:2]), exist_ok=True)
                np.save(path + 'features/' + suffix, np.squeeze(np.concatenate(features, axis=0)))

    def load_pretrained(self, model, pretrained_model, logger=print):
        ckpt = torch.load(pretrained_model, map_location='cpu')
        if len(ckpt) == 3:  # moco initialization
            ckpt = {k[17:]: v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder_q')}

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
            return model, model_layers
        elif model_name == 'resnet-50-mocov2':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))
            self.load_pretrained(model, pretrained_model='/mldisk/nfs_shared_/dh/weights/mocov2/moco_v2_200ep_pretrain.pth.tar')

            return model, model_layers
        elif model_name == 'resnet-50-byol':
            model = models.resnet50(pretrained=True)
            model_layers = []
            for layer in layers:
                model_layers.append(model._modules.get(layer))

            self.load_pretrained(model, pretrained_model='/mldisk/nfs_shared_/dh/weights/byol/byol-pretrain_res50x1.pth.tar')

            return model, model_layers
        else:
            raise KeyError('Model %s was not found' % model_name)

if __name__ == "__main__":
    v2v = Video2Vec(dataset='vcdb', model='resnet-50-byol', num_workers=12,feat='rmac')
    v2v.get_vec(path='/mldisk/nfs_shared_/dh/datasets/vcdb-byol/')