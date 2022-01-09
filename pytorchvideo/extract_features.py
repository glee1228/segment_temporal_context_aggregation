import os
import pytorchvideo
import torch
from torch import nn
import glob
import numpy as np

import cv2
from torch.utils.data import DataLoader, Dataset
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

from data import load_video_paths_fivr,VideoDataset
from models import c2d_r50, i3d_r50, slow_r50, slow_r50_detection,slowfast_r101,slowfast_r50_detection

from visualization import VideoVisualizer
import json
import itertools
import torch.nn.functional as F
from tqdm import tqdm


def extract_features(path):
    device = 'cuda'  # or 'cpu'

    model_name = 'slow_r50'
    video_model = globals()[model_name](True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        video_model = nn.DataParallel(video_model)
    video_model = video_model.eval().to(device)

    # breakpoint()
    if model_name == 'slow_r50':
        if torch.cuda.device_count() > 1:
            video_model.module.blocks[5].dropout = torch.nn.Identity()
            video_model.module.blocks[5].proj = torch.nn.Identity()
            # video_model.module.blocks[5].output_pool = torch.nn.Identity()
        else:
            video_model.blocks[5].dropout = torch.nn.Identity()
            video_model.blocks[5].proj = torch.nn.Identity()
            # video_model.blocks[5].output_pool = torch.nn.Identity()
    vid2paths_core, vid2paths_bg = load_video_paths_fivr('/workspace/TCA/datasets/FIVR/videos/')
    # breakpoint()
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        # UniformTemporalSubsample(8),
                        Lambda(lambda x: x / 255.0),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                        RandomShortSideScale(min_size=256, max_size=256),
                        # RandomCrop(244),
                        # RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )


    test_dataloader = DataLoader(VideoDataset(dataset='fivr',core_video_paths=vid2paths_core,bg_video_paths=vid2paths_bg,transform=train_transform,path=path),num_workers=2)
    bar = tqdm(test_dataloader, ncols=150)
    with torch.no_grad():
        for step, (datum) in enumerate(test_dataloader):
            if not torch.is_tensor(datum):
                input = datum['video']
                duration = datum['duration']
                input = torch.reshape(input,(duration,3,datum['subsample'],datum['width'],datum['height']))

                suffix = datum['video_path'][0].split('/')[-2] + '/' + datum['video_path'][0].split('/')[-1].split('.')[0]+'.npy'

                # print(input.shape)
                if input.shape[0] >= 170:
                    inputs = torch.split(input, input.shape[0] // 2, dim=0)
                elif input.shape[0] >= 340:
                    inputs = torch.split(input, input.shape[0] // 3, dim=0)
                else:
                    inputs = [input]

                features = []
                for input in inputs:
                    logits = video_model(input.cuda())
                    logits = F.normalize(logits)
                    logits = logits.detach().cpu().numpy()
                    features.append(logits)
                    del input
                    del logits

                # breakpoint()
                os.makedirs(path + 'features/' + '/'.join(suffix.split('/')[:-1]), exist_ok=True)
                np.save(path + 'features/' + suffix, np.squeeze(np.concatenate(features, axis=0)))
                # print(path + 'features/' + suffix,end='')

                bar.set_description(f'[{step}/{len(test_dataloader)}]{path + "features/" + suffix} duration : {duration}')
                bar.update()

        bar.close()




if __name__ == '__main__':
    extract_features(path='/workspace/TCA/pre_processing/fivr-slowr50')