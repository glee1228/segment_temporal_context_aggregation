import os
import pytorchvideo
import torch
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
from tqdm import tqdm
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths

def load_video_paths_fivr(root='/workspace/datasets/fivr/'):
    # breakpoint()
    paths = sorted(glob.glob(root + 'core/*.*'))
    vid2paths_core = {}
    for path in paths:
        vid2paths_core[path.split('/')[-1].split('.')[0]] = path

    paths = sorted(glob.glob(root + 'distraction/*/*.*'))
    # paths = []
    vid2paths_bg = {}
    for path in paths:
        vid2paths_bg[path.split('/')[-1].split('.')[0]] = path
    return vid2paths_core, vid2paths_bg


def ava_inference_transform(
        clip,
        boxes,
        num_frames=32,  # if using slowfast_r50_detection, change this to 32
        crop_size=256,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4
):
    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2], clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, core_video_paths, bg_video_paths,transform,path):
        super().__init__()
        self.dataset = dataset
        self.core_video_paths = core_video_paths
        self.bg_video_paths = bg_video_paths
        self.video_paths = dict(self.core_video_paths,**self.bg_video_paths)
        self.video_ids = list(self.video_paths.keys())
        self.transform = transform
        self.feature_path = path
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_path = self.video_paths[video_id]
        # print(video_path)
        suffix = video_path.split('/')[-2] + '/' + video_path.split('/')[-1].split('.')[0] + '.npy'
        if os.path.exists(self.feature_path + 'features/' + suffix):
            print(f'{self.feature_path + "features/" + suffix} already exists.')
            return torch.Tensor([])
        if os.path.exists(video_path):
            # Load the video
            encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"

            if fps > 144 or fps is None:
                fps = 25

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = int(frame_count / fps)

            # print('Completed loading encoded video.')

            # Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
            time_stamp_range = range(0, duration)  # time stamps in video for which clip is sampled.
            clip_duration = 1.0  # Duration of clip used for each inference step.


            frames = None
            width = 0
            height = 0
            # bar = tqdm(time_stamp_range, ncols=150)
            for i,time_stamp in enumerate(range(0,3)):
                # print("Generating predictions for {} time stamp: {} sec".format(index,time_stamp))
                # breakpoint()
                # Generate clip around the designated time stamps
                if i == 0:
                    inp_imgs = encoded_vid.get_clip(
                        time_stamp_range[0] - clip_duration / 2.0,  # -0.5
                        time_stamp_range[0] + clip_duration / 2.0  #  0.5
                    )
                    # print(time_stamp_range[0] - clip_duration / 2.0,' ',time_stamp_range[0] + clip_duration / 2.0)
                elif i == 1:
                    inp_imgs = encoded_vid.get_clip(
                        time_stamp_range[1] - clip_duration / 2.0 ,  # 0.5 -> 1.5 -> 2.5
                        time_stamp_range[-2] + clip_duration / 2.0  # 1.5 -> 2.5 -> 3.5
                    )
                    # print(time_stamp_range[1] - clip_duration / 2.0, ' ', time_stamp_range[-2] + clip_duration / 2.0)
                elif i ==2:
                    inp_imgs = encoded_vid.get_clip(
                        time_stamp_range[-1] - clip_duration / 2.0,  # 86.5
                        time_stamp_range[-1] + clip_duration / 2.0  # 87.5
                    )
                    # print(time_stamp_range[-1] - clip_duration / 2.0, ' ', time_stamp_range[-1] + clip_duration / 2.0 )
                inp_imgs = inp_imgs['video']

                sample_dict = {
                    "video": inp_imgs,
                }
                if self.transform is not None:
                    if i == 0 or i == 2:
                        sample_dict['video'] = UniformTemporalSubsample(8)(sample_dict['video'])
                        sample_dict = self.transform(sample_dict)
                    elif i == 1:
                        sample_num = int(time_stamp_range[-2]) * 8
                        sample_dict['video'] = UniformTemporalSubsample(sample_num)(sample_dict['video'])
                        sample_dict = self.transform(sample_dict)
                if sample_dict is None:
                    continue

                if i ==0:
                    frames = sample_dict['video']
                    # print(frames.shape)
                    _, subsample, width, height = frames.shape
                    del inp_imgs
                    del sample_dict
                else:
                    frames = np.concatenate((frames,sample_dict['video']),axis=1)
                # print(f'duration : {duration}, fps : {fps}, frame count : {frame_count}')
            #     bar.set_description(f'duration [{i}/{duration}] fps : {fps}, frame count : {frame_count} ')
            #     bar.update()
            #
            # bar.close()
            # breakpoint()

            data_dict = {
                "video": frames,
                "video_path" : video_path,
                "video_name": video_id,
                "duration": duration,
                "frame_count": frame_count,
                "fps": fps,
                "width" : width,
                "height" : height,
                "subsample" : subsample,
            }

            # for i,time_stamp in enumerate(time_stamp_range):
            #     print("Generating predictions for {} time stamp: {} sec".format(index,time_stamp))
            #     # breakpoint()
            #     # Generate clip around the designated time stamps
            #     inp_imgs = encoded_vid.get_clip(
            #         time_stamp - clip_duration / 2.0,  # start second
            #         time_stamp + clip_duration / 2.0  # end second
            #     )
            #
            #     inp_imgs = inp_imgs['video']
            #     sample_dict = {
            #         "video": inp_imgs,
            #     }
            #     if self.transform is not None:
            #         sample_dict = self.transform(sample_dict)
            #
            #     if sample_dict is None:
            #         continue
            #
            #     if i ==0:
            #         frames = sample_dict['video']
            #         # print(frames.shape)
            #         _, subsample, width, height = frames.shape
            #     else:
            #         frames = np.concatenate((frames,sample_dict['video']),axis=1)
            #     # print(f'duration : {duration}, fps : {fps}, frame count : {frame_count}')
            # #     bar.set_description(f'duration [{i}/{duration}] fps : {fps}, frame count : {frame_count} ')
            # #     bar.update()
            # #
            # # bar.close()
            # data_dict = {
            #     "video": frames,
            #     "video_path" : video_path,
            #     "video_name": video_id,
            #     "duration": duration,
            #     "frame_count": frame_count,
            #     "fps": fps,
            #     "width" : width,
            #     "height" : height,
            #     "subsample" : subsample,
            # }
            #
            #
            # breakpoint()

            # print(frames.shape)
        return data_dict

    def __len__(self):
        return len(self.video_ids)

if __name__ == '__main__':
    vid2paths_core, vid2paths_bg = load_video_paths_fivr('/mldisk/nfs_shared_/MLVD/FIVR/videos/')

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(4),
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

    train_dataloader = DataLoader(VideoDataset(dataset='fivr',core_video_paths=vid2paths_core,bg_video_paths=vid2paths_bg,transform=train_transform))
    with torch.no_grad():
        for step, (datum) in enumerate(train_dataloader):
            breakpoint()