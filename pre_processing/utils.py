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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Note: This file has been barrowed from facebookresearch/slowfast repo.
# TODO: Migrate this into the core PyTorchVideo libarary.

# from __future__ import annotations

import itertools
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
# from detectron2.utils.visualizer import Visualizer


logger = logging.getLogger(__name__)


def _create_text_labels(
    classes: List[int],
    scores: List[float],
    class_names: List[str],
    ground_truth: bool = False,
) -> List[str]:
    """
    Create text labels.
    Args:
        classes (list[int]): a list of class ids for each example.
        scores (list[float] or None): list of scores for each example.
        class_names (list[str]): a list of class names, ordered by their ids.
        ground_truth (bool): whether the labels are ground truth.
    Returns:
        labels (list[str]): formatted text labels.
    """
    try:
        labels = [class_names.get(c, "n/a") for c in classes]
    except IndexError:
        logger.error("Class indices get out of range: {}".format(classes))
        return None

    if ground_truth:
        labels = ["[{}] {}".format("GT", label) for label in labels]
    elif scores is not None:
        assert len(classes) == len(scores)
        labels = ["[{:.2f}] {}".format(s, label) for s, label in zip(scores, labels)]
    return labels

#
# class ImgVisualizer(Visualizer):
#     def __init__(
#         self, img_rgb: torch.Tensor, meta: Optional[SimpleNamespace] = None, **kwargs
#     ) -> None:
#         """
#         See https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/visualizer.py
#         for more details.
#         Args:
#             img_rgb: a tensor or numpy array of shape (H, W, C), where H and W correspond to
#                 the height and width of the image respectively. C is the number of
#                 color channels. The image is required to be in RGB format since that
#                 is a requirement of the Matplotlib library. The image is also expected
#                 to be in the range [0, 255].
#             meta (MetadataCatalog): image metadata.
#                 See https://github.com/facebookresearch/detectron2/blob/81d5a87763bfc71a492b5be89b74179bd7492f6b/detectron2/data/catalog.py#L90
#         """
#         super(ImgVisualizer, self).__init__(img_rgb, meta, **kwargs)
#
#     def draw_text(
#         self,
#         text: str,
#         position: List[int],
#         *,
#         font_size: Optional[int] = None,
#         color: str = "w",
#         horizontal_alignment: str = "center",
#         vertical_alignment: str = "bottom",
#         box_facecolor: str = "black",
#         alpha: float = 0.5,
#     ) -> None:
#         """
#         Draw text at the specified position.
#         Args:
#             text (str): the text to draw on image.
#             position (list of 2 ints): the x,y coordinate to place the text.
#             font_size (Optional[int]): font of the text. If not provided, a font size
#                 proportional to the image width is calculated and used.
#             color (str): color of the text. Refer to `matplotlib.colors` for full list
#                 of formats that are accepted.
#             horizontal_alignment (str): see `matplotlib.text.Text`.
#             vertical_alignment (str): see `matplotlib.text.Text`.
#             box_facecolor (str): color of the box wrapped around the text. Refer to
#                 `matplotlib.colors` for full list of formats that are accepted.
#             alpha (float): transparency level of the box.
#         """
#         if not font_size:
#             font_size = self._default_font_size
#         x, y = position
#         self.output.ax.text(
#             x,
#             y,
#             text,
#             size=font_size * self.output.scale,
#             family="monospace",
#             bbox={
#                 "facecolor": box_facecolor,
#                 "alpha": alpha,
#                 "pad": 0.7,
#                 "edgecolor": "none",
#             },
#             verticalalignment=vertical_alignment,
#             horizontalalignment=horizontal_alignment,
#             color=color,
#             zorder=10,
#         )
#
#     def draw_multiple_text(
#         self,
#         text_ls: List[str],
#         box_coordinate: torch.Tensor,
#         *,
#         top_corner: bool = True,
#         font_size: Optional[int] = None,
#         color: str = "w",
#         box_facecolors: str = "black",
#         alpha: float = 0.5,
#     ) -> None:
#         """
#         Draw a list of text labels for some bounding box on the image.
#         Args:
#             text_ls (list of strings): a list of text labels.
#             box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
#                 coordinates of the box.
#             top_corner (bool): If True, draw the text labels at (x_left, y_top) of the box.
#                 Else, draw labels at (x_left, y_bottom).
#             font_size (Optional[int]): font of the text. If not provided, a font size
#                 proportional to the image width is calculated and used.
#             color (str): color of the text. Refer to `matplotlib.colors` for full list
#                 of formats that are accepted.
#             box_facecolors (str): colors of the box wrapped around the text. Refer to
#                 `matplotlib.colors` for full list of formats that are accepted.
#             alpha (float): transparency level of the box.
#         """
#         if not isinstance(box_facecolors, list):
#             box_facecolors = [box_facecolors] * len(text_ls)
#         assert len(box_facecolors) == len(
#             text_ls
#         ), "Number of colors provided is not equal to the number of text labels."
#         if not font_size:
#             font_size = self._default_font_size
#         text_box_width = font_size + font_size // 2
#         # If the texts does not fit in the assigned location,
#         # we split the text and draw it in another place.
#         if top_corner:
#             num_text_split = self._align_y_top(
#                 box_coordinate, len(text_ls), text_box_width
#             )
#             y_corner = 1
#         else:
#             num_text_split = len(text_ls) - self._align_y_bottom(
#                 box_coordinate, len(text_ls), text_box_width
#             )
#             y_corner = 3
#
#         text_color_sorted = sorted(
#             zip(text_ls, box_facecolors), key=lambda x: x[0], reverse=True
#         )
#         if len(text_color_sorted) != 0:
#             text_ls, box_facecolors = zip(*text_color_sorted)
#         else:
#             text_ls, box_facecolors = [], []
#         text_ls, box_facecolors = list(text_ls), list(box_facecolors)
#         self.draw_multiple_text_upward(
#             text_ls[:num_text_split][::-1],
#             box_coordinate,
#             y_corner=y_corner,
#             font_size=font_size,
#             color=color,
#             box_facecolors=box_facecolors[:num_text_split][::-1],
#             alpha=alpha,
#         )
#         self.draw_multiple_text_downward(
#             text_ls[num_text_split:],
#             box_coordinate,
#             y_corner=y_corner,
#             font_size=font_size,
#             color=color,
#             box_facecolors=box_facecolors[num_text_split:],
#             alpha=alpha,
#         )
#
#     def draw_multiple_text_upward(
#         self,
#         text_ls: List[str],
#         box_coordinate: torch.Tensor,
#         *,
#         y_corner: int = 1,
#         font_size: Optional[int] = None,
#         color: str = "w",
#         box_facecolors: str = "black",
#         alpha: float = 0.5,
#     ) -> None:
#         """
#         Draw a list of text labels for some bounding box on the image in upward direction.
#         The next text label will be on top of the previous one.
#         Args:
#             text_ls (list of strings): a list of text labels.
#             box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
#                 coordinates of the box.
#             y_corner (int): Value of either 1 or 3. Indicate the index of the y-coordinate of
#                 the box to draw labels around.
#             font_size (Optional[int]): font of the text. If not provided, a font size
#                 proportional to the image width is calculated and used.
#             color (str): color of the text. Refer to `matplotlib.colors` for full list
#                 of formats that are accepted.
#             box_facecolors (str or list of strs): colors of the box wrapped around the
#                 text. Refer to `matplotlib.colors` for full list of formats that
#                 are accepted.
#             alpha (float): transparency level of the box.
#         """
#         if not isinstance(box_facecolors, list):
#             box_facecolors = [box_facecolors] * len(text_ls)
#         assert len(box_facecolors) == len(
#             text_ls
#         ), "Number of colors provided is not equal to the number of text labels."
#
#         assert y_corner in [1, 3], "Y_corner must be either 1 or 3"
#         if not font_size:
#             font_size = self._default_font_size
#
#         x, horizontal_alignment = self._align_x_coordinate(box_coordinate)
#         y = box_coordinate[y_corner].item()
#         for i, text in enumerate(text_ls):
#             self.draw_text(
#                 text,
#                 (x, y),
#                 font_size=font_size,
#                 color=color,
#                 horizontal_alignment=horizontal_alignment,
#                 vertical_alignment="bottom",
#                 box_facecolor=box_facecolors[i],
#                 alpha=alpha,
#             )
#             y -= font_size + font_size // 2
#
#     def draw_multiple_text_downward(
#         self,
#         text_ls: List[str],
#         box_coordinate: torch.Tensor,
#         *,
#         y_corner: int = 1,
#         font_size: Optional[int] = None,
#         color: str = "w",
#         box_facecolors: str = "black",
#         alpha: float = 0.5,
#     ) -> None:
#         """
#         Draw a list of text labels for some bounding box on the image in downward direction.
#         The next text label will be below the previous one.
#         Args:
#             text_ls (list of strings): a list of text labels.
#             box_coordinate (tensor): shape (4,). The (x_left, y_top, x_right, y_bottom)
#                 coordinates of the box.
#             y_corner (int): Value of either 1 or 3. Indicate the index of the y-coordinate of
#                 the box to draw labels around.
#             font_size (Optional[int]): font of the text. If not provided, a font size
#                 proportional to the image width is calculated and used.
#             color (str): color of the text. Refer to `matplotlib.colors` for full list
#                 of formats that are accepted.
#             box_facecolors (str): colors of the box wrapped around the text. Refer to
#                 `matplotlib.colors` for full list of formats that are accepted.
#             alpha (float): transparency level of the box.
#         """
#         if not isinstance(box_facecolors, list):
#             box_facecolors = [box_facecolors] * len(text_ls)
#         assert len(box_facecolors) == len(
#             text_ls
#         ), "Number of colors provided is not equal to the number of text labels."
#
#         assert y_corner in [1, 3], "Y_corner must be either 1 or 3"
#         if not font_size:
#             font_size = self._default_font_size
#
#         x, horizontal_alignment = self._align_x_coordinate(box_coordinate)
#         y = box_coordinate[y_corner].item()
#         for i, text in enumerate(text_ls):
#             self.draw_text(
#                 text,
#                 (x, y),
#                 font_size=font_size,
#                 color=color,
#                 horizontal_alignment=horizontal_alignment,
#                 vertical_alignment="top",
#                 box_facecolor=box_facecolors[i],
#                 alpha=alpha,
#             )
#             y += font_size + font_size // 2
#
#     def _align_x_coordinate(self, box_coordinate: torch.Tensor) -> Tuple[float, str]:
#         """
#         Choose an x-coordinate from the box to make sure the text label
#         does not go out of frames. By default, the left x-coordinate is
#         chosen and text is aligned left. If the box is too close to the
#         right side of the image, then the right x-coordinate is chosen
#         instead and the text is aligned right.
#         Args:
#             box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
#             coordinates of the box.
#         Returns:
#             x_coordinate (float): the chosen x-coordinate.
#             alignment (str): whether to align left or right.
#         """
#         # If the x-coordinate is greater than 5/6 of the image width,
#         # then we align test to the right of the box. This is
#         # chosen by heuristics.
#         if box_coordinate[0] > (self.output.width * 5) // 6:
#             return box_coordinate[2], "right"
#
#         return box_coordinate[0], "left"
#
#     def _align_y_top(
#         self, box_coordinate: torch.Tensor, num_text: int, textbox_width: float
#     ) -> int:
#         """
#         Calculate the number of text labels to plot on top of the box
#         without going out of frames.
#         Args:
#             box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
#             coordinates of the box.
#             num_text (int): the number of text labels to plot.
#             textbox_width (float): the width of the box wrapped around text label.
#         """
#         dist_to_top = box_coordinate[1]
#         num_text_top = dist_to_top // textbox_width
#
#         if isinstance(num_text_top, torch.Tensor):
#             num_text_top = int(num_text_top.item())
#
#         return min(num_text, num_text_top)
#
#     def _align_y_bottom(
#         self, box_coordinate: torch.Tensor, num_text: int, textbox_width: float
#     ) -> int:
#         """
#         Calculate the number of text labels to plot at the bottom of the box
#         without going out of frames.
#         Args:
#             box_coordinate (array-like): shape (4,). The (x_left, y_top, x_right, y_bottom)
#             coordinates of the box.
#             num_text (int): the number of text labels to plot.
#             textbox_width (float): the width of the box wrapped around text label.
#         """
#         dist_to_bottom = self.output.height - box_coordinate[3]
#         num_text_bottom = dist_to_bottom // textbox_width
#
#         if isinstance(num_text_bottom, torch.Tensor):
#             num_text_bottom = int(num_text_bottom.item())
#
#         return min(num_text, num_text_bottom)
#

class VideoVisualizer:
    def __init__(
        self,
        num_classes: int,
        class_names: Dict,
        top_k: int = 1,
        colormap: str = "rainbow",
        thres: float = 0.7,
        lower_thres: float = 0.3,
        common_class_names: Optional[List[str]] = None,
        mode: str = "top-k",
    ) -> None:
        """
        Args:
            num_classes (int): total number of classes.
            class_names (dict): Dict mapping classID to name.
            top_k (int): number of top predicted classes to plot.
            colormap (str): the colormap to choose color for class labels from.
                See https://matplotlib.org/tutorials/colors/colormaps.html
            thres (float): threshold for picking predicted classes to visualize.
            lower_thres (Optional[float]): If `common_class_names` if given,
                this `lower_thres` will be applied to uncommon classes and
                `thres` will be applied to classes in `common_class_names`.
            common_class_names (Optional[list of str]): list of common class names
                to apply `thres`. Class names not included in `common_class_names` will
                have `lower_thres` as a threshold. If None, all classes will have
                `thres` as a threshold. This is helpful for model trained on
                highly imbalanced dataset.
            mode (str): Supported modes are {"top-k", "thres"}.
                This is used for choosing predictions for visualization.

        """
        assert mode in ["top-k", "thres"], "Mode {} is not supported.".format(mode)
        self.mode = mode
        self.num_classes = num_classes
        self.class_names = class_names
        self.top_k = top_k
        self.thres = thres
        self.lower_thres = lower_thres

        if mode == "thres":
            self._get_thres_array(common_class_names=common_class_names)

        self.color_map = plt.get_cmap(colormap)

    def _get_color(self, class_id: int) -> List[float]:
        """
        Get color for a class id.
        Args:
            class_id (int): class id.
        """
        return self.color_map(class_id / self.num_classes)[:3]

    def draw_one_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray],
        preds: Union[torch.Tensor, List[float]],
        bboxes: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        text_alpha: float = 0.7,
        ground_truth: bool = False,
    ) -> np.ndarray:
        """
        Draw labels and bouding boxes for one image. By default, predicted
        labels are drawn in the top left corner of the image or corresponding
        bounding boxes. For ground truth labels (setting True for ground_truth flag),
        labels will be drawn in the bottom left corner.
        Args:
            frame (array-like): a tensor or numpy array of shape (H, W, C),
            where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            preds (tensor or list): If ground_truth is False, provide a float tensor of
                shape (num_boxes, num_classes) that contains all of the confidence
                scores of the model. For recognition task, input shape can be (num_classes,).
                To plot true label (ground_truth is True), preds is a list contains int32
                of the shape (num_boxes, true_class_ids) or (true_class_ids,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates
                of the bounding boxes.
            alpha (Optional[float]): transparency level of the bounding boxes.
            text_alpha (Optional[float]): transparency level of the box wrapped around
                text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
        Returns:
            An image with bounding box annotations and corresponding bbox
            labels plotted on it.
        """
        if isinstance(preds, torch.Tensor):
            if preds.ndim == 1:
                preds = preds.unsqueeze(0)
            n_instances = preds.shape[0]
        elif isinstance(preds, list):
            n_instances = len(preds)
        else:
            logger.error("Unsupported type of prediction input.")
            return

        if ground_truth:
            top_scores, top_classes = [None] * n_instances, preds

        elif self.mode == "top-k":
            top_scores, top_classes = torch.topk(preds, k=self.top_k)
            top_scores, top_classes = top_scores.tolist(), top_classes.tolist()
        elif self.mode == "thres":
            top_scores, top_classes = [], []
            for pred in preds:
                mask = pred >= self.thres
                top_scores.append(pred[mask].tolist())
                top_class = torch.squeeze(torch.nonzero(mask), dim=-1).tolist()
                top_classes.append(top_class)

        # Create labels top k predicted classes with their scores.
        text_labels = []
        for i in range(n_instances):
            text_labels.append(
                _create_text_labels(
                    top_classes[i],
                    top_scores[i],
                    self.class_names,
                    ground_truth=ground_truth,
                )
            )
        frame_visualizer = ImgVisualizer(frame, meta=None)
        font_size = min(max(np.sqrt(frame.shape[0] * frame.shape[1]) // 25, 5), 9)
        top_corner = not ground_truth
        if bboxes is not None:
            assert len(preds) == len(
                bboxes
            ), "Encounter {} predictions and {} bounding boxes".format(
                len(preds), len(bboxes)
            )
            for i, box in enumerate(bboxes):
                text = text_labels[i]
                pred_class = top_classes[i]
                colors = [self._get_color(pred) for pred in pred_class]

                box_color = "r" if ground_truth else "g"
                line_style = "--" if ground_truth else "-."
                frame_visualizer.draw_box(
                    box,
                    alpha=alpha,
                    edge_color=box_color,
                    line_style=line_style,
                )
                frame_visualizer.draw_multiple_text(
                    text,
                    box,
                    top_corner=top_corner,
                    font_size=font_size,
                    box_facecolors=colors,
                    alpha=text_alpha,
                )
        else:
            text = text_labels[0]
            pred_class = top_classes[0]
            colors = [self._get_color(pred) for pred in pred_class]
            frame_visualizer.draw_multiple_text(
                text,
                torch.Tensor([0, 5, frame.shape[1], frame.shape[0] - 5]),
                top_corner=top_corner,
                font_size=font_size,
                box_facecolors=colors,
                alpha=text_alpha,
            )

        return frame_visualizer.output.get_image()

    def draw_clip_range(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        preds: Union[torch.Tensor, List[float]],
        bboxes: Optional[torch.Tensor] = None,
        text_alpha: float = 0.5,
        ground_truth: bool = False,
        keyframe_idx: Optional[int] = None,
        draw_range: Optional[List[int]] = None,
        repeat_frame: int = 1,
    ) -> List[np.ndarray]:
        """
        Draw predicted labels or ground truth classes to clip.
        Draw bouding boxes to clip if bboxes is provided. Boxes will gradually
        fade in and out the clip, centered around the clip's central frame,
        within the provided `draw_range`.
        Args:
            frames (array-like): video data in the shape (T, H, W, C).
            preds (tensor): a tensor of shape (num_boxes, num_classes) that
                contains all of the confidence scores of the model. For recognition
                task or for ground_truth labels, input shape can be (num_classes,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates
                of the bounding boxes.
            text_alpha (float): transparency label of the box wrapped around text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
            keyframe_idx (int): the index of keyframe in the clip.
            draw_range (Optional[list[ints]): only draw frames in range
                [start_idx, end_idx] inclusively in the clip. If None, draw on
                the entire clip.
            repeat_frame (int): repeat each frame in draw_range for `repeat_frame`
                time for slow-motion effect.
        Returns:
            A list of frames with bounding box annotations and corresponding
            bbox labels ploted on them.
        """
        if draw_range is None:
            draw_range = [0, len(frames) - 1]
        if draw_range is not None:
            draw_range[0] = max(0, draw_range[0])
            left_frames = frames[: draw_range[0]]
            right_frames = frames[draw_range[1] + 1 :]

        draw_frames = frames[draw_range[0] : draw_range[1] + 1]
        if keyframe_idx is None:
            keyframe_idx = len(frames) // 2

        img_ls = (
            list(left_frames)
            + self.draw_clip(
                draw_frames,
                preds,
                bboxes=bboxes,
                text_alpha=text_alpha,
                ground_truth=ground_truth,
                keyframe_idx=keyframe_idx - draw_range[0],
                repeat_frame=repeat_frame,
            )
            + list(right_frames)
        )

        return img_ls

    def draw_clip(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        preds: Union[torch.Tensor, List[float]],
        bboxes: Optional[torch.Tensor] = None,
        text_alpha: float = 0.5,
        ground_truth: bool = False,
        keyframe_idx: Optional[int] = None,
        repeat_frame: int = 1,
    ) -> List[np.ndarray]:
        """
        Draw predicted labels or ground truth classes to clip. Draw bouding boxes to clip
        if bboxes is provided. Boxes will gradually fade in and out the clip, centered
        around the clip's central frame.
        Args:
            frames (array-like): video data in the shape (T, H, W, C).
            preds (tensor): a tensor of shape (num_boxes, num_classes) that contains
                all of the confidence scores of the model. For recognition task or for
                ground_truth labels, input shape can be (num_classes,).
            bboxes (Optional[tensor]): shape (num_boxes, 4) that contains the coordinates
                of the bounding boxes.
            text_alpha (float): transparency label of the box wrapped around text labels.
            ground_truth (bool): whether the prodived bounding boxes are ground-truth.
            keyframe_idx (int): the index of keyframe in the clip.
            repeat_frame (int): repeat each frame in draw_range for `repeat_frame`
                time for slow-motion effect.
        Returns:
            A list of frames with bounding box annotations and corresponding
            bbox labels plotted on them.
        """
        assert repeat_frame >= 1, "`repeat_frame` must be a positive integer."

        repeated_seq = range(0, len(frames))
        repeated_seq = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, repeat_frame) for x in repeated_seq
            )
        )

        frames, adjusted = self._adjust_frames_type(frames)
        if keyframe_idx is None:
            half_left = len(repeated_seq) // 2
            half_right = (len(repeated_seq) + 1) // 2
        else:
            mid = int((keyframe_idx / len(frames)) * len(repeated_seq))
            half_left = mid
            half_right = len(repeated_seq) - mid

        alpha_ls = np.concatenate(
            [
                np.linspace(0, 1, num=half_left),
                np.linspace(1, 0, num=half_right),
            ]
        )
        text_alpha = text_alpha
        frames = frames[repeated_seq]
        img_ls = []
        for alpha, frame in zip(alpha_ls, frames):
            draw_img = self.draw_one_frame(
                frame,
                preds,
                bboxes,
                alpha=alpha,
                text_alpha=text_alpha,
                ground_truth=ground_truth,
            )
            if adjusted:
                draw_img = draw_img.astype("float32") / 255

            img_ls.append(draw_img)

        return img_ls

    def _adjust_frames_type(
        self, frames: torch.Tensor
    ) -> Tuple[List[np.ndarray], bool]:
        """
        Modify video data to have dtype of uint8 and values range in [0, 255].
        Args:
            frames (array-like): 4D array of shape (T, H, W, C).
        Returns:
            frames (list of frames): list of frames in range [0, 1].
            adjusted (bool): whether the original frames need adjusted.
        """
        assert (
            frames is not None and len(frames) != 0
        ), "Frames does not contain any values"
        frames = np.array(frames)
        assert np.array(frames).ndim == 4, "Frames must have 4 dimensions"
        adjusted = False
        if frames.dtype in [np.float32, np.float64]:
            frames *= 255
            frames = frames.astype(np.uint8)
            adjusted = True

        return frames, adjusted

    def _get_thres_array(self, common_class_names: Optional[List[str]] = None) -> None:
        """
        Compute a thresholds array for all classes based on `self.thes` and `self.lower_thres`.
        Args:
            common_class_names (Optional[list of str]): a list of common class names.
        """
        common_class_ids = []
        if common_class_names is not None:
            common_classes = set(common_class_names)

            for key, name in self.class_names.items():
                if name in common_classes:
                    common_class_ids.append(key)
        else:
            common_class_ids = list(range(self.num_classes))

        thres_array = np.full(shape=(self.num_classes,), fill_value=self.lower_thres)
        thres_array[common_class_ids] = self.thres
        self.thres = torch.from_numpy(thres_array)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Any, Callable

import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet, create_resnet_with_roi_head
from torch.hub import load_state_dict_from_url
from pytorchvideo.models.slowfast import create_slowfast, create_slowfast_with_roi_head

"""
ResNet style models for video recognition.
"""

root_dir = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo"
checkpoint_paths = {
    "slow_r50": f"{root_dir}/kinetics/SLOW_8x8_R50.pyth",
    "slow_r50_4x16": f"{root_dir}/kinetics/SLOW_4x16_R50.pyth",
    "slow_r50_detection": f"{root_dir}/ava/SLOW_4x16_R50_DETECTION.pyth",
    "c2d_r50": f"{root_dir}/kinetics/C2D_8x8_R50.pyth",
    "i3d_r50": f"{root_dir}/kinetics/I3D_8x8_R50.pyth",
    "slowfast_r50": f"{root_dir}/kinetics/SLOWFAST_8x8_R50.pyth",
    "slowfast_r50_detection": f"{root_dir}/ava/SLOWFAST_8x8_R50_DETECTION.pyth",
    "slowfast_r101": f"{root_dir}/kinetics/SLOWFAST_8x8_R101.pyth",
    "slowfast_16x8_r101_50_50": f"{root_dir}/kinetics/SLOWFAST_16x8_R101_50_50.pyth",
}


def _resnet(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: str = "",
    model_builder: Callable = create_resnet,
    **kwargs: Any,
) -> nn.Module:
    model = model_builder(**kwargs)
    if pretrained:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model

def slow_r50_4x16(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Slow R50 model architecture [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 74.58.

    [1] "SlowFast Networks for Video Recognition"
        Christoph Feichtenhofer et al
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slow_r50_4x16"],
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(4, 7, 7),
        model_depth=50,
        **kwargs,
    )

def slow_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Slow R50 model architecture [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 74.58.

    [1] "SlowFast Networks for Video Recognition"
        Christoph Feichtenhofer et al
        https://arxiv.org/pdf/1812.03982.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slow_r50"],
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(8, 7, 7),
        model_depth=50,
        **kwargs,
    )


def slow_r50_detection(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    Slow R50 model architecture [1] with pretrained weights based on 4x16 setting.
    The model is initially trained on Kinetics dataset for classification and later
    finetuned on AVA dataset for detection.

    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slow_r50_detection"],
        model_builder=create_resnet_with_roi_head,
        **kwargs,
    )


def c2d_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    C2D R50 model architecture with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 71.46.

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["c2d_r50"],
        stem_conv_kernel_size=(1, 7, 7),
        stage1_pool=nn.MaxPool3d,
        stage_conv_a_kernel_size=(
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ),
        **kwargs,
    )


def i3d_r50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> nn.Module:
    r"""
    I3D R50 model architecture from [1] with pretrained weights based on 8x8 setting
    on the Kinetics dataset. Model with pretrained weights has top1 accuracy of 73.27.

    [1] "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/abs/1705.07750

    Args:
        pretrained (bool): If True, returns a model pre-trained on the Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/resnet.py

    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _resnet(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["i3d_r50"],
        stem_conv_kernel_size=(5, 7, 7),
        stage1_pool=nn.MaxPool3d,
        stage_conv_a_kernel_size=(
            (3, 1, 1),
            [(3, 1, 1), (1, 1, 1)],
            [(3, 1, 1), (1, 1, 1)],
            [(1, 1, 1), (3, 1, 1)],
        ),
        **kwargs,
    )




def _slowfast(
    pretrained: bool = False,
    progress: bool = True,
    checkpoint_path: str = "",
    model_builder: Callable = create_slowfast,
    **kwargs: Any,
) -> nn.Module:
    model = model_builder(**kwargs)
    if pretrained:
        # All models are loaded onto CPU by default
        checkpoint = load_state_dict_from_url(
            checkpoint_path, progress=progress, map_location="cpu"
        )
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict)
    return model


def slowfast_r50(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R50 model architecture [1] trained with an 8x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 76.4.
    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_r50"],
        model_depth=50,
        slowfast_fusion_conv_kernel_size=(7, 1, 1),
        **kwargs,
    )


def slowfast_r101(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R101 model architecture [1] trained with an 8x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 77.9.
    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_r101"],
        model_depth=101,
        slowfast_fusion_conv_kernel_size=(5, 1, 1),
        **kwargs,
    )


def slowfast_16x8_r101_50_50(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R101_50_50 model architecture [1] trained with an 16x8 setting on the
    Kinetics dataset. Model with pretrained weights has top1 accuracy of 78.7.
    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    # slowfast_r101_50_50 has 6 conv blocks with kernel=(3, 1, 1) in stage 4.
    stage_conv_a_kernel_sizes = (
        (
            (1, 1, 1),
            (1, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
        (
            (3, 1, 1),
            (3, 1, 1),
            ((3, 1, 1),) * 6 + ((1, 1, 1),) * (23 - 6),
            (3, 1, 1),
        ),
    )
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_16x8_r101_50_50"],
        model_depth=101,
        slowfast_fusion_conv_kernel_size=(5, 1, 1),
        stage_conv_a_kernel_sizes=stage_conv_a_kernel_sizes,
        head_pool_kernel_sizes=((16, 7, 7), (64, 7, 7)),
        **kwargs,
    )


def slowfast_r50_detection(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> nn.Module:
    r"""
    SlowFast R50 model architecture [1] with pretrained weights based on 8x8 setting.
    The model is initially trained on Kinetics dataset for classification and later
    finetuned on AVA dataset for detection.
    [1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
        https://arxiv.org/pdf/1812.03982.pdf
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics dataset
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: use these to modify any of the other model settings. All the
            options are defined in pytorchvideo/models/slowfast.py
    NOTE: to use the pretrained model, do not modify the model configuration
    via the kwargs. Only modify settings via kwargs to initialize a new model
    without pretrained weights.
    """
    return _slowfast(
        pretrained=pretrained,
        progress=progress,
        checkpoint_path=checkpoint_paths["slowfast_r50_detection"],
        model_builder=create_slowfast_with_roi_head,
        **kwargs,
    )