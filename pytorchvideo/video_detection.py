from functools import partial
import numpy as np

import cv2
import torch

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from models import c2d_r50, i3d_r50, slow_r50, slow_r50_detection,slowfast_r101,slowfast_r50_detection

from visualization import VideoVisualizer
import json
import itertools

device = 'cuda' # or 'cpu'
# video_model = slow_r50_detection(True,head=None) # Another option is slowfast_r50_detection
model_name = 'slowfast_r50_detection'
video_model = globals()[model_name](True)
video_model = video_model.eval().to(device)
# video_model.detection_head = torch.nn.Sequential(*[video_model.detection_head[i] for i in range(3)])
# video_model.detection_head.roi_layer = torch.nn.Identity()
# breakpoint()
if model_name =='slow_r50':
    # video_model.blocks[5].dropout = torch.nn.Identity()
    # video_model.blocks[5].proj = torch.nn.Identity()
    # video_model.blocks[5].output_pool = torch.nn.Identity()
    pass
elif model_name =='slow_r50_detection':
    video_model.detection_head.dropout = torch.nn.Identity()
    video_model.detection_head.proj = torch.nn.Identity()
    video_model.detection_head.activation = torch.nn.Identity()
    pass

# breakpoint()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes

class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


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






label_map = {}
label_num = 0
# Create an id to label name mapping
if model_name in ['slow_r50_detection','slowfast_r50_detection']:
    label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
    label_num = 81
elif model_name == 'slow_r50':
    with open("kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)
    label_num = 400
    # Create an id to label name mapping
    for k, v in kinetics_classnames.items():
        label_map[v] = str(k).replace('"', "")
# breakpoint()
# Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
video_visualizer = VideoVisualizer(label_num, label_map, top_k=3, mode="thres", thres=0.5)


import os
import pickle as pk
root = '/mldisk/nfs_shared_/MLVD/FIVR'
with open('/workspace/TCA/datasets/fivr.pickle', 'rb') as f:
    dataset = pk.load(f)
annotation = dataset['annotation']
queries = dataset['5k']['queries']
database = dataset['5k']['database']


for query in queries:
    video_path = os.path.join(root ,f'videos/core/{query}.mp4')
    print(video_path)
    if os.path.exists(video_path):
        vide_save_path = f'{query}_output_detections.mp4'
        # Load the video
        encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"

        if fps > 144 or fps is None:
            fps = 25

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps)
        print(f'duration : {duration}, fps : {fps}, frame count : {frame_count}')
        print('Completed loading encoded video.')

        # Video predictions are generated at an internal of 1 sec from 90 seconds to 100 seconds in the video.
        time_stamp_range = range(0, duration)  # time stamps in video for which clip is sampled.
        clip_duration = 1.0  # Duration of clip used for each inference step.
        gif_imgs = []

        for time_stamp in time_stamp_range:
            print("Generating predictions for time stamp: {} sec".format(time_stamp))
            # breakpoint()
            # Generate clip around the designated time stamps
            inp_imgs = encoded_vid.get_clip(
                time_stamp - clip_duration / 2.0,  # start second
                time_stamp + clip_duration / 2.0  # end second
            )
            inp_imgs = inp_imgs['video']

            # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
            # We use the the middle image in each clip to generate the bounding boxes.
            inp_img = inp_imgs[:, inp_imgs.shape[1] // 2, :, :]
            inp_img = inp_img.permute(1, 2, 0)

            # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
            predicted_boxes = get_person_bboxes(inp_img, predictor)
            if len(predicted_boxes) == 0:
                print("Skipping clip no frames detected at time stamp: ", time_stamp)
                continue

            # Preprocess clip and bounding boxes for video action recognition.
            inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())

            # Prepend data sample id for each bounding box.
            # For more details refere to the RoIAlign in Detectron2
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

            # Generate actions predictions for the bounding boxes in the clip.
            # The model here takes in the pre-processed video clip and the detected bounding boxes.
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)

            breakpoint()

            if model_name =='slow_r50' :
                preds = video_model(inputs)

            elif model_name in ['slow_r50_detection','slowfast_r50_detection']:
                preds = video_model(inputs, inp_boxes.to(device))

            elif model_name =='slowfast_r101' :
                preds = video_model(inputs)[0]

            # breakpoint()
            if 'slowfast' in model_name:
                print(
                    f'input img : {inp_imgs.shape}, input[0] shape : {inputs[0].shape}, input[1] shape : {inputs[1].shape}, pred shape : {preds.shape}, inp_boxes : {inp_boxes}')
            else:
                print(f'input img : {inp_imgs.shape}, input shape : {inputs.shape}, pred shape : {preds.shape}, inp_boxes : {inp_boxes}')
            preds = preds.to('cpu')
            if model_name in ['slow_r50_detection','slowfast_r50_detection']:
                # The model is trained on AVA and AVA labels are 1 indexed so, prepend 0 to convert to 0 index.
                preds = torch.cat([torch.zeros(preds.shape[0], 1), preds], dim=1)

                # Plot predictions on the video and save for later visualization.
                inp_imgs = inp_imgs.permute(1, 2, 3, 0)
                inp_imgs = inp_imgs / 255.0
                out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds, predicted_boxes)
                gif_imgs += out_img_pred

            elif model_name =='slow_r50':
                # Get the predicted classes
                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(preds)
                pred_classes = preds.topk(k=5).indices

                # Map the predicted classes to the label names
                pred_class_names = [label_map[int(i)] for i in pred_classes[0]]
                print("Predicted labels: %s" % ", ".join(pred_class_names))
                # breakpoint()
                # Plot predictions on the video and save for later visualization.
                inp_imgs = inp_imgs.permute(1, 2, 3, 0)
                inp_imgs = inp_imgs / 255.0
                out_img_pred = video_visualizer.draw_clip_range(inp_imgs, preds,torch.tensor([[0.0,360.0,0.0, 288.0]]))
                gif_imgs += out_img_pred


        print("Finished generating predictions.")

        height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]


        video = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'DIVX'), 7, (width,height))

        for image in gif_imgs:
            img = (255*image).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.write(img)
        video.release()

        print('Predictions are saved to the video file: ', vide_save_path)