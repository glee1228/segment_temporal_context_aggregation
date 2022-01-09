import os
import cv2
import glob
import numpy as np
import pickle as pk
import json
from tqdm import tqdm
from multiprocessing import Pool
# from tqdm_multiprocess import TqdmMultiProcessPool


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def center_crop(frame, desired_size):
    old_size = frame.shape[:2]
    # print(old_size)
    top = int(np.maximum(0, (old_size[0] - desired_size)/2))
    left = int(np.maximum(0, (old_size[1] - desired_size)/2))
    # print(frame.shape)
    return frame[top: top+desired_size, left: left+desired_size, :]


def load_video(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25

    i = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            if i == 0 or int((i + 1) % round(fps)) == 0:
                frames.append(center_crop(resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 256), 256))
        else:
            break
        i = i + 1
    cap.release()
    if i > 0:
        frames.append(frames[-1])

    return np.array(frames)


def load_video_fps_4(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 144 or fps is None:
        fps = 25

    i = 0
    video_frames = []

    keyframes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):
            video_frames.append(center_crop(resize_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 256), 256))
            if i == 0 or int((i + 1) % round(fps)) == 0:
                keyframes.append(i)
        else:
            break
        i = i + 1
    cap.release()

    video_frames = np.array(video_frames)

    clip_indices = np.zeros((len(keyframes),4),dtype=int)

    out_frames = []
    for k, j in enumerate(keyframes):
        frame_indices = np.linspace(max(0, j - round(fps) / 2), min(len(video_frames)-1,j + 1 + round(fps) / 2), 4, endpoint=False, dtype=int)
        clip_indices[k,:] = frame_indices

    for clip_i in range(len(clip_indices)):
        out_frames.append(video_frames[clip_indices[clip_i]])

    if i > 0:
        out_frames.append(out_frames[-1])
    out_frames = np.array(out_frames)

    return np.array(out_frames)


def load_video_paths_ccweb(root='/workspace/datasets/ccweb/'):
    paths = sorted(glob.glob(root + 'videos/*.*'))
    vid2paths = {}
    for path in paths:
        vid2paths[path.split('/')[-1].split('.')[0]] = path
    return vid2paths

def load_video_paths_vcdb(root='/workspace/datasets/vcdb/'):
    paths = sorted(glob.glob(root + 'core_dataset/*/*.*'))
    vid2paths_core = {}
    for path in paths:
        vid2paths_core[path.split('/')[-1].split('.')[0]] = path

    paths = sorted(glob.glob(root + 'distraction/*/*.*'))
    vid2paths_bg = {}
    for path in paths:
        vid2paths_bg[path.split('/')[-1].split('.')[0]] = path
    return vid2paths_core, vid2paths_bg


def load_video_paths_fivr(root='/workspace/datasets/fivr/'):
    paths = sorted(glob.glob(root + 'core/*.*'))
    vid2paths_core = {}
    for path in paths:
        vid2paths_core[path.split('/')[-1].split('.')[0]] = path

    paths = sorted(glob.glob(root + 'distraction/*/*.*'))
    # paths = sorted(glob.glob(root + 'distraction/0001/*.*'))
    vid2paths_bg = {}
    for path in paths:
        vid2paths_bg[path.split('/')[-1].split('.')[0]] = path
    return vid2paths_core, vid2paths_bg

def get_frames_ccweb(vid2paths, root='/workspace/datasets/ccweb/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        np.save(root + 'frames/' + vid + '.npy', frames)

def get_frames_vcdb_core(vid2paths, root='/workspace/datasets/vcdb/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        if not os.path.exists(root + 'frames/core/'):
            os.mkdir(root + 'frames/core/')
        np.save(root + 'frames/core/' + vid + '.npy', frames)

def get_frames_vcdb_bg(vid2paths, root='/workspace/datasets/vcdb/'):
    for vid, path in tqdm(vid2paths.items()):
        frames = load_video(path)
        if not os.path.exists(root + 'frames/background_dataset/' + path.split('/')[-2] + '/'):
            os.mkdir(root + 'frames/background_dataset/' + path.split('/')[-2] + '/')
        np.save(root + 'frames/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)

def load_video_paths_evve(root='/workspace/datasets/evve/'):
    paths = sorted(glob.glob(root + 'videos/*/*.mp4'))
    vid2paths = {}
    for path in paths:
        vid2paths[path.split('/')[-1].split('.')[0]] = path
    return vid2paths

def ccweb_1save(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/ccweb/frames'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/ccweb/frames',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/ccweb/frames/' + vid + '.npy', frames)

def vcdb_core_1save(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/core/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/core/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/core/' + vid + '.npy', frames)

def vcdb_bg_1save(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/vcdb/frames/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)

def vcdb_core_4save(args):
    vid, path = args
    if vid == None:
        return None
    frames = load_video_fps_4(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/core/' + path.split('/')[-2] + '/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/core/' + path.split('/')[-2] + '/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/core/' + path.split('/')[-2] + '/' + vid + '.npy', frames)

def vcdb_bg_4save(args):
    vid, path = args
    if vid == None:
        return None
    frames = load_video_fps_4(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)

def fivr_core_1save(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames/core/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/fivr/frames/core/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/fivr/frames/core/' + vid + '.npy', frames)

def fivr_bg_1save(args):
    vid, path = args
    frames = load_video(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames/background_dataset/' + path.split('/')[-2] + '/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/fivr/frames/background_dataset/' + path.split('/')[-2] + '/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/fivr/frames/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)

def fivr_core_4save(args):
    vid, path = args
    if vid == None:
        return None
    frames = load_video_fps_4(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/core/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/core/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/core/' + vid + '.npy', frames)

def fivr_bg_4save(args):
    vid, path = args
    if vid == None:
        return None
    frames = load_video_fps_4(path)
    if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/'):
        os.makedirs('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/',exist_ok=True)
    np.save('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy', frames)


def load_video_paths_fivr_version(root='/mldisk/nfs_shared_/MLVD/FIVR/videos/', version='5k'):
    with open('/workspace/CTCA/datasets/fivr.pickle', 'rb') as f:
        dataset = pk.load(f)
    annotation = dataset['annotation']
    queries = dataset[version]['queries']
    database = dataset[version]['database']
    anno_videos = []
    for q,types in annotation.items():
        if q in queries:
            for type, values in types.items():
                    for value in values:
                        anno_videos.append(value)

    anno_videos = list(set(anno_videos))
    vid2paths_core, vid2paths_bg = load_video_paths_fivr(root)
    new_vid2paths_core, new_vid2paths_bg = {},{}
    need_5k = list(set(anno_videos + list(database) + queries))
    for vid, path in vid2paths_core.items():
        if vid in need_5k:
            new_vid2paths_core[vid]=path
    for vid, path in vid2paths_bg.items():
        if vid in need_5k:
            new_vid2paths_bg[vid]=path
    # breakpoint()
    return new_vid2paths_core,new_vid2paths_bg




if __name__ == "__main__":
    def update(*a):
        pbar.update()

    vid2paths_core, vid2paths_bg = load_video_paths_vcdb('/mldisk/nfs_shared_/MLVD/VCDB/videos/')

    pbar = tqdm(total=len(vid2paths_core.keys()))
    pool = Pool(16)
    for vid, path in tqdm(vid2paths_core.items()):
        pool.apply_async(vcdb_core_1save, ((vid, path),))
    pool.close()
    pool.join()


    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_bg.keys()))
    for vid, path in tqdm(vid2paths_bg.items()):
        pool.apply_async(vcdb_bg_1save, ((vid, path),))
    pool.close()
    pool.join()

    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_core.keys()))
    for vid, path in tqdm(vid2paths_core.items()):
        if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/core/' +  vid + '.npy'):
            pool.apply_async(vcdb_core_4save, ((vid, path),), callback=update)
        else:
            pool.apply_async(vcdb_core_4save, ((None, None),), callback=update)
    pool.close()
    pool.join()

    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_bg.keys()))
    vid2paths_bg_list=list(zip(vid2paths_bg.keys(), vid2paths_bg.values()))
    for vid, path in tqdm(vid2paths_bg_list):
        if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/vcdb/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy'):
            pool.apply_async(vcdb_bg_4save, ((vid, path),),callback=update)
        else:
            pool.apply_async(vcdb_bg_4save, ((None, None),), callback=update)
    pool.close()
    pool.join()



    vid2paths_core, vid2paths_bg = load_video_paths_fivr_version('/mldisk/nfs_shared_/MLVD/FIVR/videos/', version='5k')

    pbar = tqdm(total=len(vid2paths_core.keys()))
    pool = Pool(16)
    for vid, path in tqdm(vid2paths_core.items()):
        pool.apply_async(fivr_core_1save, ((vid, path),))
    pool.close()
    pool.join()


    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_bg.keys()))
    for vid, path in tqdm(vid2paths_bg.items()):
        pool.apply_async(fivr_bg_1save, ((vid, path),))
    pool.close()
    pool.join()

    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_core.keys()))
    for vid, path in tqdm(vid2paths_core.items()):
        if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/core/' +  vid + '.npy'):
            pool.apply_async(fivr_core_4save, ((vid, path),), callback=update)
        else:
            pool.apply_async(fivr_core_4save, ((None, None),), callback=update)
    pool.close()
    pool.join()

    pool = Pool(16)
    pbar = tqdm(total=len(vid2paths_bg.keys()))
    vid2paths_bg_list=list(zip(vid2paths_bg.keys(), vid2paths_bg.values()))
    for vid, path in tqdm(vid2paths_bg_list):
        if not os.path.exists('/mldisk/nfs_shared_/dh/datasets/fivr/frames_fps_4/background_dataset/' + path.split('/')[-2] + '/' + vid + '.npy'):
            pool.apply_async(fivr_bg_4save, ((vid, path),),callback=update)
        else:
            pool.apply_async(fivr_bg_4save, ((None, None),), callback=update)
    pool.close()
    pool.join()






