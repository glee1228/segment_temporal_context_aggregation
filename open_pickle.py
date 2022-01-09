import argparse
import pickle as pk
import glob
import os


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
    vid2paths_bg = {}
    for path in paths:
        vid2paths_bg[path.split('/')[-1].split('.')[0]] = path
    return vid2paths_core, vid2paths_bg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--annotation_path', type=str, default='/workspace/TCA/datasets/vcdb.pickle',
                        help='Path to the .pk file that contains the annotations of the train set')
    args = parser.parse_args()
    vcdb = pk.load(open(args.annotation_path, 'rb'))
    pairs = []
    for pair in vcdb['video_pairs']:
        vid1, vid2 = pair['videos'][0], pair['videos'][1]
        pairs.append([vid1, vid2])
    negs = vcdb['negs']
    vid2paths_core, vid2paths_bg = load_video_paths_vcdb('/mldisk/nfs_shared_/MLVD/VCDB/videos/')
    print(vid2paths_core[pairs[0][0]])
    print(vid2paths_core[pairs[0][1]])

    import pdb;pdb.set_trace()
if __name__ == '__main__':
    main()
