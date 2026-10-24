# ------------------------------------------------------------------------
# Modified from SSU (https://github.com/cvlab-epfl/social-scene-understanding)
# Modified from ARG (https://github.com/wjchaoGit/Group-Activity-Recognition)
# ------------------------------------------------------------------------
from .cafe import *

import pickle
import os
import json
from concurrent.futures import ThreadPoolExecutor

TRAIN_CAFE_P = ['1', '2', '3', '4', '9', '10', '11', '12', '17', '18', '19', '20', '21', '22', '23', '24']
VAL_CAFE_P = ['13', '14', '15', '16']
TEST_CAFE_P = ['5', '6', '7', '8']

TRAIN_CAFE_V = ['1', '2', '5', '6', '9', '10', '13', '14', '17', '18', '21', '22']
VAL_CAFE_V = ['3', '7', '11', '15', '19', '23']
TEST_CAFE_V = ['4', '8', '12', '16', '20', '24']


def read_dataset(args):
    if args.dataset == 'cafe':
        data_path = os.path.join(args.data_path, 'cafe')
        print(f"    Data path: {data_path}")

        # split-by-place setting
        if args.split == 'place':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_P
            VAL_VIDEOS_CAFE = VAL_CAFE_P
            TEST_VIDEOS_CAFE = TEST_CAFE_P
            print(f"    Using split-by-place")
        # split-by-view setting
        elif args.split == 'view':
            TRAIN_VIDEOS_CAFE = TRAIN_CAFE_V
            VAL_VIDEOS_CAFE = VAL_CAFE_V
            TEST_VIDEOS_CAFE = TEST_CAFE_V
            print(f"    Using split-by-view")
        else:
            assert False

        clip_manifest = getattr(args, 'clip_manifest', '')
        selected = None
        if clip_manifest:
            with open(clip_manifest) as stream:
                selected = json.load(stream)

        def read_annotations(videos, split=None):
            clip_ids = ({tuple(item['clip']) for item in selected[split]}
                        if selected is not None else None)
            workers = getattr(args, 'annotation_workers', 0)
            if workers <= 1:
                return cafe_read_annotations(data_path, videos, args.num_class, clip_ids=clip_ids)
            # Read disjoint video directories concurrently; map returns parts
            # in the supplied video order, so clip ordering stays stable.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                parts = pool.map(lambda video: cafe_read_annotations(
                    data_path, [video], args.num_class, clip_ids=clip_ids), videos)
                merged = {}
                for part in parts:
                    merged.update(part)
            return merged

        if args.val_mode:
            print(f"    Val mode: using TRAIN for training, VAL for testing")
            train_data = read_annotations(TRAIN_VIDEOS_CAFE, 'train')
            train_frames = cafe_all_frames(train_data)

            test_data = read_annotations(VAL_VIDEOS_CAFE, 'val')
            test_frames = cafe_all_frames(test_data)
        else:
            if selected is not None:
                raise ValueError('clip_manifest is only supported for TRAIN->VAL diagnostics')
            print(f"    Normal mode: using TRAIN+VAL for training, TEST for testing")
            print(f"    Loading training data from videos: {TRAIN_VIDEOS_CAFE + VAL_VIDEOS_CAFE}")
            train_data = read_annotations(TRAIN_VIDEOS_CAFE + VAL_VIDEOS_CAFE)
            train_frames = cafe_all_frames(train_data)
            print(f"    Loaded {len(train_frames)} training clips")

            print(f"    Loading test data from videos: {TEST_VIDEOS_CAFE}")
            test_data = read_annotations(TEST_VIDEOS_CAFE)
            test_frames = cafe_all_frames(test_data)
            print(f"    Loaded {len(test_frames)} test clips")

        # actor tracklets for all frames.  The pickle can be placed on local
        # storage to avoid a repeated large NAS read during each DDP startup.
        tracks_path = getattr(args, 'tracks_path', '') or os.path.join(data_path, 'gt_tracks.pkl')
        print(f"    Loading actor tracklets from: {tracks_path}")
        with open(tracks_path, 'rb') as tracks_file:
            all_tracks = pickle.load(tracks_file)
        print(f"    Tracklets loaded")

        print(f"    Creating dataset objects...")
        train_set = CafeDataset(train_frames, train_data, all_tracks, data_path, args, is_training=True)
        test_set = CafeDataset(test_frames, test_data, all_tracks, data_path, args, is_training=False)
        if getattr(args, 'fixed_center_train', False):
            # Explicitly request frozen segment centers for TRAIN.
            train_set.is_training = False
    else:
        assert False

    print("%d train samples and %d test samples" % (len(train_frames), len(test_frames)))

    return train_set, test_set
