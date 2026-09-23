#!/usr/bin/env python3
"""Verify complete, finite 1408-D VideoMAE features for the Café dataset."""

import argparse
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def numeric_sort_key(path):
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def expected_feature_names(dataset_root):
    expected = set()
    for video_path in sorted(dataset_root.iterdir(), key=numeric_sort_key):
        if not video_path.is_dir():
            continue
        for clip_path in sorted(video_path.iterdir(), key=numeric_sort_key):
            if not clip_path.is_dir():
                continue
            image_path = clip_path / 'images'
            search_path = image_path if image_path.is_dir() else clip_path
            if any(file_path.suffix.lower() in IMAGE_EXTENSIONS for file_path in search_path.iterdir()):
                expected.add('{}_{}.npy'.format(video_path.name, clip_path.name))
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--dataset', default='cafe')
    parser.add_argument('--features_path', required=True)
    args = parser.parse_args()

    root_path = Path(args.data_path)
    dataset_root = root_path / args.dataset if (root_path / args.dataset).is_dir() else root_path
    features_path = Path(args.features_path)
    expected = expected_feature_names(dataset_root)
    actual = {path.name for path in features_path.glob('*.npy')}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    invalid = []
    for feature_name in sorted(expected & actual):
        try:
            feature = np.load(features_path / feature_name)
            if feature.shape != (1408,) or not np.isfinite(feature).all():
                invalid.append('{} (shape: {})'.format(feature_name, feature.shape))
        except Exception as error:
            invalid.append('{} ({})'.format(feature_name, error))

    print('Expected: {}; found: {}; missing: {}; unexpected: {}; invalid: {}'.format(
        len(expected), len(actual), len(missing), len(unexpected), len(invalid)))
    for label, values in [('Missing', missing), ('Unexpected', unexpected), ('Invalid', invalid)]:
        if values:
            print('{}: {}'.format(label, ', '.join(values)))

    if missing or unexpected or invalid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
