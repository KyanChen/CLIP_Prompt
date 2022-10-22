import json
import os
import argparse
import img2dataset
import mmcv
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download CC images of the VAW/COCO attributes')
    # parser.add_argument('--name-file', default='/data/kyanchen/prompt/data/CC/Train_GCC-training.tsv',
    parser.add_argument('--name-file', default='/Users/kyanchen/Documents/CC/Validation_GCC-1.1.0-Validation.tsv',
                        help='LVIS/COCO category name and description')
    parser.add_argument('--valid-ind-file', default='val_valid_cc_idx.json',
                        help='index of the LVIS/COCO base categories')
    parser.add_argument('--base-category', action='store_true',
                        help='whether to retrieval the images of the base categories')
    parser.add_argument('--output-folder', default='/Users/kyanchen/Documents/CC/images',
                        help='output path')
    parser.add_argument('--num-thread', type=int, default=10,
                        help='the number of the thread to download the images')
    args = parser.parse_args()
    return args


def open_tsv(fname):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', header=None)
    print("Processing", len(df), " Images:")
    return df


def main():
    args = parse_args()
    output_folder = args.output_folder
    name_file = args.name_file
    df = open_tsv(name_file)
    valid_idxs_dict = json.load(open(args.valid_ind_file))
    url_list = [df.iloc[int(idx), 1] for idx in valid_idxs_dict.keys()]
    url_list = [x+'\n' for idx, x in enumerate(url_list)]
    all_data = {'ori_id': list(valid_idxs_dict.keys()), 'url': url_list, 'caption': list(valid_idxs_dict.values())}
    print('need down num imgs: ', len(url_list))
    mmcv.dump(all_data, 'tmp_pair_json_file.json')
    img2dataset.download(
        url_list='tmp_pair_json_file.json',
        input_format='json',
        save_additional_columns=['ori_id'],
        caption_col='caption',
        min_image_size=48,
        max_aspect_ratio=5.,
        image_size=1024,
        thread_count=8,
        output_folder=output_folder,
        processes_count=2,
        timeout=20
    )


if __name__ == '__main__':
    main()
