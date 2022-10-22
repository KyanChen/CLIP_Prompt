import json
import os
import argparse
# import img2dataset
import pandas as pd
from tqdm import tqdm
from concurrent import futures

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download CC images of the VAW/COCO attributes')
    parser.add_argument('--name-file', default='/Users/kyanchen/Documents/CC/Validation_GCC-1.1.0-Validation.tsv',
                        help='LVIS/COCO category name and description')
    parser.add_argument('--valid-ind-file', default='val_valid_cc_idx.json',
                        help='index of the LVIS/COCO base categories')
    parser.add_argument('--base-category', action='store_true',
                        help='whether to retrieval the images of the base categories')
    parser.add_argument('--output-folder', default='data/cc/images',
                        help='output path')
    parser.add_argument('--num-thread', type=int, default=10,
                        help='the number of the thread to download the images')
    args = parser.parse_args()
    return args


def download_fun(cls_names, output_folder):
    for i, cls_name in tqdm(enumerate(cls_names), total=len(cls_names)):
        file_path = os.path.join(output_folder, cls_name + ".txt")
        image_path = os.path.join(output_folder, cls_name)
        img2dataset.download(url_list=file_path, image_size=1024, output_folder=image_path, processes_count=64, timeout=20)

    return True

def open_tsv(fname):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', header=None)
    print("Processing", len(df), " Images:")
    return df

def main():
    args = parse_args()

    output_folder = args.output_folder
    name_file = args.name_file
    base_ind_file = args.base_ind_file
    num_thread = args.num_thread

    df = open_tsv(name_file)
    valid_idxs_dict = json.load(open(args.valid_ind_file))
    base_inds = open(base_ind_file, 'r').readline().strip().split(', ')
    base_inds = [int(ind) for ind in base_inds]
    novel_inds = [i for i in range(len(names)) if i not in base_inds]

    if args.base_category:
        names = [names[i] for i in base_inds]
    else:
        names = [names[i] for i in novel_inds]

    count_per_thread = (len(names) + num_thread - 1) // num_thread
    names_list = [names[i * count_per_thread:(i + 1) * count_per_thread] for i in range(num_thread)]

    with futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        threads = [executor.submit(download_fun, name, output_folder=output_folder) for name in names_list]
        for future in futures.as_completed(threads):
            print(future.result())


if __name__ == '__main__':
    main()
