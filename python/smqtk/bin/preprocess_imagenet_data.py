import os.path as osp
import os
import argparse
from glob import glob
from tqdm import tqdm

def process_label_file(file_name):
    label_d = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            folder_name = line.split()[0]

            line = line.replace(folder_name+' ', '')
            label = line.split(', ')[0]
            label = label.replace(' ', '_')

            label_d[folder_name] = label

    f.close()
    return label_d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DenseNet Cifar foolbox testing')

    parser.add_argument('--root_dir', default='/home/bdong/personal/data/imagenet_val/val', type=str,
                        help='root path to imagenet val data')
    parser.add_argument('--folder2label_file', default='/home/bdong/personal/data/imagenet_val/synset_words.txt', type=str,
                        help='folder name to label file')
    parser.add_argument('--out_root', default='/home/bdong/XAI/data/ImageNet_val_links', type=str,
                        help='output root path')
    parser.add_argument('--out_list_file', default='/home/bdong/XAI/imagenet_gt_label_list.txt', type=str,
                        help='output list file name')

    args = parser.parse_args()
    label_d = process_label_file(args.folder2label_file)

    folder_list = glob(osp.join(args.root_dir, "*", ""))

    out_list_file = open(args.out_list_file, 'w')

    for i in tqdm(range(len(folder_list))):
        src_folder_name = folder_list[i].split('/')[-2]
        des_folder_name = label_d[src_folder_name]

        des_file_path = osp.join(args.out_root, des_folder_name)
        os.makedirs(des_file_path, exist_ok=True)

        src_file_path = osp.join(args.root_dir, src_folder_name)

        img_list = glob(osp.join(src_file_path, "*.*"))

        for img in tqdm(img_list, total=len(img_list), desc="process class {}".format(des_folder_name)):
            src_img_name = img.split('/')[-1]
            des_img_name = osp.join(des_file_path, src_img_name)
            os.symlink(img, des_img_name)

            out_list_file.write(des_img_name + '\n')

    out_list_file.close()


