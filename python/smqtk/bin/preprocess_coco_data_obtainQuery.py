import argparse
import os.path as osp

from tqdm import tqdm

from smqtk.utils.coco_api.pycocotools.coco import COCO

__author__ = 'bo.dong@kitware.com'


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation_file', type=str,
                        help="COCO annotation file")
    parser.add_argument('--data_path', type=str,
                        help="COCO data root path")
    parser.add_argument('--query_imgID_list', type=str,
                        help="AMT query image ID")
    parser.add_argument('--out_uuid_file', type=str,
                        help="output uuid file")
    parser.add_argument('--min_class_num', type=int,
                        help="required minimum class number", default=0)
    return parser


def main():
    parser = cli_parser()
    args = parser.parse_args()

    coco = COCO(args.annotation_file)

    # obtain all images
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds)

    count = 0
    for img in tqdm(imgs, total=len(imgs), desc='add image to dataset'):
        # obtain corresponding annoation
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=0)
        anns = coco.loadAnns(annIds)
        cat_names = coco.obtainCatNames(anns)

        if len(cat_names) >= args.min_class_num:
            count += 1
            img_path = osp.join(args.data_path, img['file_name'])

    print("{}/{} image has been added".format(count, len(imgs)))


if __name__ == "__main__":
    main()
    print('Done')
