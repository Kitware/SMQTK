import glob
import json
import logging
import argparse
import six
import os.path as osp

# import skimage.io as io
from PIL import Image

from smqtk import algorithms
from smqtk import representation
from smqtk.utils import bin_utils, jsmin, plugin
from tqdm import tqdm
from smqtk.utils.bin_utils import initialize_logging

from smqtk.utils.coco_api.pycocotools.coco import COCO

__author__ = 'bo.dong@kitware.com'


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-c", "--config",
                        required=True,
                        help="IQR application configuration file.")
    parser.add_argument("-t", "--tab",
                        type=str, default=0,
                        help="The configuration tab name to generate the model for.")
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='Show debug logging.')

    # parser.add_argument("input_files",
    #                     metavar='GLOB', nargs="*",
    #                     help="Shell glob to files to add to the configured "
    #                          "data set.")
    parser.add_argument('--annotation_file', type=str,
                        help="COCO annotation file")
    parser.add_argument('--data_path', type=str,
                        help="COCO data root path")
    parser.add_argument('--query_imgID_list', type=str,
                        help="AMT query image ID")
    parser.add_argument('--out_uuid_file', type=str,
                        help="output uuid file")
    parser.add_argument('--min_class_num', type=int,
                        help="required minimum class number", default=2)

    return parser

def main():
    parser = cli_parser()
    args = parser.parse_args()

    #
    # Setup logging
    #
    if not logging.getLogger().handlers:
        if args.verbose:
            bin_utils.initialize_logging(logging.getLogger(), logging.DEBUG)
        else:
            bin_utils.initialize_logging(logging.getLogger(), logging.INFO)
    log = logging.getLogger("smqtk.scripts.iqr_app_model_generation")

    search_app_config = json.loads(jsmin.jsmin(open(args.config).read()))

    #
    # Input parameters
    #
    # The following dictionaries are JSON configurations that are used to
    # configure the various data structures and algorithms needed for the IQR demo
    # application. Values here can be changed to suit your specific data and
    # algorithm needs.
    #
    # See algorithm implementation doc-strings for more information on configuration
    # parameters (see implementation class ``__init__`` method).
    #

    # base actions on a specific IQR tab configuration (choose index here)
    if args.tab not in search_app_config["iqr_tabs"]:
        log.error("Invalid tab name provided.")
        exit(1)

    search_app_iqr_config = search_app_config["iqr_tabs"][args.tab]

    # Configure DataSet implementation and parameters
    data_set_config = search_app_iqr_config['data_set']

    query_set_config = search_app_iqr_config['query_set']

    # Configure DescriptorGenerator algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    descriptor_generator_config = search_app_iqr_config['descr_generator']

    # Configure NearestNeighborIndex algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    nn_index_config = search_app_iqr_config['nn_index']

    # Configure RelevancyIndex algorithm implementation, parameters and
    # persistant model component locations (if implementation has any).
    #
    # The LibSvmHikRelevancyIndex implementation doesn't actually build a persistant
    # model (or doesn't have to that is), but we're leaving this block here in
    # anticipation of other potential implementations in the future.
    #
    rel_index_config = search_app_iqr_config['rel_index_config']

    # Configure DescriptorElementFactory instance, which defines what implementation
    # of DescriptorElement to use for storing generated descriptor vectors below.
    descriptor_elem_factory_config = search_app_iqr_config['descriptor_factory']

    #
    # Initialize data/algorithms
    #
    # Constructing appropriate data structures and algorithms, needed for the IQR
    # demo application, in preparation for model training.
    #

    descriptor_elem_factory = \
        representation.DescriptorElementFactory \
            .from_config(descriptor_elem_factory_config)

    #: :type: representation.DataSet
    data_set = \
        plugin.from_plugin_config(data_set_config,
                                  representation.get_data_set_impls())

    query_set = \
        plugin.from_plugin_config(query_set_config,
                                  representation.get_data_set_impls())

    #: :type: algorithms.DescriptorGenerator
    descriptor_generator = \
        plugin.from_plugin_config(descriptor_generator_config,
                                  algorithms.get_descriptor_generator_impls())

    #: :type: algorithms.NearestNeighborsIndex
    nn_index = \
        plugin.from_plugin_config(nn_index_config,
                                  algorithms.get_nn_index_impls())

    #: :type: algorithms.RelevancyIndex
    rel_index = \
        plugin.from_plugin_config(rel_index_config,
                                  algorithms.get_relevancy_index_impls())

    #
    # Build models
    #
    # Perform the actual building of the models.
    #

    # Add data files to DataSet
    DataFileElement = representation.get_data_element_impls()["DataFileElement"]

    coco = COCO(args.annotation_file)

    of = open(args.out_uuid_file, 'w')

    # add query image to query_set
    with open(args.query_imgID_list, 'r') as f:
        for line in f:
            coco_img_id = line.rstrip('\n')
            coco_img_id = coco_img_id.strip('0').rstrip('.jpg')
            q_img = coco.loadImgs(int(coco_img_id))[0]

            annIds = coco.getAnnIds(imgIds=q_img['id'], iscrowd=0)
            anns = coco.loadAnns(annIds)
            cat_names = coco.obtainCatNames(anns)
            sorted_cat_bboxArea = coco.obtainAnns_cat_bboxArea(anns)

            if len(cat_names) == args.min_class_num:
                max_area = sorted_cat_bboxArea[0][1]
                max_key = sorted_cat_bboxArea[0][0]

                sec_max_area = sorted_cat_bboxArea[1][1]
                sec_max_key = sorted_cat_bboxArea[1][0]

                img_path = osp.join(args.data_path, q_img['file_name'])
                I = Image.open(img_path)
                img_area = I.size[0] * I.size[1]
                area_threshold = 0.6 * img_area

                if max_area > area_threshold and sec_max_area > area_threshold:
                    cur_data = None
                    if osp.isfile(img_path):
                        cur_data = DataFileElement(img_path, coco_catNM=cat_names)
                        query_set.add_data(cur_data)

                        # of.write('{}&{}&{}\n'.format(cur_data.uuid(), max_key, 0))
                        # of.write('{}&{}&{}\n'.format(cur_data.uuid(), max_key, 1))
                    else:
                        log.debug("Expanding glob: %s" % img_path)
                        for g in glob.iglob(img_path):
                            cur_data = DataFileElement(g, coco_catNM=cat_names)
                            query_set.add_data(cur_data)

                    of.write('{}&{}&{}\n'.format(cur_data.uuid(), max_key, 0))
                    of.write('{}&{}&{}\n'.format(cur_data.uuid(), max_key, 1))
                    if sec_max_key != max_key:
                        of.write('{}&{}&{}\n'.format(cur_data.uuid(), sec_max_key, 0))
                        of.write('{}&{}&{}\n'.format(cur_data.uuid(), sec_max_key, 1))

    of.close()


if __name__ == "__main__":
    main()
    print('Done')
