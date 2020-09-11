"""
Script for asynchronously computing classifications for DescriptorElements
in a DescriptorSet specified via a list of UUIDs. Results are output to a
CSV file in the format:

    uuid, label1_confidence, label2_confidence, ...

CSV columns labels are output to the given CSV header file path. Label
columns will be in the order as reported by the classifier implementations
``get_labels`` method.

Due to using an input file-list of UUIDs, we require that the UUIDs of
indexed descriptors be strings, or equality comparable to the UUIDs' string
representation.
"""

import csv
import logging
import os

from smqtk.algorithms import (
    Classifier
)
from smqtk.representation import (
    ClassificationElementFactory,
    ClassificationElement,
    DescriptorSet,
)
from smqtk.utils import (
    cli,
    parallel,
)
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
)
from smqtk.utils.file import safe_create_dir


__author__ = "paul.tunison@kitware.com"


def default_config():
    return {
        "utility": {
            "classify_overwrite": False,
            "parallel": {
                "use_multiprocessing": False,
                "index_extraction_cores": None,
                "classification_cores": None,
            }
        },
        "plugins": {
            "classifier": make_default_config(Classifier.get_impls()),
            "classification_factory": make_default_config(
                ClassificationElement.get_impls()
            ),
            "descriptor_set": make_default_config(
                DescriptorSet.get_impls()
            ),
        }
    }


def cli_parser():
    parser = cli.basic_cli_parser(__doc__)

    g_io = parser.add_argument_group("Input Output Files")
    g_io.add_argument('--uuids-list', metavar='PATH',
                      help='Path to the input file listing UUIDs to process.')
    g_io.add_argument('--csv-header', metavar='PATH',
                      help='Path to the file to output column header labels.')
    g_io.add_argument('--csv-data', metavar='PATH',
                      help='Path to the file to output the CSV data to.')
    return parser


def main():
    args = cli_parser().parse_args()
    config = cli.utility_main_helper(default_config, args)
    log = logging.getLogger(__name__)

    # - parallel_map UUIDs to load from the configured index
    # - classify iterated descriptors

    uuids_list_filepath = args.uuids_list
    output_csv_filepath = args.csv_data
    output_csv_header_filepath = args.csv_header
    classify_overwrite = config['utility']['classify_overwrite']

    p_use_multiprocessing = \
        config['utility']['parallel']['use_multiprocessing']
    p_index_extraction_cores = \
        config['utility']['parallel']['index_extraction_cores']
    p_classification_cores = \
        config['utility']['parallel']['classification_cores']

    if not uuids_list_filepath:
        raise ValueError("No uuids_list_filepath specified.")
    elif not os.path.isfile(uuids_list_filepath):
        raise ValueError("Given uuids_list_filepath did not point to a file.")
    if output_csv_header_filepath is None:
        raise ValueError("Need a path to save CSV header labels")
    if output_csv_filepath is None:
        raise ValueError("Need a path to save CSV data.")

    #
    # Initialize configured plugins
    #

    log.info("Initializing descriptor index")
    #: :type: smqtk.representation.DescriptorSet
    descriptor_set = from_config_dict(
        config['plugins']['descriptor_set'],
        DescriptorSet.get_impls()
    )

    log.info("Initializing classification factory")
    c_factory = ClassificationElementFactory.from_config(
        config['plugins']['classification_factory']
    )

    log.info("Initializing classifier")
    #: :type: smqtk.algorithms.Classifier
    classifier = from_config_dict(
        config['plugins']['classifier'], Classifier.get_impls()
    )

    #
    # Setup/Process
    #
    def iter_uuids():
        with open(uuids_list_filepath) as f:
            for l in f:
                yield l.strip()

    def descr_for_uuid(uuid):
        """
        :type uuid: collections.Hashable
        :rtype: smqtk.representation.DescriptorElement
        """
        return descriptor_set.get_descriptor(uuid)

    def classify_descr(d):
        """
        :type d: smqtk.representation.DescriptorElement
        :rtype: smqtk.representation.ClassificationElement
        """
        return classifier.classify_one_element(d, c_factory,
                                               classify_overwrite)

    log.info("Initializing uuid-to-descriptor parallel map")
    #: :type: collections.Iterable[smqtk.representation.DescriptorElement]
    element_iter = parallel.parallel_map(
        descr_for_uuid, iter_uuids(),
        use_multiprocessing=p_use_multiprocessing,
        cores=p_index_extraction_cores,
        name="descr_for_uuid",
    )

    log.info("Initializing descriptor-to-classification parallel map")
    #: :type: collections.Iterable[smqtk.representation.ClassificationElement]
    classification_iter = parallel.parallel_map(
        classify_descr, element_iter,
        use_multiprocessing=p_use_multiprocessing,
        cores=p_classification_cores,
        name='classify_descr',
    )

    #
    # Write/Output files
    #

    c_labels = classifier.get_labels()

    def make_row(e):
        """
        :type e: smqtk.representation.ClassificationElement
        """
        c_m = e.get_classification()
        return [e.uuid] + [c_m[l] for l in c_labels]

    # column labels file
    log.info("Writing CSV column header file: %s", output_csv_header_filepath)
    safe_create_dir(os.path.dirname(output_csv_header_filepath))
    with open(output_csv_header_filepath, 'wb') as f_csv:
        w = csv.writer(f_csv)
        w.writerow(['uuid'] + [str(cl) for cl in c_labels])

    # CSV file
    log.info("Writing CSV data file: %s", output_csv_filepath)
    safe_create_dir(os.path.dirname(output_csv_filepath))
    pr = cli.ProgressReporter(log.info, 1.0)
    pr.start()
    with open(output_csv_filepath, 'wb') as f_csv:
        w = csv.writer(f_csv)
        for c in classification_iter:
            w.writerow(make_row(c))
            pr.increment_report()
        pr.report()

    log.info("Done")


if __name__ == '__main__':
    main()
