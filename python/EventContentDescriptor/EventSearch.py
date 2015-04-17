"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import json
import os.path as osp
import os
import glob
import event_agent_generator as eag
import numpy as np

from SMQTK_Backend.ControllerProcess import ControllerProcess
from SMQTK_Backend.utils import ProgressStatus


class EventSearch (ControllerProcess):
    """
    Process specifically responsible for executing search engine queries.
    """

    NAME = "SearchEngineQuery"

    # DEMO: manually encode small number of classifiers
    # TODO: finding the list of underlying classifiers should be automated
    #       by running multiple EventSearchProcess with different CLASSIFIER_PAIRS, it can be parallelized
    #       That means that classifier pair should be part of controller configurations
    CLASSIFIER_PAIRS = [
                        ('HoG3D_cw4000', 'ngd'),
                        ('MFCCs', 'ngd'),
                        ('CSIFT', 'hik'),
                        ]
    # DEMO: file to store scores computed by base/fusion classifiers
    FILE_SCORE = 'scores.txt'

    def __init__(self, controller, event_type, dataset_config, output_filepath):
        """
        Create a new search query process.

        :param controller: The controller that owns this process
        :param event_type: The event type ID
        :param dataset_config: The dataset label to use
        :param output_filepath: The file to output results to. This file will
            be modified while RUNNING.

        ..
            :type event_type: int
            :type dataset_config: str
            :type output_filepath: str

        """
        super(EventSearch, self).__init__(controller)

        self._event_type = int(event_type)
        self._dataset_config = str(dataset_config)

        # DEMO: this needs to be updated with VCD path later.
        self._data_directory = str(controller.data_dirrectory)

        self._output_filepath = str(output_filepath)

        # SharedAttribute initialization

    @property
    def event_type(self):
        return self._event_type

    @property
    def dataset(self):
        return self._dataset_config

    @property
    def output_filepath(self):
        return self._output_filepath

    @property
    def data_directory(self):
        return self._data_directory

    def get_results(self, top_n=None):
        """
        :return: The search results. There may be no search results under
            certain conditions.
        :param top_n: Return only the top N results based on confidence score.
            (not functional)

        ..
            :type top_n: int
            :rtype: dict of (int, float)

        """
        # TODO: If output filepath doesn't exist yet, return None
        #       If not yet started running, return None as we couldn't have
        #           possibly done anything yet.
        if not osp.isfile(self.output_filepath):
            return {}
        elif self.get_status().status == ProgressStatus.NOT_STARTED:
            return {}
        elif not osp.isfile(self.output_filepath):
            return {}
        else:
            # open file as json and return contained dict
            d = json.load(open(self.output_filepath))
            # JSON dict keys **have** to be strings, so convert them back into
            # integers now
            return dict((int(e[0]), e[1]) for e in d.items())


    def _run(self):

        # DEMO: configured to assume MED'12 system on videonas2
        #    path_model: location to load computed models
        #    path_search_data: location to load search data
        #    both variables are hard coded.
        # TODO: make path configurations more generic,
        #       directly using VCD, and intermediate MongoDB usable

        str_eid = 'E%03d'%self.event_type

        if self.dataset == 'MED11-compare':
            path_ECD = osp.join(self.data_directory, 'MED_Systems',
                                  self.dataset, 'Run3_2012Dec', 'EventAgents')
            path_search_data = osp.join(self.data_directory, 'MED_Data_TEST', 'MED11TEST')
        elif self.dataset == 'MED11-DEV':
            path_ECD = osp.join(self.data_directory, 'MED_Systems',
                                  self.dataset, 'EventAgents')
            path_search_data = osp.join(self.data_directory, 'MED_Data_TEST', 'MED11part1DEV')
        else:
            raise ValueError("Invalid dataset specifier")

        # configure separate output paths per base classifier configuration
        jobs_base_classifiers = []
        for (feature, str_kernel) in self.CLASSIFIER_PAIRS:
            job = dict()

            # base classifier ECD name
            job['str_ECD']             = feature +'_'+ str_kernel
            job['eid']                 = str_eid
            # base classifier ECD location (already exists)
            job['dirpath_ECD']         = osp.join(path_ECD, job['str_ECD'], str_eid, eag.FILE_EA_MODEL_LIBSVM_COMPACT_DIR)
            # location of test data
            job['dirpath_TEST_DATA']   = osp.join(path_search_data, feature)
            # location to store outputs by this base classifier
            job['dirpath_OUTPUT']      = osp.join(osp.dirname(self.output_filepath),
                                                  job['str_ECD'], str_eid)

            # exact file to
            job['filepath_OUTPUT']     = osp.join(job['dirpath_OUTPUT'], self.FILE_SCORE)

            # depending on kernel type, use approximate model or exactm odel
            if str_kernel == 'hik':
                use_approx_model = True
            elif str_kernel in ['ngd', 'linear', 'L1norm_linear']:
                use_approx_model = False
            else:
                raise ValueError("Invalid kernel specifier", str_kernel)
            job['use_approx_model'] = use_approx_model

            # add the formulated job into job queue
            jobs_base_classifiers.append(job)


        # stores all the scores across base classifiers, prior to final fusion
        scores_base_ecds = dict()

        # execute every base classifier
        for job in jobs_base_classifiers:
            print 'processing %s on %s'%(job['str_ECD'], job['eid'])

            if not osp.exists(job['dirpath_OUTPUT']):
                os.makedirs(job['dirpath_OUTPUT'])
                print '\nCreated output directory: ', job['dirpath_OUTPUT']
            else:
                print '\nOuput directory exits: ', job['dirpath_OUTPUT']

            print '\nComputing scores using :', job['str_ECD']

            eag.apply_compact_event_agents_libSVM(job['dirpath_ECD'],
                                                  job['dirpath_TEST_DATA'],
                                                  job['dirpath_OUTPUT'],
                                                  use_approx_model = job['use_approx_model'],
                                                  report_margins_recounting = False)

            # agglomerate scattered computed scores into one file
            FILE_PROB_PATTERN = 'part_*_probs.txt'
            SUBDIR = 'compact'
            files_full = sorted(glob.glob(osp.join(job['dirpath_OUTPUT'], FILE_PROB_PATTERN)))
            scores_base = np.vstack(tuple(map(np.loadtxt, files_full)))
            np.savetxt(job['filepath_OUTPUT'], scores_base)
            for (clipid, score) in scores_base.tolist():
                if scores_base_ecds.has_key(int(clipid)):
                    scores_base_ecds[int(clipid)].append(score)
                else:
                    scores_base_ecds[int(clipid)] = [score]

        # fusion by simple average fusion
        scores_fusion = dict()
        for clipid in scores_base_ecds.iterkeys():
            scores_fusion[clipid] = np.average(np.array(scores_base_ecds[clipid]))

        # TODO: Incerementally store data into a PQ, sleeping a little in
        # between inserts to simulate incremental generation and access
        #data = dict((int(r[0]), r[2]) for r in data if r[1] == self.event_type)
        json.dump(scores_fusion, open(self.output_filepath, 'w'))





