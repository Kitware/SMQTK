"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import event_agent_generator as eag
import json

class EventLearningBaseLIBSVM ():
    """
    Learn Base ECDs using libSVM
    """

    NAME = "BaseEventLearningProcess_LIBSVM"

    # STATE_MACHINE_STAGE_STRINGS = [('USE_SETTING_KERNEL', 'Compute kernel matrices, usually takes long time'),
    #                                ('USE_SETTING_TRAIN', 'Do initial training and find best model parameters'),
    #                                ('USE_SETTING_TRAIN_COMPACT', 'Learn more compact/approximate model')]
    #
    # # integer assignments for states
    # STATE_USE_SETTING_KERNEL = 0
    # STATE_USE_SETTING_TRAIN = 1
    # STATE_USE_SETTING_TRAIN_COMPACT = 2
    #
    # # range of states
    # STATE_MACHINE_STAGE_MIN = STATE_USE_SETTING_KERNEL
    # STATE_MACHINE_STAGE_MAX = STATE_USE_SETTING_TRAIN_COMPACT

    def __init__(self,
                 # state=STATE_USE_SETTING_KERNEL,
                 # flag_state_machine_stage_auto=True,
                 ):
        """
        Create a new base ECD learning process

        @param state: indicates where in different stages in learning this process will start
        @param flag_state_machine_stage_auto: if True, moves to the next learning stages automatically, if not, terminate after the initial state

        """

        # if (state < self.STATE_MACHINE_STAGE_MIN) or (state > self.STATE_MACHINE_STAGE_MAX):
        #     raise Exception('Unknown initial state', state)
        #
        # self._state = state
        # self._flag_state_machine_stage_auto = flag_state_machine_stage_auto
        # self._kernel = None  # kernel type string

    # def set_state(self, state):
    #     """
    #     Set new state for learning process, useful if we are re-using an object
    #     @param state: STATE_MACHINE_STAGE_MIN <= state < STATE_MACHINE_STAGE_MAX
    #     """
    #     if (state < self.STATE_MACHINE_STAGE_MIN) or (state > self.STATE_MACHINE_STAGE_MAX):
    #         raise Exception('Unknown initial state', state)
    #     self._state = state
    #     pass
    #
    # def reset_state(self):
    #     """
    #     Reset state to initial state
    #     """
    #
    #     self._state = self.STATE_USE_SETTING_KERNEL
    #     pass
    #
    # def _advance_state_machine(self):
    #     """
    #     Moves the stage to next step
    #     """
    #
    #     if self._flag_state_machine_stage_auto:
    #         self._state += 1
    #     else:
    #         state_machine_stage = self.STATE_MACHINE_STAGE_MAX + 1 # state beyond max indicates termination
    #     return state_machine_stage
    #
    # @property
    # def state(self):
    #     """
    #     Check the current state, and its description
    #     @return: current state, short description, long description
    #     @rtype: (int, string, string)
    #     """
    #
    #     state = self._state
    #     if state <= self.STATE_MACHINE_STAGE_MAX:
    #         desc_short, desc_long = self.STATE_MACHINE_STAGE_STRINGS[state]
    #     else:
    #         desc_short = 'Learning completed'
    #         desc_long = 'Learning completed'
    #
    #     return state, desc_short, desc_long

    # def get_VCD_data(self, clipids, vcd, name, dirpath_ECD_intermediate):
    #     """
    #     Retrieve data from VCD, store it into a directory with a name
    #     @param clipids: list of clipids
    #     @param vcd: VCD identifier
    #     @param name: name of dataset
    #     @param dirpath_ECD_intermediate:
    #     @return:
    #     """
    #
    #     # to be implemented
    #     pass

    def parameters_conf_load(self, filename_conf=None):
        """
        Return a parameter configuration.
        If filename is None, return a basic configuration which can be set in detail.
        Otherwise, load from json file.
        @filename: path to load configuration, a json file.
        @return: a configuration dictionary with various fields
        @rtype: dict
        """

        if filename_conf is None:
            conf = dict()
            conf['setting'] = dict()
            conf['setting']['eid'] = None  # Event ID
            conf['setting']['ename'] = None  # Event Name
            conf['setting']['feature'] = None  # Feature name (for VCD access)
            conf['setting']['str_kernel'] = None  # kernel name
            conf['setting']['use_approx_model'] = False # use fast approximate model, only applicable to certain kernel types
            conf['setting']['preprocess_kernel'] = False  # perform pre-processing before kernel computation
            conf['setting']['dirpath_PARAMETERS'] = None # location to find parameter search list for optimization

            # TODO: fully implement this later.
            # list of preprocessing functions to apply in order,
            # each list element is a tuple of (name, preprocess function, parameter load function, parameter_file)
            conf['setting']['preprocess_function_list'] = []

            conf['input'] = dict()
            conf['input']['dirpath_in_EVENT'] = None
            conf['input']['dirpath_in_BG'] = None
            conf['input']['data_EVENT'] = None
            conf['input']['data_BG'] = None
            conf['input']['clipids_EVENT'] = None
            conf['input']['clipids_BG'] = None
            conf['input']['labels_EVENT'] = None
            conf['input']['labels_BG'] = None
            conf['input']['clipids_EVENT'] = None
            conf['input']['clipids_BG'] = None
            conf['input']['cvids_train_EVENT'] = None
            conf['input']['cvids_train_BG'] = None
            conf['input']['cvids_test_EVENT'] = None
            conf['input']['cvids_test_BG'] = None

            conf['output'] = dict()
            conf['output']['dirpath_out_ECDG_EVENT'] = None
            conf['output']['dirpath_out_ECDG_BG'] = None
            conf['output']['dirpath_out_ECD'] = None

            conf['test'] = dict()
            conf['test']['dirpath_data'] = None
            conf['test']['dirpath_output'] = None
            conf['test']['report_margins_recounting'] = None
        else:
            with open(filename_conf, 'rb') as fin:
                conf = json.load(fin)

        return conf

    def parameter_conf_save(self, conf, filename):
        """
        Save current parameter configuration to a file
        @param conf: configuration
        @param filename:
        @type conf: dict
        @type filename: str
        """
        with open(filename, 'wb') as fout:
            json.dump(conf, fout, indent=4)

    def learn(self, conf_filepath=None, conf=None):
        """
        Learn classifier with simple single argument classifier.
        @param conf_filepath: configuration file
        @param conf: loaded configuration, usually loaded using 'parameters_conf_load'
        @type conf_filepath: optional str
        @type conf: optional dict
        """

        if conf_filepath is not None:
            conf = self.parameters_conf_load(filename_conf=conf_filepath)

        self.learn_detailed(conf['setting']['str_kernel'],
                            conf['output']['dirpath_out_ECD'],
                            conf['setting']['dirpath_PARAMETERS'],
                            conf['output']['dirpath_out_ECDG_EVENT'],
                            conf['output']['dirpath_out_ECDG_BG'],
                            dirpath_in_EVENT=conf['input']['dirpath_in_EVENT'],
                            dirpath_in_BG=conf['input']['dirpath_in_BG'],
                            data_EVENT=conf['input']['data_EVENT'],
                            data_BG=conf['input']['data_BG'],
                            clipids_EVENT=conf['input']['clipids_EVENT'],
                            clipids_BG=conf['input']['clipids_BG'],
                            labels_EVENT=conf['input']['labels_EVENT'],
                            labels_BG=conf['input']['labels_BG'],
                            cvids_train_EVENT=conf['input']['cvids_train_EVENT'],
                            cvids_train_BG=conf['input']['cvids_train_BG'],
                            cvids_test_EVENT=conf['input']['cvids_test_EVENT'],
                            cvids_test_BG=conf['input']['cvids_test_BG'],
                            preprocess_kernel=conf['setting']['preprocess_kernel'],
                            )

    def learn_detailed(self,
                       str_kernel,
                       dirpath_out_ECD,
                       dirpath_PARAMETERS,
                       dirpath_out_ECDG_EVENT, dirpath_out_ECDG_BG,
                       dirpath_in_EVENT=None, dirpath_in_BG=None,
                       data_EVENT=None, data_BG=None,
                       clipids_EVENT=None,clipids_BG=None,
                       labels_EVENT=None, labels_BG=None,
                       cvids_train_EVENT=None, cvids_train_BG=None,
                       cvids_test_EVENT=None, cvids_test_BG=None,
                       preprocess_kernel=True,
                       ):
        """
        Learn classifier with direct detailed arguments.
        For a version of learn function, which takes a single configuration input, see 'learn'.
        @param str_kernel: kernel name
        @param dirpath_out_ECD: location for final ECD learning output
        @param dirpath_PARAMETERS: dirpath to the trial parameter list file (depending on kernel type)
        @param dirpath_out_ECDG_EVENT: path to output the computed kernel matrices specific to EVENT Kit
        @param dirpath_out_ECDG_BG: path to output the computed kernel matrices related to BG only
        @param dirpath_in_EVENT: (optional) path to input data for event kit
        @param dirpath_in_BG: (optional) path to input data for BG
        @param data_EVENT: (optional) loaded data, either datapath_in_EVENT or this should be provided.
        @param data_BG:(optional) loaded data, either datapath_in_BG or this should be provided.
        @param cvids_train_EVENT: (optional) training cross validation splits for EVENT
        @param cvids_train_BG: (optional) training cross validation splits for BG
        @param cvids_test_EVENT: (optional) test cross validation splits for EVENT
        @param cvids_test_BG: (optional) test cross validation splits for BG
        @return:dictionary with matrices 'EVENTxEVENT', 'BGxEVENT', 'BGxBG' in numpy format,
                and 'clipids_EVENT', 'clipids_BG' in list format
        """

        # ask for loaded data return after kernel matrix computation, if needed, to save file access.
        flag_return_data = False
        if (data_EVENT is None) or (data_BG is None):
            flag_return_data = True

        # compute kernel matrices
        results = eag.compute_EAG_kernels2(str_kernel,
                                           dirpath_out_ECDG_EVENT, dirpath_out_ECDG_BG,
                                           dirpath_in_EVENT=dirpath_in_EVENT, dirpath_in_BG=dirpath_in_BG,
                                           data_EVENT=data_EVENT, data_BG=data_BG,
                                           clipids_EVENT=clipids_EVENT, clipids_BG=clipids_BG,
                                           flag_return_data=flag_return_data)

        # set up some learning parameters dependent on kernel type
        save_approx_model=False
        # HIK supports fast approx model
        if str_kernel == 'hik':
            save_approx_model = True

        # learn model
        if flag_return_data:
            # data_EVENT & data_BG are loaded during compute_EAG_kernels2, so, reuse them here.
            eag.run_EAG_libSVM(dirpath_out_ECDG_EVENT,
                               dirpath_out_ECD,
                               dirpath_in_EVENT, dirpath_in_BG,
                               dirpath_PARAMETERS,
                               clipids_EVENT=results['clipids_EVENT'],
                               clipids_BG=results['clipids_BG'],
                               labels_EVENT=labels_EVENT,
                               labels_BG=labels_BG,
                               data_EVENT=results['data_EVENT'],
                               data_BG=results['data_BG'],
                               kernel_EVENTxEVENT=results['kernel_EVENTxEVENT'],
                               kernel_BGxEVENT=results['kernel_BGxEVENT'],
                               kernel_BGxBG=results['kernel_BGxBG'],
                               flag_run_cv=True, cv_indices=None,
                               cvids_train_EventKit=cvids_train_EVENT,
                               cvids_train_BG=cvids_train_BG,
                               cvids_test_EventKit=cvids_test_EVENT,
                               cvids_test_BG=cvids_test_BG,
                               flag_run_full=True,
                               flag_recompute_best_param_cv=False,
                               path_kernel_BG_check=dirpath_out_ECDG_BG,
                               save_compact_model=True,
                               save_approx_model=save_approx_model)
        else:
            eag.run_EAG_libSVM(dirpath_out_ECDG_EVENT,
                               dirpath_out_ECD,
                               dirpath_in_EVENT, dirpath_in_BG,
                               dirpath_PARAMETERS,
                               clipids_EVENT=results['clipids_EVENT'],
                               clipids_BG=results['clipids_BG'],
                               labels_EVENT=labels_EVENT,
                               labels_BG=labels_BG,
                               data_EVENT=data_EVENT,
                               data_BG=data_BG,
                               kernel_EVENTxEVENT=results['kernel_EVENTxEVENT'],
                               kernel_BGxEVENT=results['kernel_BGxEVENT'],
                               kernel_BGxBG=results['kernel_BGxBG'],
                               flag_run_cv=True, cv_indices=None,
                               cvids_train_EventKit=cvids_train_EVENT,
                               cvids_train_BG=cvids_train_BG,
                               cvids_test_EventKit=cvids_test_EVENT,
                               cvids_test_BG=cvids_test_BG,
                               flag_run_full=True,
                               flag_recompute_best_param_cv=False,
                               path_kernel_BG_check=dirpath_out_ECDG_BG,
                               save_compact_model=True,
                               save_approx_model=save_approx_model)

        pass  # end of 'learn'









