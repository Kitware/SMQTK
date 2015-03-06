"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import numpy

# ABSTRACT class for MER

def recount_abstract(info, func_MER, identifier, params_mer):
    """ Abstract MER function

    Return recounting results for 'identifier' based on info, using MER function 'func_mer'.
    Then, separate visualization routine is supposed to deliver the information to users.

    @param info: information needed for MER, possibly a combination of MED results/MED by-products etc.
    @param func_MER: func_MER(info, identifier) generates MER output
    @param identifier: an identifier to select proper information from info
    @param params_mer: parameters for MER. examples include MER granularity etc.
    """

    recounting = func_MER(info, identifier, params_mer)
    return recounting

def func_MER_margins_mer_SumOfTopBins(info, identifier, params_mer):
    """
    Basic MER function to select key dimensions (from clip-level features), then,
    select frames with high dose of such key dimensions if detailed frame/segment-wise data is given.
    @param info: data, functions to access data in various ways
    @param identifier: identfier for the specific video clip to be recounted
    @param params_mer: MER parameters specifying the # key dimensions and # of final MER evidences per clip
    @return: recounting for identifier data, with params_mer['n_evidences'] of frames with params_mer['n_bins'] of multiple key evidences
    @type info: dictionary
        info['margins_mer']: dictionary, with identifiers as key and clip-level feature (or bin-wise classifier contribution) as data
        info['func_get_data_detailed_data']: (optional) a function. given an identifier, returns iterable (detailed) data across frames.
        info['func_get_detailed_info_time']: (optional)a function. given an item from iterable detailed data, extract timestamp information
        info['time_unit']: (optional) type: string, time unit, 'sec', 'frame' etc.
        info['func_get_detailed_info_bin']: (optional)a function. given an item from iterable detailed data & bin idx, extract value
        info['bin_names']: optional dictionary. Bin idx is key, and names are contents.
    @type params_mer: dictionary
        params_mer['n_bins']: specifies how many top contributing bins will be used for MER
        params_mer['n_per_bin_segments']: specifies
        params_mer['n_segments']: (optional) specifies how many frames will be used for MER, per clip
    @return:
    @rtype: (total n_segments of) sorted list of (timestamp, list of (idx_bin, value), sum of values)
    """

    # output
    output = dict()

    # --------- Top evidence bin selection at the clip level

    margins_mer = (info['margins_mer'])[identifier]

    # clip-level recounting
    margins_mer_sorted = sorted(enumerate(margins_mer), key=lambda x: x[1])[::-1]  # most contributing margins first
    if 'n_bins' in params_mer:  # find top bins with most margin contributions at the clip-level
        n_bins = params_mer['n_bins']
        margins_mer_sorted = margins_mer_sorted[:n_bins]  # number of top dimensions to consider

    # Add clip-level MER to output
    mer_clip = dict()
    mer_clip['evidences'] = []

    for (idx, iclip_value) in margins_mer_sorted:
        e = dict()
        if 'bin_names' in info:
            e['name'] = info['bin_names'][idx]
        if 'group' in info:
            e['group'] = info['group']
        e['idx'] = idx
        e['value'] = iclip_value

        mer_clip['evidences'].append(e)

    output['clip'] = mer_clip

    if 'func_get_data_detailed_data' in info:

        # add time unit information
        if (not 'time_unit' in output) and ('time_unit' in info):
            output['time_unit'] = info['time_unit']

        # load detailed data at segment/frame-level
        data_mer_detailed = info['func_get_data_detailed'](identifier)

        func_get_detailed_info_time = info['func_get_detailed_info_time']  # function to access particular time
        func_get_detailed_info_bin = info['func_get_detailed_info_bin']  # function to access particular bin value

        # idxs of top clip-level evidences
        idxs_selected = [idx for (idx, _) in margins_mer_sorted]

        # --------- For each clip-level evidence, find corresponding top scoring segment locations

        for (ie, e) in enumerate(mer_clip['evidences']):
            _time_values = [(func_get_detailed_info_time(_data),
                             func_get_detailed_info_bin(_data, e['idx'])
                             )
                            for _data in data_mer_detailed]
            _time_values_sorted = sorted(_time_values, key=lambda x: x[1])[::-1]  # decreasing order
            _time_to_select = min(params_mer['n_per_bin_segments'], len(_time_values_sorted))
            _timestamps = map(lambda x: {'timestamp': x[0], 'value': x[1]}, _time_values_sorted[:_time_to_select])
            output['clip']['evidences'][ie]['top_timestamps'] = _timestamps

        """
        This block is commented out because we do not do multimedia MER/triage anymore.

        # --------- Find temporal locations with top scores (sum of high-ranking bins)

        #  [ (timeinfo1, [(idx1, value1), (idx2, value2), ...] ),
        #    (timeinfo2, [(idx1, value1), )idx2, value2), ...] ),
        #  ] # for video with identifier
        mer_detailed = [(func_get_detailed_info_time(_data),
                         [(_idx, func_get_detailed_info_bin(_data, _idx)) for _idx in idxs_selected]
                         )
                         for _data in data_mer_detailed
                        ]

        #  [ (timeinfo1, [(idx1, value_sorted1), (idx2, value2_sorted), ..], sum_of_values_time1),
        #    (timeinfo2, [(idx1, value_sorted1), (idx2, value2_sorted), ..], sum_of_values_time2), ...
        #  ]
        mer_detailed_sums = [(info_time,
                              sorted(info_values, key=lambda x:x[1])[::-1],
                              reduce(lambda x, y: x+y[1], info_values, 0)
                              )
                             for (info_time, info_values) in mer_detailed
                             ]

        # sort frames based on their significance
        mer_detailed_sums_sorted = sorted(mer_detailed_sums, key=lambda x: x[-1])[::-1]
        n_evidences = params_mer['n_segments']  # number of evidence segments per detection
        if n_evidences < len(mer_detailed_sums_sorted):
            mer_detailed_sums_sorted = mer_detailed_sums_sorted[:n_evidences]

        mer_detailed_sums_sorted_dict = []
        for (info_time, info_values, score_sum) in mer_detailed_sums_sorted:
            mer_detailed_sums_sorted_dict.append({'timestamp': info_time,
                                                  'score_segment': score_sum,
                                                  'evidences': [{'idx': idx,
                                                                 'name': info['bin_names'][idx],
                                                                 'value': value
                                                                 } for (idx, value) in info_values
                                                                ]
                                                  })
        output['top_segments'] = mer_detailed_sums_sorted_dict
        """

    return output



