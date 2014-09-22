"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import _abcoll
import json
import re

from ... import ECDWorkerBaseProcess
from ....ECDStore.ECDStoreElement import ECDStoreElement
from ....utils.jsmin import jsmin


class SVMClassifier_searcher (ECDWorkerBaseProcess):

    def _process(self, work_packet):
        """
        Perform work on the given ECDQueuePacket of data, returning
        ECDStoreElement objects for storage.

        :param work_packet: A data packet
        :type work_packet: ECDQueuePacket
        :return: ECDStoreElement objects containing the data to store.
        :rtype: ECDStoreElement

        """
        # Doing things
        if isinstance(work_packet.clips, int):
            clip_ids = (work_packet.clips,)
        elif isinstance(work_packet.clips, _abcoll.Iterable):
            clip_ids = sorted(tuple(work_packet.clips))  # pesky sets...
        else:
            self._log.warn("[%s] No valid clip ID or IDs given! (given: '%s') "
                           "Nothing to be done.",
                           self.name, str(work_packet.clips))
            return

        # If we have a valid model file, attempt to load it. For this dummy
        # process it will be pre-computed rankings for clips IDs.
        pre_comp_results = {}
        if self._model_file_path:
            self._log.info("Using model file")
            id_re = re.compile("HVC(\d+)")
            with open(self._model_file_path) as model_file:
                raw_model = json.loads(jsmin(model_file.read()))
                pre_comp_results.update(
                    (int(id_re.match(e['id']).group(1)), e['score'])
                    for e in raw_model['clips']
                )

        #: :type: list of ECDStoreElement
        elems = []
        p_set = set()  # set of + clip indices
        n_set = set()  # set of - clip indices

        #print "ClipIDs:", clip_ids

        for i, clip_id in enumerate(clip_ids):
            e = ECDStoreElement(self.name, clip_id, 0.0)

            if clip_id in work_packet.positives:
                e.probability = 1.0
                p_set.add(i)
            elif clip_id in work_packet.negatives:
                e.probability = 0.0
                n_set.add(i)
            elif pre_comp_results:
                e.probability = pre_comp_results.get(clip_id, 0.0)
            else:
                e.probability = 1.0 - (float(clip_id) / 1000000)

            self._log.debug("[%s] Creating ECDStoreElement for clip %d "
                            "(prob: %f)",
                            e.model_id, e.clip_id, e.probability)
            elems.append(e)

        # swap clip probabilities on the sides of adjudicated clip indices
        num_elems = len(elems)

        for i in p_set:
            i_l = (i+1) % num_elems
            i_h = (i+2) % num_elems
            if not n_set.intersection((i_l, i_h)):
                #print "(POS) Swapping probabilities for indices %d and %d" \
                #      % (i_l, i_h)
                p = elems[i_l].probability
                elems[i_l].probability = elems[i_h].probability
                elems[i_h].probability = p

        for i in n_set:
            i_l = (i-2) % num_elems
            i_h = (i-1) % num_elems
            if not p_set.intersection((i_l, i_h)):
                #print "(NEG) Swapping probabilities for indices %d and %d" \
                #      % (i_l, i_h)
                p = elems[i_l].probability
                elems[i_l].probability = elems[i_h].probability
                elems[i_h].probability = p

        return elems
