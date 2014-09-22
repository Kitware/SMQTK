"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import _abcoll
import abc
import cPickle
import time

from .. import ECDWorkerBaseProcess
from ...ECDStore.ECDStoreElement import ECDStoreElement
from ...ECDStore.errors import ECDNoElementError

from EventContentDescriptor.iqr_modules import iqr_model_test


class BaseFusion (ECDWorkerBaseProcess):
    """
    Fusion process that, given a classifier configuration structure and a model
    file, runs and fuses scores from various base ECD workers with various
    configurations.

    Implementations of this base class define how score fusion occurs.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, model_path, vcd_controller, ecd_controller,
                 child_classifiers, storage_overwrite=False,
                 skip_bad_packets=True, fetch_timeout=120):
        """
        Same initialization as the ECDWorkerBaseProcess, but with an additional
        classifier configuration object. We also store the ecd_controller's work
        queue so that we may transfer work messages to it for underlying, base
        classifier worker messages to it.

        NOTES:
            - ??? Fusion has no configuration file ???
            - The classifier configuration object is a dictionary of a
              pre-defined format. See the ECDController documentation.

        :param model_path: Path to a model file.
        :type model_path: str
        :param vcd_controller: The VCD Controller object so we can know about
            the VCD store and work queue.
        :type vcd_controller: SMQTK_Backend.VCDController.VCDController
        :param ecd_controller: The ECD Controller object so we can know about
            the ECD store and work queue.
        :type ecd_controller: SMQTK_Backend.ECDController.ECDController
        :param child_classifiers: The model IDs that this fusion process is
            dependent on for data to fuse. We make a shallow copy of whatever is
            passed to this.
        :type child_classifiers: Iterable of ECDWorkerBaseProcess
        :param storage_overwrite: Allow this process to overwrite data in the
            configured ECD storage space, as found through the controller.
        :type storage_overwrite: bool
        :param skip_bad_packets: Warn about, but otherwise ignore bad objects
            found in the work queue.
        :type skip_bad_packets: bool
        :param fetch_timeout: Timeout period when waiting for scores to be
            populated in the ECD store. This should be set to above the longest
            period expected + worker initialization period.
        :type fetch_timeout: float

        """
        super(BaseFusion, self).__init__(name, model_path, None,
                                         vcd_controller, ecd_controller,
                                         storage_overwrite, skip_bad_packets)

        self._fetch_timeout = fetch_timeout

        # Not all fusion processes require that there by child workers. Those
        # that do will just generate invalid scores.
        #assert child_classifiers, \
        #    "No child classifiers set. Must depend on someone! Fusion is " \
        #    "needy like that."

        # Making a shallow copy
        self._child_classifiers = tuple(child_classifiers)


class SimpleAvgFusion (BaseFusion):

    def _process(self, work_packet):
        """
        Simple wait on expected processing to garner results and then averaging
        (positive) scores found (negative indicating error happened).

        :type work_packet: ECDQueuePacket

        """
        self._log.info("Starting fusion for event type %s",
                       work_packet.event_type)

        wait_interval = 0.1  # seconds

        if isinstance(work_packet.clips, int):
            clip_ids = (work_packet.clips,)
        elif isinstance(work_packet.clips, _abcoll.Iterable):
            clip_ids = tuple(work_packet.clips)
        else:
            self._log.warn("No valid clip ID or IDs given! (given: '%s')",
                           str(work_packet.clips))
            clip_ids = ()

        fusion_mid = self.name  # as per instructions
        ecd_store = self._ecdc.store(work_packet.custom_collection)

        for clip in clip_ids:
            # Check if there is already a fused score stored. If so, skip for
            # this clip id.
            try:
                ecd_store.get(fusion_mid, clip)
                self._log.info("Found existing fused score for MID '%s' for "
                               "clip %d. Skipping to next clip.",
                               fusion_mid, clip)
                continue
            except ECDNoElementError:
                # OK, continue with fusion then.
                self._log.debug("No existing fused score found for clip %d for "
                                "event %s. Performing fusion.",
                                clip, work_packet.event_type)

            clip_scores = []
            for classifier in self._child_classifiers:
                not_found = True
                s = time.time()
                while not_found:
                    try:
                        elem = ecd_store.get(classifier.name, clip)
                        not_found = False
                        if elem.probability >= 0:
                            clip_scores.append(elem.probability)
                            self._log.debug(".. Found score for classifier "
                                            "'%s': %f",
                                            elem.model_id, elem.probability)
                    except ECDNoElementError:
                        # Ignore. Keeps not_found as True.
                        pass

                    if time.time() - s >= self._fetch_timeout:
                        raise RuntimeError("!! Exceeded timeout wait period "
                                           "when looking for scores for '%s'"
                                           % classifier.name)
                    elif not_found:
                        time.sleep(wait_interval)

            if clip_scores:
                fused_elem = ECDStoreElement(
                    fusion_mid, clip,
                    sum(clip_scores) / float(len(clip_scores))
                )
            else:
                # I don't think that this will ever be reached.
                self._log.info("!! No valid clip scores encountered (clip: %d)",
                               clip)
                fused_elem = ECDStoreElement(fusion_mid, clip, -1)

            self._log.debug("=> Fused element: %s", fused_elem)
            ecd_store.store(fused_elem, self._ecdc_store_overwrite)

        # Updated ECDStore here, so not returning any elements for
        # storage/update.
        self._log.info("Fusion complete")


class IqrDemoFusion (BaseFusion):

    def _process(self, work_packet):
        """
        Special "fusion" process for IQR demo sake. We will perform all clip
        ranking here in this method.

        :type work_packet: ECDQueuePacket

        """
        # Split the overloaded "model file" path (::)
        model_file_path, svIDs_file_path = self._model_file_path.split('::')
        self._log.info("[%s] Model file path: %s", work_packet.requester_uuid,
                       model_file_path)
        self._log.info("[%s] svIDs file path: %s", work_packet.requester_uuid,
                       svIDs_file_path)

        with open(svIDs_file_path) as ifile:
            sv_clip_ids = cPickle.load(ifile)

        idx2cid_row, idx2cid_col, kernel_test = \
            work_packet.distance_kernel.extract_rows(*sv_clip_ids)

        # Testing/ranking call
        #   Passing the array version of the kernel sub-matrix. The returned
        #   output['probs'] type matches the type passed in here, and using an
        #   array makes syntax cleaner.
        self._log.info("[%s] Testing clip IDs", work_packet.requester_uuid)
        output = iqr_model_test(model_file_path, kernel_test.A, idx2cid_col)

        # Parallel lists pairing the clip IDs to their rankings
        fusion_mid = self.name
        elements = []
        probability_map = dict(zip(output['clipids'], output['probs']))
        for cid in work_packet.clips:
            #self._log.info("[%s] Storing cID/prob pair: %i %f",
            #               work_packet.requester_uuid, cid, prob)
            elements.append(ECDStoreElement(fusion_mid, cid,
                                            probability_map[cid]))
            # TODO: If ``cid`` in negative/background set, force 0.0 prob?
        ecd_store = self._ecdc.store(work_packet.custom_collection)
        ecd_store.store(elements, self._ecdc_store_overwrite)
