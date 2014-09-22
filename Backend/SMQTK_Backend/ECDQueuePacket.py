"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import _abcoll
import uuid


class ECDQueuePacket (object):
    """
    Data packet that should be sent along the ECDController queue
    """

    def __init__(self, requester_uuid, req_type, event_type, clip_or_clips,
                 distance_kernel, classifier_config,
                 positives=(), negatives=(),
                 custom_collection=None, result_MID=None):
        """
        Create a work specification to be sent to the ECD Controller work queue.

        :param requester_uuid: The UUID of the requester agent.
        :type requester_uuid: uuid.UUID
        :param req_type: The request type. This should either be "search" or
            "learn", which will determine what action will be taken on the given
            clips.
        :type req_type: str
        :param event_type: The event type. This is the integer ID of the event.
        :type event_type: int or None
        :param clip_or_clips: A single clip id or iterable of clip ids that
            should be processed. When the ``req_type`` is learning, this must
            be an iterable of clip ids. A None value may be given to request
            graceful shutdowns.
        :type clip_or_clips: int or Iterable of int or None
        :param classifier_config: Custom classifier configuration
            dictionary to initialize agent processing with. This only does
            anything if this is the first work packet for an agent, which
            triggers initialization.
        :type classifier_config: dict
        :param positives: Iterable of clip IDs that are user-confirmed positive
            for this event type.
        :type positives: Iterable of int
        :param negatives: Iterable of clip IDs that are user-confirmed negative
            for this event type.
        :type negatives: Iterable of int
        :param custom_collection: If provided, results generated as a result of
            this work packet will be stored under this custom collection in the
            ECDController's configured database instead of the default
            collection.
        :type custom_collection: str
        :param result_MID: A custom model ID that fused scores during ranking
            will be stored under
        :type result_MID: str
        :param distance_kernel: DistanceKernel to use when asking for
            matrix components for ranking.
        :type distance_kernel: DistanceKernel

        """
        self.requester_uuid = requester_uuid
        self.request_type = str(req_type)
        self.event_type = event_type
        self.clips = clip_or_clips
        self.distance_kernel = distance_kernel
        self.positives = positives
        self.negatives = negatives
        self.classifier_config = classifier_config
        self.custom_collection = custom_collection
        self.result_MID = result_MID

        assert isinstance(requester_uuid, uuid.UUID), \
            'Invalid requester UUID object! (given: %s)' % str(requester_uuid)
        assert req_type in ('search', 'learn'), \
            "Invalid request type '%s'! Must be one of 'search' or 'learn'."

    @classmethod
    def from_packet(cls, packet):
        """
        Construct a new packet from a another packet.

        This is basically a copy method...

        :param packet: The other packet to construct from.
        :type packet: ECDQueuePacket

        """
        return ECDQueuePacket(packet.requester_uuid, packet.request_type,
                              packet.event_type, packet.clips,
                              packet.distance_kernel,
                              packet.classifier_config,
                              packet.positives, packet.negatives,
                              packet.custom_collection, packet.result_MID)

    def __repr__(self):
        if isinstance(self.clips, _abcoll.Iterable):
            # noinspection PyTypeChecker
            # reason -> I just checked for type
            clips_msg = "%s with %i elems" % (type(self.clips), len(self.clips))
        else:
            clips_msg = self.clips

        return "ECDQueuePacket{req_uuid: %s, req_typ: '%s', event_type: %s, " \
               "clips: %s, distnace_kernel: %s, " \
               "custom_classifier_config: %s, positives: %s, negatives: %s, " \
               "custom_collection: %s, fusion_label: %s}" \
               % (self.requester_uuid, self.request_type, self.event_type,
                  clips_msg, self.distance_kernel,
                  self.classifier_config, self.positives, self.negatives,
                  self.custom_collection, self.result_MID)


class ECDInterruptAgentPacket (ECDQueuePacket):
    """
    "Work" packet designating that a particular agent's workers should be
    immediately interrupted.
    """

    def __init__(self, requester_uuid):
        super(ECDInterruptAgentPacket, self).__init__(requester_uuid, 'learn',
                                                      0, None, None, None)

    def __repr__(self):
        return "ECDInterruptAgentPacket{req_uuid: %s}" % self.requester_uuid
