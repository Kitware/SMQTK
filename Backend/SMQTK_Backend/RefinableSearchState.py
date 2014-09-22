"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

import logging
import uuid

from .SharedAttribute import SharedAttribute


class RefinableSearchState (object):
    """
    Object encapsulating the state of an IRQ session, including:
        - user feedback on video event classifications

    Pickle-able

    """

    _classifier_config = SharedAttribute()

    # TODO: Make thread safe without hindering on transportability or ability
    #       to hash. i.e. can't use Locks.

    pool_size_schedule = (10, 30, 75, 200, 1000)

    def __init__(self, event_type_or_prev_state, search_query=None,
                 distance_kernel=None, classifier_config=None, mdb_info=None,
                 result_model_id=None, state_uuid=None):
        """
        Initialize an IQR session state as a base state (given the UUID of the
        search) or as a new state in the chain by providing the previous
        state object,

        If this state is constructed from a parent state, we also inherit that
        parent's refinement state.

        When constructing the first state in a chain, the first state of a chain
        must be initialized with all parameters, not just the event ID.
        >>> RefinableSearchState(None)
        Traceback (most recent call last):
          ...
        ValueError: Initial state must be given the search query!

        Subsequent state along a chain should be constructed with the parent
        state as the sole parameter. The new state will inherit the parent
        state's values, except for the state UUID, which is globally unique
        across all states.:

        :param event_type_or_prev_state: The event type goal of the search or a
            previous search state to start from.
        :type event_type_or_prev_state: None or int or RefinableSearchState
        :param search_query: If given, the search UUID (this is the first state
            along a chain), the search query must be provided for historical
            purposes. If a previous state, this parameter will be ignored.
        :type search_query: None or *
        :param distance_kernel: The distance kernel interface instance to use
            for this search session.
        :type distance_kernel: DistanceKernel
        :param classifier_config: Only required for the first search state of a
            chain, and is the configuration dictionary for ECD classifiers.
        :type classifier_config: dict
        :param mdb_info: Database connection information. This will be copied
            because we will overwrite the collection parameter of our internal
            copy to a relevant value.
        :type mdb_info: DatabaseInfo

        """
        self._log = logging.getLogger('.'.join((self.__module__,
                                                self.__class__.__name__)))

        self._state_uuid = state_uuid or uuid.uuid4()

        # Maps defining user input positive and negative classifications
        # If a user declares that a video ID is positive for an event type, it
        # cannot be negative for that event type at the same time.
        #
        # Expected format for both maps:
        #   {
        #       <eventID>: set(<video_IDs>),
        #       ...
        #   }
        #
        #: :type: set of int
        self._positives = set()
        #: :type: set of int
        self._negatives = set()

        # slot to record "child" states when this state becomes the parent of
        # another state.
        self._child_state = None

        if isinstance(event_type_or_prev_state, (int, long, type(None))):
            self._log.info("Creating an initial state")

            if search_query is None:
                raise ValueError("Initial state must be given the search "
                                 "query!")
            if distance_kernel is None:
                raise ValueError("Initial state must be given a distance "
                                 "kernel interface instance!")
            if classifier_config is None:
                raise ValueError("Initial state must be given a classifier "
                                 "configuration dictionary!")
            if mdb_info is None:
                raise ValueError("Initial state must be given a MongoDB info "
                                 "object!")
            if result_model_id is None:
                raise ValueError("Initial state must be given a result storage "
                                 "model ID string!")

            self._parent_state = None
            self._search_uuid = uuid.uuid4()
            self._search_event_type = event_type_or_prev_state
            self._search_query = search_query
            self._distance_kernel = distance_kernel
            self._classifier_config = classifier_config
            self._mdb_info = mdb_info.copy()
            self._mdb_info.collection = str(self._state_uuid)
            self._result_mID = result_model_id

        elif isinstance(event_type_or_prev_state, RefinableSearchState):
            self._log.info("Extending previous state %s",
                           event_type_or_prev_state)
            self._parent_state = event_type_or_prev_state
            event_type_or_prev_state._child_state = self
            self._search_uuid = event_type_or_prev_state.search_uuid
            self._search_event_type = event_type_or_prev_state.search_event_type
            self._search_query = event_type_or_prev_state.search_query
            self._distance_kernel = event_type_or_prev_state.distance_kernel

            # Inherit refinement state
            # single layer container, so we don't need deepcopy
            #: :type: set of int
            self._positives = set(event_type_or_prev_state._positives)
            #: :type: set of int
            self._negatives = set(event_type_or_prev_state._negatives)

            # Inherit previous state's classifier config for now, but it will
            # need to be overwritten when new modes are trained.
            self._classifier_config = event_type_or_prev_state.classifier_config

            self._mdb_info = event_type_or_prev_state.mdb_info.copy()
            self._mdb_info.collection = str(self._state_uuid)
            self._result_mID = event_type_or_prev_state.result_mID

        else:
            raise ValueError("Invalid parameter given to constructor (%s: %s)."
                             % (type(event_type_or_prev_state),
                                event_type_or_prev_state))

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, RefinableSearchState) and self.uuid == other.uuid:
            return True
        return False

    def __ne__(self, other):
        return not (self == other)

    @property
    def search_uuid(self):
        """
        The search UUID this state is associated with.

        States inheriting from one another will share the same search UUID.
        States not on the same inheritance chain will not have the same search
        UUID.

        :return: The UUID of the search this state is associated with.
        :rtype: uuid.UUID

        """
        return self._search_uuid

    @property
    def search_event_type(self):
        """
        The event type context of this search state.

        :return: The event type of the parent search.
        :rtype: int or None
        """
        return self._search_event_type

    @property
    def search_query(self):
        return self._search_query

    @property
    def distance_kernel(self):
        """
        :rtype: DistanceKernel
        """
        return self._distance_kernel

    @property
    def uuid(self):
        """
        The unique UUID of this search state. No two states will share the same
        UUID.

        :return: the UUID of this state
        :rtype: uuid.UUID

        """
        return self._state_uuid

    @property
    def parent_state(self):
        """
        Return the parent state of this search state. If there is no parent,
        meaning that this is the first state of the chain, then None is
        returned.

        :return: The parent search state of this state. It may be None if this
            state doesn't have a parent.
        :rtype: RefinableSearchState or None

        """
        return self._parent_state

    @property
    def num_parents(self):
        """
        :return: The number of parent states of this state along the sequence.
        :rtype: int

        """
        p_states = 0
        state = self.parent_state
        while state is not None:
            p_states += 1
            state = state.parent_state
        return p_states

    @property
    def child_state(self):
        """
        Return the child of this state. This state will not have a child until
        it has become the parent of a state, aka passed to the constructor when
        creating a new RefinableSearchState.

        :return: The child search state of this state. It may be None.
        :rtype: RefinableSearchState or None

        """
        return self._child_state

    @property
    def num_children(self):
        """
        :return: The number of child states of this state along the sequence.
        :rtype: int

        """
        c_states = 0
        state = self.child_state
        while state is not None:
            c_states += 1
            state = state.child_state
        return c_states

    @property
    def positives(self):
        """
        :return: Positive user feedback of videos for events.
        :rtype: set of int
        """
        return frozenset(self._positives)

    @property
    def negatives(self):
        """
        :return: Negative user feedback of videos for events.
        :rtype: set of int
        """
        return frozenset(self._negatives)

    @property
    def classifier_config(self):
        return self._classifier_config

    @classifier_config.setter
    def classifier_config(self, value):
        self._classifier_config = value

    @property
    def mdb_info(self):
        """
        :rtype: DatabaseInfo
        """
        return self._mdb_info

    @property
    def result_mID(self):
        """
        :rtype: str
        """
        return self._result_mID

    def register_positive_feedback(self, vID_or_IDs):
        """
        Register the given video ID ``vID`` as a positive match to the search.

        :param vID_or_IDs: Video integer ID key or keys.
        :type vID_or_IDs: int or Iterable of int

        """
        # Convert to an iterable if a single int
        if not hasattr(vID_or_IDs, '__iter__'):
            # make sure its int-able
            assert isinstance(int(vID_or_IDs), int)
            vID_or_IDs = (vID_or_IDs,)
        self._positives.update(vID_or_IDs)

        # If this video was previously classified negatively for this event
        # type, remove that negative entry. Can't be both positive and
        # negative for the same event type at the same time.
        self.remove_negative_feedback(vID_or_IDs)

    def register_negative_feedback(self, vID_or_IDs):
        """
        Register the given video ID ``vID`` as negatively match to the search

        :param vID_or_IDs: Video integer ID key.
        :type vID_or_IDs: int or Iterable of int

        """
        # Convert to an iterable if a single int
        if not hasattr(vID_or_IDs, '__iter__'):
            # make sure its int-able
            assert isinstance(int(vID_or_IDs), int)
            vID_or_IDs = (vID_or_IDs,)
        self._negatives.update(vID_or_IDs)

        # If this video was previously classified positively for this event
        # type, remove that positive entry. Can't be both positive and
        # negative for the same event type at the same time.
        self.remove_positive_feedback(vID_or_IDs)

    def remove_positive_feedback(self, vID_or_IDs):
        """
        Remove the given video ID as a positive match to the given event ID. If
        the pairing doesn't exist in the positives registry, this does nothing.

        :param vID_or_IDs: Video integer ID key
        :type vID_or_IDs: int or Iterable of int

        """
        # Convert to an iterable if a single int
        if not hasattr(vID_or_IDs, '__iter__'):
            # make sure its int-able
            assert isinstance(int(vID_or_IDs), int)
            vID_or_IDs = (vID_or_IDs,)

        self._positives.difference_update(vID_or_IDs)

    def remove_negative_feedback(self, vID_or_IDs):
        """
        Remove the given video ID as a negative match to the given event ID. If
        the pairing doesn't exist in the negatives registry, this does nothing.

        :param vID_or_IDs: Video integer ID key
        :type vID_or_IDs: int

        """
        # Convert to an iterable if a single int
        if not hasattr(vID_or_IDs, '__iter__'):
            # make sure its int-able
            assert isinstance(int(vID_or_IDs), int)
            vID_or_IDs = (vID_or_IDs,)

        self._negatives.difference_update(vID_or_IDs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
