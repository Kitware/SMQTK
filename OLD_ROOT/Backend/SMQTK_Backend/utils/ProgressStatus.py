"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

class ProgressStatus (object):
    """
    Encapsulation of progress information for a Controller component.
    ProgressStatus objects may also be added together to create new
    ProgressStatus objects representing a group of processes.
    """

    ### Status identifiers
    # Processing not started. Implies 0% completion
    NOT_STARTED = 0
    # Processing currently active and running
    RUNNING = 1
    # Processing complete. Implies 100% completion.
    COMPLETE = 2
    # Processing failed. Implies progress < 100%.
    FAILED = 3
    # Active processing **not** occurring, but what has been done is not
    # complete nor failed.
    PARTIAL_PROGRESS = 4

    #: :param: dict mapping a status enumeration to a string representation
    #: :type: dict of (int, str)
    STAT_STR_MAP = {
        NOT_STARTED: 'not-started',
        RUNNING: 'running',
        COMPLETE: 'complete',
        FAILED: 'failed',
        PARTIAL_PROGRESS: 'partial-progress',
    }

    def __init__(self, component_name,
                 status=0,  # corresponding to "NOT_STARTED"
                 progress_percent=0.0):
        """
        Initialize a progress status object.

        At least a name is required to construct a ProgressStatus object:
            >>> ProgressStatus()
            Traceback (most recent call last):
                ...
            TypeError: __init__() takes at least 2 arguments (1 given)

            >>> ProgressStatus('foo')
            ProgressStatus(foo, not-started, 0.000000)

            >>> ProgressStatus('bar', ProgressStatus.COMPLETE)
            ProgressStatus(bar, complete, 0.000000)

            >>> ProgressStatus('baz', ProgressStatus.FAILED, 0.5936)
            ProgressStatus(baz, failed, 0.593600)

        :param component_name: The name of the component this status represents.
        :type component_name: str
        :param status: The run status of this component.
        :type status: int
        :param progress_percent: Decimal percentage, if available, of the amount
            this component has completed.
        :type progress_percent: float

        """
        self.name = str(component_name)
        self.status = status
        self.progress = progress_percent

        # as we add process status objects together, we need to know how many
        # were taken into account so we can calculate percentages correctly
        self._p_repd = 1

    def __add__(self, other):
        """
        Add two ProgressStatus objects together, creating a union of the two
        statuses.

        Two objects representing single processes may be added together to
        create a new object representing the status of the two in combination:
            >>> a = ProgressStatus('foo', ProgressStatus.RUNNING, 0.20)
            >>> b = ProgressStatus('bar', ProgressStatus.COMPLETE, 1.0)
            >>> c = a + b
            >>> c
            ProgressStatus(combined, running, 0.600000)

        Note that the status of the new ProgressStatus is 'running' still.
        Depending on the statuses of the two ProgressStatus objects added
        together, a sensible new status will be determined according to the
        following table:
            - let:
                - NS = NOT_STARTED
                - R  = RUNNING
                - C  = COMPLETE
                - F  = FAILED
                - PP = PARTIAL_PROGRESS

            +----+----+---+----+----+----+
            |    | NS | R | C  | F  | PP |
            +----+----+---+----+----+----+
            | NS | NS |   |    |    |    |
            +----+----+---+----+----+----+
            | R  | R  | R |    |    |    |
            +----+----+---+----+----+----+
            | C  | PP | R | C  |    |    |
            +----+----+---+----+----+----+
            | F  | PP | R | F  | F  |    |
            +----+----+---+----+----+----+
            | PP | PP | R | PP | PP | PP |
            +----+----+---+----+----+----+

        ProcessStatus adding may extend to grouping more than just 2 statuses.
        Generated objects remember the number of processes they represent, so
        percentage completions are accurately maintained.
            >>> d = ProgressStatus('baz', ProgressStatus.FAILED, 0.30)
            >>> c + d
            ProgressStatus(combined, running, 0.500000)
            >>> e = ProgressStatus('arg', ProgressStatus.NOT_STARTED)
            >>> d + e
            ProgressStatus(combined, partial-progress, 0.150000)

        :param other: The other ProgressStatus object to add to this one.
        :type other: ProgressStatus
        :return: A new ProgressStatus object representing the union of the two
            added together.
        :rtype: ProgressStatus

        """
        if not isinstance(other, ProgressStatus):
            raise TypeError("Can only add ProgressStatus object with another "
                            "ProgressStatus object.")

        new_name = 'combined'

        # Determine new combined progress percentage
        s = ((self._p_repd * self.progress) + (other.progress * other._p_repd))
        percent = s / (self._p_repd + other._p_repd)

        # Determine combined status
        # noinspection PySetFunctionToLiteral
        # reason -> Not valid in python 2.6.x
        stats = set((self.status, other.status))
        if len(stats) == 1:  # meaning that they're the same
            status = stats.pop()
        elif ProgressStatus.RUNNING in stats:
            status = ProgressStatus.RUNNING
        elif ProgressStatus.COMPLETE in stats \
                and ProgressStatus.FAILED in stats:
            status = ProgressStatus.FAILED
        else:
            status = ProgressStatus.PARTIAL_PROGRESS

        p = ProgressStatus(new_name, status, percent)
        p._p_repd = (self._p_repd + other._p_repd)
        return p

    def __repr__(self):
        return "ProgressStatus(%s, %s, %f)" % (self.name,
                                               self.STAT_STR_MAP[self.status],
                                               self.progress)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
