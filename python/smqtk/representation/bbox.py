import functools
import operator

import numpy

from smqtk.representation import SmqtkRepresentation


class AxisAlignedBoundingBox (SmqtkRepresentation):
    """
    Representation of an axis-aligned rectangular region within N-dimensional
    euclidean space.

    AxisAlignedBoundingBox currently does not support the concept of an
    "empty" region since it requires a min and max vertex to be set. We
    currently do not consider a zero-area region "empty" (represents a spatial
    point).

    The class attributes ``EQUALITY_ATOL`` and ``EQUALITY_RTOL`` are used as
    the tolerance attributes when comparing equality between two
    AxisAlignedBoundingBox instances. These may be changed on the class level
    to change the desired tolerance used at runtime. This cannot be changed
    on specific instances due to the use of python slots.

    Number of vertices of a hyper-cube: ``2**D``, where ``D`` is the number of
    dimensions.
    """

    __slots__ = 'min_vertex', 'max_vertex'

    # Same tolerance values as default on ``numpy.allclose``. These may be
    # changed on the class level to affect tolerance used when comparing
    # AxisAlignedBoundingBoxes at runtime.
    EQUALITY_ATOL = 1.e-8
    EQUALITY_RTOL = 1.e-5

    def __init__(self, min_vertex, max_vertex):
        """
        Create a new AxisAlignedBoundingBox from the given minimum and maximum
        euclidean-space vertex.

        :param collections.Sequence[int|float] min_vertex:
            Minimum bounding vertex of the (hyper) rectangle.
        :param collections.Sequence[int|float] max_vertex:
            Maximum bounding vertex of the (hyper) rectangle.

        :raises ValueError:
            When input vertices are not both 1 dimensional in
            shape, match in vertex dimensionality or if ``max_vertex`` is not
            greater-than-or-equal to ``min_vertex``.

        """
        # TODO: Default ``max_vertex`` to ``None`` to ease the creation of
        #       "points".
        self._set_vertices(min_vertex, max_vertex)

        if not (self.min_vertex.ndim == self.max_vertex.ndim == 1):
            raise ValueError("One or both vertices provided had more than "
                             "one array dimension (min_vertex.ndim == {}, "
                             "max_vertex.ndim == {})."
                             .format(self.min_vertex.ndim,
                                     self.max_vertex.ndim))
        if self.min_vertex.size != self.max_vertex.size:
            raise ValueError("Both vertices provided are not the same "
                             "dimensionality (min_vertex = {}, "
                             "max_vertex = {})."
                             .format(self.min_vertex.size,
                                     self.max_vertex.size))
        if not (self.max_vertex >= self.min_vertex).all():
            raise ValueError("The maximum vertex was not strictly >= the "
                             "minimum vertex."
                             .format(tuple(self.min_vertex),
                                     tuple(self.max_vertex)))

    def __str__(self):
        return "<{} [{}, {}]>"\
            .format(self.__class__.__name__, self.min_vertex, self.max_vertex)

    def __repr__(self):
        return "<{}.{} min_vertex={} max_vertex={}>"\
            .format(self.__class__.__module__, self.__class__.__name__,
                    self.min_vertex, self.max_vertex)

    def __hash__(self):
        return hash((tuple(self.min_vertex), tuple(self.max_vertex)))

    def __eq__(self, other):
        """
        Two bounding boxes are equal if the describe the same spatial area.

        :param AxisAlignedBoundingBox other:
            Other bounding box instance to test equality against.

        :return: If this and `other` describe the same spatial area.
        :rtype: bool
        """
        if not isinstance(other, AxisAlignedBoundingBox):
            return False
        # Should tolerances be parameterized in constructor?
        return (numpy.allclose(self.min_vertex, other.min_vertex,
                               rtol=self.EQUALITY_RTOL,
                               atol=self.EQUALITY_ATOL) and
                numpy.allclose(self.max_vertex, other.max_vertex,
                               rtol=self.EQUALITY_RTOL,
                               atol=self.EQUALITY_ATOL))

    def __ne__(self, other):
        return not (self == other)

    def __getstate__(self):
        return (
            self.min_vertex.tolist(),
            self.max_vertex.tolist(),
        )

    def __setstate__(self, state):
        self._set_vertices(*state)

    def _set_vertices(self, min_v, max_v):
        self.min_vertex = numpy.asarray(min_v)
        self.min_vertex.flags.writeable = False
        self.max_vertex = numpy.asarray(max_v)
        self.max_vertex.flags.writeable = False

    def get_config(self):
        return {
            'min_vertex': self.min_vertex.tolist(),
            'max_vertex': self.max_vertex.tolist(),
        }

    def intersection(self, other):
        """
        Get the AxisAlignedBoundingBox that represents the intersection between
        this box and the given ``other`` box.

        :param AxisAlignedBoundingBox other:
            An other box to get the intersection of. If there is no
            intersection ``None`` is returned.

        :return: An AxisAlignedBoundingBox instance if there is an intersection
            or None if there is no intersection.
        :rtype: AxisAlignedBoundingBox | None
        """
        inter_min_v = numpy.maximum(self.min_vertex, other.min_vertex)
        inter_max_v = numpy.minimum(self.max_vertex, other.max_vertex)
        # Constructor allows zero area boxes, but that is not a valid
        # intersection. Returning None if the minimum of the difference between
        # the intersection's maximum and minimum vertices is <= 0.
        if (inter_max_v - inter_min_v).min() <= 0:
            return None
        return AxisAlignedBoundingBox(inter_min_v, inter_max_v)

    @property
    def ndim(self):
        """
        :return: The number of dimensions this bounding volume covers.
        :rtype: int
        """
        # we know because of assert in constructor that both min and max vertex
        # must match in coordinate dimensionality, so we can draw this from
        # either.
        return self.min_vertex.size

    @property
    def deltas(self):
        """
        Get the lengths of this bounding box's edges along its dimensions.

        I.e. if this bounding box is 2-dimensional, this returns the [width,
        height] of the bounding box.

        :return: Array of dimension deltas.
        :rtype: numpy.ndarray[int|float]
        """
        return self.max_vertex - self.min_vertex

    @property
    def dtype(self):
        """
        :return: Most representative data type required to fully express this
            bounding box.
        :rtype: numpy.dtype
        """
        return self.deltas.dtype

    @property
    def hypervolume(self):
        """
        :return: The volume of this [hyper-dimensional] spatial bounding box.
            Unit of volume depends on the dimensionality of the vertices
            provided.
        :rtype: float
        """
        return functools.reduce(operator.mul,
                                self.max_vertex - self.min_vertex)
