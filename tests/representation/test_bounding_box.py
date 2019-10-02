import pickle
import random

import numpy
import pytest

# noinspection PyUnresolvedReferences
from six.moves import mock  # move defined in ``tests`` top module

from smqtk.representation.bbox import AxisAlignedBoundingBox
from smqtk.utils.configuration import configuration_test_helper


###############################################################################
# pytest Fixtures

@pytest.fixture(params=[tuple, list, numpy.array],
                ids=[None, None, 'numpy.array'])
def seq_type(request):
    """
    Enumerate via fixture sequence types of coordinates to test with
    AxisAlignedBoundingBox.

    ``request`` is a special parameter name required by pytest.
    """
    return request.param


@pytest.fixture(params=[1, 2, 3, 4, 32])
def ndim(request):
    """
    Enumerate coordinate dimensionality via pytest fixture.

    ``request`` is a special parameter name required by pytest.
    """
    return request.param


###############################################################################
# Tests

def test_configuration():
    """ test standard instance configuration """
    inst = AxisAlignedBoundingBox([0, 1], [1, 2])
    for i in configuration_test_helper(inst):  # type: AxisAlignedBoundingBox
        numpy.testing.assert_allclose(i.min_vertex, (0, 1))
        numpy.testing.assert_allclose(i.max_vertex, (1, 2))


def test_bbox_set_vertices(ndim, seq_type):
    """
    Test constructing an AxisAlignedBoundingBox with ``ndim`` coordinates.
    """
    minv = [random.randint(0, 9) for _ in range(ndim)]
    maxv = [random.randint(10, 19) for _ in range(ndim)]
    minv_s = seq_type(minv)
    maxv_s = seq_type(maxv)

    # Mock instance so as to not actually hit __init__ method.
    m_bb = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    # Invoke private method, which should set attributes onto `m_bb`.
    AxisAlignedBoundingBox._set_vertices(m_bb, minv_s, maxv_s)

    assert isinstance(m_bb.min_vertex, numpy.ndarray)
    assert isinstance(m_bb.max_vertex, numpy.ndarray)
    numpy.testing.assert_allclose(m_bb.min_vertex, minv)
    numpy.testing.assert_allclose(m_bb.max_vertex, maxv)


def test_bbox_set_vertices_maintain_type_int():
    """
    Test that ndarray dtypes inherit from input integer values explicitly.
    """
    # Integer input coordinates (1d)
    minv = [0]
    maxv = [1]

    # Mock instance so as to not actually hit __init__ method.
    m_bb = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    # Invoke private method, which should set attributes onto `m_bb`.
    AxisAlignedBoundingBox._set_vertices(m_bb, minv, maxv)

    # BOTH vertices should be integer since input coordinates are integers.
    assert issubclass(m_bb.min_vertex.dtype.type, numpy.integer)
    assert issubclass(m_bb.max_vertex.dtype.type, numpy.integer)


def test_bbox_set_vertices_maintain_type_float():
    """
    Test that ndarray dtypes inherit from input float values explicitly.
    """
    # Float input coordinates (1d)
    minv = [0.]
    maxv = [1.]

    # Mock instance so as to not actually hit __init__ method.
    m_bb = mock.MagicMock(spec_set=AxisAlignedBoundingBox)
    # Invoke private method, which should set attributes onto `m_bb`.
    AxisAlignedBoundingBox._set_vertices(m_bb, minv, maxv)

    # BOTH vertices should be integer since input coordinates are integers.
    assert issubclass(m_bb.min_vertex.dtype.type, numpy.float)
    assert issubclass(m_bb.max_vertex.dtype.type, numpy.float)


def test_bbox_set_vertices_maintain_type_mixed():
    """
    Test that ndarray dtypes inherit from mixed float and integer values
    explicitly.
    """
    # Mock instance so as to not actually hit __init__ method.
    m_bb = mock.MagicMock(spec_set=AxisAlignedBoundingBox)

    # Integer/Float coordinates (3d)
    minv = [0, 1, 2]  # integer
    maxv = [1, 2.0, 3]  # float
    AxisAlignedBoundingBox._set_vertices(m_bb, minv, maxv)
    assert issubclass(m_bb.min_vertex.dtype.type, numpy.integer)
    assert issubclass(m_bb.max_vertex.dtype.type, numpy.float)

    # Float/Integer coordinates (3d)
    minv = [0, 1, 2.0]  # float
    maxv = [1, 2, 3]  # integer
    AxisAlignedBoundingBox._set_vertices(m_bb, minv, maxv)
    assert issubclass(m_bb.min_vertex.dtype.type, numpy.float)
    assert issubclass(m_bb.max_vertex.dtype.type, numpy.integer)


def test_bbox_construction_incongruous_shape():
    """
    Test that construction fails when one or both input coordinates are not a
    single array dimension (i.e. multi-dimensional shape in numpy.array
    parlance).
    """
    minp_1dim = (0,)
    minp_2dim = ((0,), (1,))

    maxp_1dim = (1,)
    maxp_2dim = ((1,), (2,))

    with pytest.raises(ValueError, match=r"One or both vertices provided had "
                                         r"more than one array dimension "
                                         r"\(min_vertex\.ndim == 2, "
                                         r"max_vertex\.ndim == 1\)\."):
        # noinspection PyTypeChecker
        AxisAlignedBoundingBox(minp_2dim, maxp_1dim)
    with pytest.raises(ValueError, match=r"One or both vertices provided had "
                                         r"more than one array dimension "
                                         r"\(min_vertex\.ndim == 1, "
                                         r"max_vertex\.ndim == 2\)\."):
        # noinspection PyTypeChecker
        AxisAlignedBoundingBox(minp_1dim, maxp_2dim)
    with pytest.raises(ValueError, match=r"One or both vertices provided had "
                                         r"more than one array dimension "
                                         r"\(min_vertex\.ndim == 2, "
                                         r"max_vertex\.ndim == 2\)\."):
        # noinspection PyTypeChecker
        AxisAlignedBoundingBox(minp_2dim, maxp_2dim)


def test_bbox_construction_incongruous_dimensionality():
    """
    Test that construction fails if min and max coordinates are not of the same
    dimensionality.
    """
    minp = (0, 0, 0)
    maxp = (1, 1, 1, 1)

    with pytest.raises(ValueError,
                       match=r"Both vertices provided are not the same "
                             r"dimensionality \(min_vertex = 3, "
                             r"max_vertex = 4\)\."):
        AxisAlignedBoundingBox(minp, maxp)


def test_bbox_construction_maxp_not_greater():
    """
    Test the check that the max-coordinate must be >= min-coordinate.
    """
    minp = (10, 10)
    maxp = (11, 9)
    with pytest.raises(ValueError,
                       match=r"The maximum vertex was not strictly >= the "
                             r"minimum vertex\."):
        AxisAlignedBoundingBox(minp, maxp)


def test_bbox_str():
    """
    Test that __str__ returns without error.
    """
    assert str(AxisAlignedBoundingBox([0], [1.2])) == \
        "<AxisAlignedBoundingBox [[0], [1.2]]>"


def test_bbox_repr():
    """
    Test that __repr__ returns without error.
    """
    assert repr(AxisAlignedBoundingBox([0], [1.2])) == \
        "<smqtk.representation.bbox.AxisAlignedBoundingBox " \
        "min_vertex=[0] max_vertex=[1.2]>"


def test_bbox_hash():
    """
    Test expected hashing of bounding box.
    """
    p1 = (0, 1, 2)
    p2 = (1, 2, 3)
    expected_hash = hash((p1, p2))
    assert hash(AxisAlignedBoundingBox(p1, p2)) == expected_hash


def test_bbox_equality_with_self():
    """
    Test that a bounding box is equal to itself.
    """
    bb = AxisAlignedBoundingBox([0], [0])
    assert bb == bb  # lgtm[py/comparison-of-identical-expressions]


def test_bbox_equality_other_is_copy():
    """
    Test that a bounding box is equal to an equivalent other bounding box
    instance.
    """
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    bb2 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    assert bb1 == bb2


def test_bbox_equality_other_is_close():
    """
    Test that a bounding box is equal to an equivalent other bounding box
    instance.
    """
    e = 1e-8  # default absolute tolerance on ``numpy.allclose`` function.
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    bb2 = AxisAlignedBoundingBox([1 + e, 2 + e, 3 + e], [2 + e, 3 + e, 4 + e])

    # Basic array equality is exact, which should show that these are not
    # strictly equal.
    # noinspection PyUnresolvedReferences
    assert (bb1.min_vertex != bb2.min_vertex).all()
    # noinspection PyUnresolvedReferences
    assert (bb1.max_vertex != bb2.max_vertex).all()

    assert bb1 == bb2


@mock.patch.object(AxisAlignedBoundingBox, 'EQUALITY_ATOL',
                   new_callable=mock.PropertyMock)
@mock.patch.object(AxisAlignedBoundingBox, 'EQUALITY_RTOL',
                   new_callable=mock.PropertyMock)
def test_bbox_equality_other_not_close_enough(m_bbox_rtol, m_bbox_atol):
    """
    Test modifying tolerance values
    :return:
    """
    e = 1e-8  # default absolute tolerance on ``numpy.allclose`` function.
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    bb2 = AxisAlignedBoundingBox([1 + e, 2 + e, 3 + e], [2 + e, 3 + e, 4 + e])

    # Basic array equality is exact, which should show that these are not
    # strictly equal.
    # noinspection PyUnresolvedReferences
    assert (bb1.min_vertex != bb2.min_vertex).all()
    # noinspection PyUnresolvedReferences
    assert (bb1.max_vertex != bb2.max_vertex).all()

    # If we reduce the tolerances, the 1e-8 difference will become intolerable.
    m_bbox_rtol.return_value = 1.e-10
    m_bbox_atol.return_value = 1.e-16
    assert not (bb1 == bb2)


def test_bbox_equality_other_not_close():
    """
    Test that other bbox is not equal when bbox is sufficiently different.
    """
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    # Differs in max z bounds.
    bb2 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 5])
    assert not (bb1 == bb2)


def test_bbox_equality_other_not_bbox():
    """
    Test that equality fails when the RHS is not a bounding box instance.
    """
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    assert not (bb1 == 'not a bbox')


@mock.patch('smqtk.representation.bbox.AxisAlignedBoundingBox.__eq__')
def test_bbox_not_equal(m_bbox_eq):
    """
    Test that non-equality is just calling the __eq__ in
    AxisAlignedBoundingBox.

    :param mock.MagicMock m_bbox_eq:
    """
    bb1 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])
    bb2 = AxisAlignedBoundingBox([1, 2, 3], [2, 3, 4])

    m_bbox_eq.assert_not_called()

    # noinspection PyStatementEffect
    bb1 != bb2

    m_bbox_eq.assert_called_once_with(bb2)


def test_getstate_format():
    """
    Test expected __getstate__ format.
    """
    min_v = (4.2, 8.9, 1)
    max_v = (9.2, 9.0, 48)
    expected_state = (
        [4.2, 8.9, 1],
        [9.2, 9.0, 48]
    )

    bb1 = AxisAlignedBoundingBox(min_v, max_v)
    assert bb1.__getstate__() == expected_state


def test_setstate_format():
    """
    Test expected state format compatible with setstate
    """
    state = (
        [4.2, 8.9, 1],
        [9.2, 9.0, 48]
    )
    expected_min_v = (4.2, 8.9, 1)
    expected_max_v = (9.2, 9.0, 48)

    bb = AxisAlignedBoundingBox([0], [1])
    bb.__setstate__(state)
    numpy.testing.assert_allclose(bb.min_vertex, expected_min_v)
    numpy.testing.assert_allclose(bb.max_vertex, expected_max_v)


def test_serialize_deserialize_pickle():
    """
    Test expected state representation.
    """
    min_v = (4.2, 8.9, 1)
    max_v = (9.2, 9.0, 48)

    bb1 = AxisAlignedBoundingBox(min_v, max_v)
    #: :type: AxisAlignedBoundingBox
    bb2 = pickle.loads(pickle.dumps(bb1))

    numpy.testing.assert_allclose(bb2.min_vertex, min_v)
    numpy.testing.assert_allclose(bb2.max_vertex, max_v)


def test_bbox_get_config_from_config():
    """
    Test that the expected configuration form is returned and can be turned
    around to create an equivalent AxisAlignedBoundingBox instance.
    """
    minp = [0, 0]
    maxp = [1, 1]
    expected_config = {
        'min_vertex': [0.0, 0.0],
        'max_vertex': [1.0, 1.0],
    }

    bb1 = AxisAlignedBoundingBox(minp, maxp)
    bb1_config = bb1.get_config()
    assert bb1_config == expected_config

    # Try to create a second bbox from the config of the first, asserting that
    # internal values are the same, and yet-again generated config is the same.
    bb2 = AxisAlignedBoundingBox.from_config(bb1_config)
    numpy.testing.assert_allclose(bb2.min_vertex, minp)
    numpy.testing.assert_allclose(bb2.max_vertex, maxp)
    bb2_config = bb2.get_config()
    assert bb2_config == expected_config
    assert bb2_config == bb1_config


def test_bbox_ndim(ndim):
    """
    Test that the ``ndim`` property correctly reflects the dimensionality of
    the coordinates stored.

    :param ndim: Dimension integer fixture result.

    """
    bb = AxisAlignedBoundingBox([1] * ndim, [2] * ndim)
    assert bb.ndim == ndim


def test_bbox_deltas_1d():
    """
    Test that `deltas` property returns the correct value for an example 1D
    region.
    """
    minp = [5]
    maxp = [7]
    expected = [2]
    numpy.testing.assert_allclose(AxisAlignedBoundingBox(minp, maxp).deltas,
                                  expected)


def test_bbox_deltas_2d():
    """
    Test that `deltas` property returns the correct value for an example 1D
    region.
    """
    minp = [5, 2]
    maxp = [7, 83]
    expected = [2, 81]
    numpy.testing.assert_allclose(AxisAlignedBoundingBox(minp, maxp).deltas,
                                  expected)


def test_bbox_deltas_3d():
    """
    Test that `deltas` property returns the correct value for an example 1D
    region.
    """
    minp = [29, 38, 45]
    maxp = [792, 83, 45]
    expected = [763, 45, 0]
    numpy.testing.assert_allclose(AxisAlignedBoundingBox(minp, maxp).deltas,
                                  expected)


def test_bbox_deltas_4d():
    """
    Test that `deltas` property returns the correct value for an example 1D
    region.
    """
    minp = [3] * 4
    maxp = [9] * 4
    expected = [6] * 4
    numpy.testing.assert_allclose(AxisAlignedBoundingBox(minp, maxp).deltas,
                                  expected)


def test_bbox_dtype():
    """
    Test getting the representative dtype of the bounding box, including mix
    vertex array types
    """
    # int
    bb = AxisAlignedBoundingBox([0], [1])
    assert issubclass(bb.dtype.type, numpy.signedinteger)

    bb = AxisAlignedBoundingBox(numpy.array([0], dtype=numpy.uint8),
                                numpy.array([1], dtype=numpy.uint8))
    assert issubclass(bb.dtype.type, numpy.uint8)

    bb = AxisAlignedBoundingBox(numpy.array([0], dtype=numpy.uint8),
                                numpy.array([1], dtype=numpy.uint32))
    assert issubclass(bb.dtype.type, numpy.uint32)

    # float
    bb = AxisAlignedBoundingBox([0.], [1.])
    assert issubclass(bb.dtype.type, numpy.float)

    bb = AxisAlignedBoundingBox(numpy.array([0], dtype=numpy.float32),
                                numpy.array([1], dtype=numpy.float16))
    assert issubclass(bb.dtype.type, numpy.float32)

    # mixed
    bb = AxisAlignedBoundingBox([0], [1.0])
    assert issubclass(bb.dtype.type, numpy.float)
    bb = AxisAlignedBoundingBox([0.0], [1])
    assert issubclass(bb.dtype.type, numpy.float)


def test_bbox_hypervolume_1(ndim):
    """
    Test that we get the expected 1-area from various 1-area hyper-cubes.
    """
    minp = [0] * ndim
    maxp = [1] * ndim
    expected_area = 1
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area


def test_bbox_hypervolume_other():
    """
    Test that we get the expected non-trivial area for bboxes of various
    dimensions.
    """
    # 1D
    minp = [4]
    maxp = [8]
    expected_area = 4
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area

    # 2D
    minp = [0, 4]
    maxp = [2, 10]
    expected_area = 12  # 2 * 6
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area

    # 3D
    minp = [0, 4, 3]
    maxp = [1, 10, 8]
    expected_area = 30  # 1 * 6 * 5
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area

    # 4D
    minp = [0, 4, 3, 5]
    maxp = [1, 10, 8, 9]
    expected_area = 120  # 1 * 6 * 5 * 4
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area


def test_bbox_intersection():
    """
    Test expected return when there is a valid intersection.
    """
    # With self
    bb = AxisAlignedBoundingBox([1, 1], [2, 2])
    assert bb.intersection(bb) == bb

    # Diagonal
    # +---+
    # | +---+
    # +-|-+ |
    #   +---+
    bb1 = AxisAlignedBoundingBox([0, 0], [2, 2])
    bb2 = AxisAlignedBoundingBox([1, 1], [3, 3])
    expected = AxisAlignedBoundingBox([1, 1], [2, 2])
    assert bb1.intersection(bb2) == expected
    assert bb2.intersection(bb1) == expected

    # ``other`` fully contained with, and ``other`` fully enclosing
    # +-----+
    # | +-+ |
    # | +-+ |
    # +-----+
    bb1 = AxisAlignedBoundingBox([3, 2, 8], [6, 5, 11])
    bb2 = AxisAlignedBoundingBox([4, 3, 9], [5, 4, 10])
    expected = bb2
    assert bb1.intersection(bb2) == expected
    assert bb2.intersection(bb1) == expected


def test_bbox_no_intersection():
    """
    Test that the expected conditions result in no intersection.
    """
    # +-+
    # +-+
    #    +-+
    #    +-+
    bb1 = AxisAlignedBoundingBox([0, 0], [1, 1])
    bb2 = AxisAlignedBoundingBox([3, 3], [4, 4])
    assert bb1.intersection(bb2) is None
    assert bb2.intersection(bb1) is None

    # Edge adjacency is not intersection
    #  +-+-+
    #  +-+-+
    bb1 = AxisAlignedBoundingBox([0, 0], [1, 1])
    bb2 = AxisAlignedBoundingBox([1, 0], [2, 1])
    assert bb1.intersection(bb2) is None
    assert bb2.intersection(bb1) is None
    # +-+
    # +-+
    # +-+
    bb1 = AxisAlignedBoundingBox([0, 0], [1, 1])
    bb2 = AxisAlignedBoundingBox([0, 1], [1, 2])
    assert bb1.intersection(bb2) is None
    assert bb2.intersection(bb1) is None
