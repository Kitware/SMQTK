import numpy
import pytest

# noinspection PyUnresolvedReferences
from six.moves import mock  # move defined in ``smqtk.tests``

from smqtk.representation.bbox import AxisAlignedBoundingBox


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

def test_bbox_construction_2d(seq_type):
    """ Test construction a AxisAlignedBoundingBox with 2D coordinates. """
    minp = (1, 1)
    maxp = (6, 7)

    minp_s = seq_type(minp)
    maxp_s = seq_type(maxp)
    bb = AxisAlignedBoundingBox(minp_s, maxp_s)

    assert isinstance(bb.min_vertex, numpy.ndarray)
    assert isinstance(bb.max_vertex, numpy.ndarray)
    numpy.testing.assert_allclose(bb.min_vertex, minp)
    numpy.testing.assert_allclose(bb.max_vertex, maxp)


def test_bbox_construction_3d(seq_type):
    """ Test construction a AxisAlignedBoundingBox with 2D coordinates. """
    minp = (1, 1, 0)
    maxp = (6, 7, 10)

    minp_s = seq_type(minp)
    maxp_s = seq_type(maxp)
    bb = AxisAlignedBoundingBox(minp_s, maxp_s)

    assert isinstance(bb.min_vertex, numpy.ndarray)
    assert isinstance(bb.max_vertex, numpy.ndarray)
    numpy.testing.assert_allclose(bb.min_vertex, minp)
    numpy.testing.assert_allclose(bb.max_vertex, maxp)


def test_bbox_construction_32d(seq_type):
    """ Test construction a AxisAlignedBoundingBox with 2D coordinates. """
    minp = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    maxp = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

    minp_s = seq_type(minp)
    maxp_s = seq_type(maxp)
    bb = AxisAlignedBoundingBox(minp_s, maxp_s)

    assert isinstance(bb.min_vertex, numpy.ndarray)
    assert isinstance(bb.max_vertex, numpy.ndarray)
    numpy.testing.assert_allclose(bb.min_vertex, minp)
    numpy.testing.assert_allclose(bb.max_vertex, maxp)


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

    with pytest.raises(ValueError, match="One or both vertices provided had "
                                         "more than one array dimension "
                                         "\(min_vertex\.ndim == 2, "
                                         "max_vertex\.ndim == 1\)\."):
        # noinspection PyTypeChecker
        AxisAlignedBoundingBox(minp_2dim, maxp_1dim)
    with pytest.raises(ValueError, match="One or both vertices provided had "
                                         "more than one array dimension "
                                         "\(min_vertex\.ndim == 1, "
                                         "max_vertex\.ndim == 2\)\."):
        # noinspection PyTypeChecker
        AxisAlignedBoundingBox(minp_1dim, maxp_2dim)
    with pytest.raises(ValueError, match="One or both vertices provided had "
                                         "more than one array dimension "
                                         "\(min_vertex\.ndim == 2, "
                                         "max_vertex\.ndim == 2\)\."):
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
                       match="Both vertices provided are not the same "
                             "dimensionality \(min_vertex = 3, "
                             "max_vertex = 4\)\."):
        AxisAlignedBoundingBox(minp, maxp)


def test_bbox_construction_maxp_not_greater():
    """
    Test the check that the max-coordinate must be >= min-coordinate.
    """
    minp = (10, 10)
    maxp = (11, 9)
    with pytest.raises(ValueError,
                       match="The maximum vertex was not strictly >= the "
                             "minimum vertex\."):
        AxisAlignedBoundingBox(minp, maxp)


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
    assert bb == bb


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


def test_bbox_area_1(ndim):
    """
    Test that we get the expected 1-area from various 1-area hyper-cubes.
    """
    minp = [0] * ndim
    maxp = [1] * ndim
    expected_area = 1
    assert AxisAlignedBoundingBox(minp, maxp).hypervolume == expected_area


def test_bbox_area_other():
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
