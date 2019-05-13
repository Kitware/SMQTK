from six.moves import mock

from smqtk.algorithms import ImageReader, ImageMatrixObjectDetector
from smqtk.representation import DataElement


@mock.patch('smqtk.algorithms.object_detection._interface'
            '.make_default_config')
def test_get_default_config(m_mdc):
    """
    Test configuration default generation
    """
    m_mdc_return_value = "make default expected return"
    m_mdc.return_value = m_mdc_return_value

    imod_dflt_config = ImageMatrixObjectDetector.get_default_config()
    m_mdc.assert_called_once()
    assert imod_dflt_config == {
        'image_reader': m_mdc_return_value,
    }


@mock.patch('smqtk.algorithms.object_detection._interface'
            '.from_config_dict')
@mock.patch('smqtk.algorithms.object_detection._interface'
            '.to_config_dict')
def test_config_cycle(m_tcd, m_fcd):
    """
    Test that get_config/from_config cycle results in instance with same
    appropriate attribute reflection.
    """
    class MockIMOD (ImageMatrixObjectDetector):

        @classmethod
        def is_usable(cls):
            return True

        def get_config(self):
            """ stub to be mocked """
            return super(MockIMOD, self).get_config()

        def _detect_objects_matrix(self, mat):
            """ stub to be mocked """
            raise NotImplementedError()

    t_imgreader_value = 'imma image reader'

    # noinspection PyTypeChecker
    inst = MockIMOD(t_imgreader_value)

    m_tcd_return_value = 'expected tcd return value'
    m_tcd.return_value = m_tcd_return_value

    m_fcd_return_value = 'expected fcd return value'
    m_fcd.return_value = m_fcd_return_value

    # Test running the cycle
    inst_config = inst.get_config()
    inst2 = MockIMOD.from_config(inst_config)
    inst2_config = inst2.get_config()

    assert m_tcd.call_count == 2
    m_tcd.assert_any_call(t_imgreader_value)
    m_fcd.assert_called_once_with(m_tcd_return_value,
                                  ImageReader.get_impls())
    m_tcd.assert_any_call(m_fcd_return_value)
    assert inst_config == inst2_config


def test_valid_content_types():
    """
    Test that valid content types are inherited from the ImageReader algo
    provided.
    """
    expected_content_types = {
        'text/plain',
        'image/png',
        'this is the test set'
    }

    m_image_reader = mock.MagicMock(spec=ImageReader)
    m_image_reader.valid_content_types.return_value = expected_content_types

    # Mock instance of abstract class
    m_inst = mock.MagicMock(spec=ImageMatrixObjectDetector)
    m_inst._image_reader = m_image_reader

    assert ImageMatrixObjectDetector.valid_content_types(m_inst) == \
        expected_content_types
    m_image_reader.valid_content_types.assert_called_once()


def test_is_valid_element():
    """
    Test that valid element determination is inherited from ImageReader algo
    provided.
    """
    expected_de = mock.MagicMock(spec=DataElement)
    expected_ive_return = 'test return value'

    m_image_reader = mock.MagicMock(spec=ImageReader)
    m_image_reader.is_valid_element.return_value = expected_ive_return

    # Mock instance of abstract class
    m_inst = mock.MagicMock(spec=ImageMatrixObjectDetector)
    m_inst._image_reader = m_image_reader

    assert ImageMatrixObjectDetector.is_valid_element(m_inst, expected_de) \
        == expected_ive_return
    m_image_reader.is_valid_element.assert_called_once_with(expected_de)


def test_detect_objects():
    """
    Test that ``_detect_objects`` wrapper acts as expected
    """
    expected_de = mock.MagicMock(spec=DataElement)
    expected_load_mat = "expected load return"

    m_image_reader = mock.MagicMock(spec=ImageReader)
    m_image_reader.load_as_matrix.return_value = expected_load_mat

    # Mock instance of abstract class
    m_inst = mock.MagicMock(spec=ImageMatrixObjectDetector)
    m_inst._image_reader = m_image_reader

    actual_dom_ret = \
        ImageMatrixObjectDetector._detect_objects(m_inst, expected_de)

    m_image_reader.load_as_matrix.assert_called_once_with(expected_de)
    m_inst._detect_objects_matrix.assert_called_once_with(expected_load_mat)
    assert actual_dom_ret == m_inst._detect_objects_matrix()
