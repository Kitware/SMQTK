import os
import pickle

from six.moves import mock

from smqtk.representation.classification_element.file \
    import FileClassificationElement


def test_is_usable():
    """
    Test that the file implementation is usable (it should always be)
    """
    assert FileClassificationElement.is_usable()


def test_init_abs_save_dir_no_split():
    """
    Test that using an absolute directory path results in an expected filepath
    result.
    """
    expected_save_dir = '/some/abs/path'
    expected_filename = "test.0.classification.pickle"
    expected_filepath = os.path.join(expected_save_dir, expected_filename)

    e = FileClassificationElement('test', 0, expected_save_dir,
                                  subdir_split=None)
    assert e.save_dir == expected_save_dir
    assert e.filepath == expected_filepath


def test_init_rel_save_dir_no_split():
    """
    Test that using a relative directory path results in an expected
    filepath result.
    """
    cwd = os.getcwd()

    given_typename = 'test'
    given_uuid = '6172bar'
    given_save_dir = 'foo/bar'

    expected_save_dir = os.path.join(cwd, given_save_dir)
    expected_filename = "test.6172bar.classification.pickle"
    expected_filepath = os.path.join(expected_save_dir, expected_filename)

    e = FileClassificationElement(given_typename, given_uuid, given_save_dir,
                                  subdir_split=None)
    assert e.save_dir == expected_save_dir
    assert e.filepath == expected_filepath


def test_init_with_split_0_and_1():
    """
    Test that initializing with subdir_split of 0 or 1 results in an expected
    filepath result that is equivalent to not passing .
    """
    given_typename = 'test'
    given_uuid = '6172barH'
    given_save_dir = '/foo/bar'

    # We expect the last split to be dropped as per documentation.
    expected_filepath = os.path.join(given_save_dir,
                                     'test.6172barH.classification.pickle')

    e = FileClassificationElement(given_typename, given_uuid, given_save_dir,
                                  subdir_split=0)
    assert e.filepath == expected_filepath, "Failed for split = 0"

    e = FileClassificationElement(given_typename, given_uuid, given_save_dir,
                                  subdir_split=1)
    assert e.filepath == expected_filepath, "Failed for split = 1"


def test_init_with_split_4():
    """
    Test that initializing with subdir_split results in an expected filepath
    result.
    """
    given_typename = 'test'
    given_uuid = '6172barH'
    given_save_dir = '/foo/bar'
    given_split = 4

    # We expect the last split to be dropped as per documentation.
    expected_filepath = os.path.join(given_save_dir, '61', '72', 'ba',
                                     'test.6172barH.classification.pickle')

    e = FileClassificationElement(given_typename, given_uuid, given_save_dir,
                                  subdir_split=given_split)
    assert e.filepath == expected_filepath


def test_serialize_deserialize_pickle():
    """
    Test that we can serialize and deserialize element and maintain equal
    attributes.
    """
    expected_typename = 'test'
    expected_uuid = 235246
    expected_save_dir = '/foo/bar/thing'
    expected_pp = 2
    expected_subdir_split = 2
    expected_filepath = '/foo/bar/thing/235/test.235246.classification.pickle'

    e1 = FileClassificationElement(expected_typename, expected_uuid,
                                   expected_save_dir, expected_subdir_split,
                                   expected_pp)

    buff = pickle.dumps(e1)
    #: :type: FileClassificationElement
    e2 = pickle.loads(buff)

    assert e2.type_name == expected_typename
    assert e2.uuid == expected_uuid
    assert e2.save_dir == expected_save_dir
    assert e2.pickle_protocol == expected_pp
    assert e2.subdir_split == expected_subdir_split
    assert e2.filepath == expected_filepath


def test_get_config():
    """
    Test that get_config returns an expected configuration.
    """
    expected_typename = 'test'
    expected_uuid = 235246
    expected_save_dir = '/foo/bar/thing'
    expected_pp = 1
    expected_subdir_split = 2

    e = FileClassificationElement(expected_typename, expected_uuid,
                                  expected_save_dir, expected_subdir_split,
                                  expected_pp)

    expected_conf = {
        'save_dir': '/foo/bar/thing',
        'subdir_split': 2,
        'pickle_protocol': 1,
    }
    assert e.get_config() == expected_conf


@mock.patch('os.path.isfile')
def test_has_classification(m_os_isfile):
    """
    Test that method returns true when the file exists on disk.
    """
    given_tn = 'test'
    given_uuid = 0
    given_subdir = '/some/test'
    expected_fp = '/some/test/test.0.classification.pickle'

    # Simulate the expected file existing and nothing else.
    m_os_isfile.side_effect = lambda p: p == expected_fp

    e = FileClassificationElement(given_tn, given_uuid, given_subdir)

    assert e.has_classifications() is True
    m_os_isfile.assert_called_once_with(expected_fp)


@mock.patch('os.path.isfile')
def test_has_classification_no_file(m_os_isfile):
    """
    Test that has_classification returns False when the target file being
    checked does not exist on the filesystem.
    """
    given_tn = 'test'
    given_uuid = 0
    given_subdir = '/some/test'
    expected_fp = '/some/test/test.0.classification.pickle'

    # Simulate the expected file existing and nothing else.
    m_os_isfile.return_value = False

    e = FileClassificationElement(given_tn, given_uuid, given_subdir)

    assert e.has_classifications() is False
    m_os_isfile.assert_called_once_with(expected_fp)
