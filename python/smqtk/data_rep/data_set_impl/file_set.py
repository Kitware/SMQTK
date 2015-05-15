__author__ = 'purg'

from smqtk.data_rep import DataSet


class FileSet (DataSet):
    """
    File-based data set. Data elements will all be file-based (DataFile type,
    see ``../data_element_impl/file_element.py``).
    """

    # def __init__(self, content_directoriy):
