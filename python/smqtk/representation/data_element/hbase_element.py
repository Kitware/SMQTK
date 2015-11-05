from smqtk.representation import DataElement


__author__ = 'paul.tunison@kitware.com'


# attempt to import required modules
try:
    import happybase
    import tika
    from tika import detector as tika_detector
except ImportError:
    happybase = None
    tika = None
    tika_detector = None


class HBaseDataElement (DataElement):
    """
    Wrapper for binary data contained on an HBase server somewhere. Uses Tika
    content type detection to determine content type of served data.
    """

    @classmethod
    def is_usable(cls):
        return None not in {happybase, tika_detector}

    def __init__(self, element_key, binary_column, hbase_address,
                 hbase_table, timeout=10000):
        """
        Create a new HBase data element wrapper/reference.

        :param element_key: Key to the table row containing the binary data
        :type element_key: str

        :param binary_column: Name of the column that contains the binary
            data.
        :type binary_column: str

        :param hbase_address: Address of the HBase server. This should at least
            be the hostname of the server. This might also take a ":port"
            suffix?
        :type hbase_address: str

        :param hbase_table: Name of the table the content is contained in.
        :param timeout:
        :return:
        """
        super(HBaseDataElement, self).__init__()
        self.element_key = element_key
        self.binary_column = binary_column
        self.hbase_address = hbase_address
        self.hbase_table = hbase_table
        self.timeout = int(timeout)

        self._binary_ct_cache = None

    def get_config(self):
        return {
            "element_key": self.element_key,
            "binary_column": self.binary_column,
            "hbase_address": self.hbase_address,
            "hbase_table": self.hbase_table,
            "timeout": self.timeout,
        }

    def content_type(self):
        if self._binary_ct_cache is None:
            self._binary_ct_cache = tika_detector.from_buffer(self.get_bytes())
        return self._binary_ct_cache

    def _new_hbase_table_connection(self):
        return happybase.Connection(self.hbase_address, timeout=self.timeout)\
            .table(self.hbase_table)

    def get_bytes(self):
        table = self._new_hbase_table_connection()
        r = table.row(self.element_key, columns=[self.binary_column])
        return r[self.binary_column]
