from __future__ import division, print_function
import unittest

import numpy
import pytest
import unittest.mock as mock

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
import smqtk.representation
from smqtk.representation import DescriptorElement, DescriptorElementFactory


class DummyDescriptorGenerator (DescriptorGenerator):
    """
    Shell implementation of abstract class in order to test abstract class
    functionality.
    """

    @classmethod
    def is_usable(cls):
        return True

    def get_config(self):
        return {}

    def valid_content_types(self):
        return {}

    def _generate_arrays(self, data_iter):
        # Make sure we go through iter, yielding "arrays"
        for i, d in enumerate(data_iter):
            yield [i]
        self._post_iterator_check()

    def _post_iterator_check(self):
        """ Stub method for testing functionality is called post-final-yield.
        """

    def _generate_too_many_arrays(self, data_iter):
        """
        Swap-in generator to test error checking on over generation.
        """
        for i, d in enumerate(data_iter):
            yield [i]
        yield [-1]
        yield [-2]
        self._post_iterator_check()

    def _generate_too_few_arrays(self, data_iter):
        """
        Swap-in generator to test error checking on under generation.
        """
        # yield all but one data element.
        data_list = list(data_iter)
        for i in range(len(data_list) - 1):
            yield [i]
        self._post_iterator_check()


class TestDescriptorGeneratorAbstract (unittest.TestCase):
    """
    Create mock object (look up mock module?) to test abstract super-class
    functionality in isolation.

    Test abstract super-class functionality where there is any
    """

    def setUp(self):
        self.inst = DummyDescriptorGenerator()
        self.inst.valid_content_types = mock.Mock(return_value={'image/png'})
        self.inst._post_iterator_check = mock.Mock()

    def test_generate_arrays_invalid_type(self):
        """ Test that the raise-valid-element method catches an invalid input
        data type. """
        # Using dummy to pull in integrated mixin class functionality.
        inst = self.inst

        m_d = mock.Mock(spec=smqtk.representation.DataElement)
        m_d.content_type.return_value = 'image/jpeg'

        with pytest.raises(ValueError,
                           match="Data element does not match a content type "
                                 "reported as valid."):
            # list(inst.generate_arrays([m_d]))
            list(DescriptorGenerator.generate_arrays(inst, [m_d]))

        # No or incomplete iteration should have occurred, so post-yield
        # function should not be expected to have been called.
        inst._post_iterator_check.assert_not_called()

    def test_generate_arrays_valid_type(self):
        """ Test passing "valid" data elements causing dummy array returns."""
        # Using dummy to pull in integrated mixin class functionality.
        inst = self.inst

        data_list = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_list:
            # match "valid_content_types"
            d.content_type.return_value = 'image/png'

        expected_vectors = [[0], [1], [2]]
        actual_vectors = list(inst.generate_arrays(data_list))
        assert numpy.allclose(
            actual_vectors,
            expected_vectors
        )

        # Complete iteration should cause post-yield method to be called.
        inst._post_iterator_check.assert_called_once()

    def test_generate_arrays_empty_iter(self):
        """ Test that we correctly return an empty generator if an empty
        iterable is provided. """
        # Using dummy to pull in integrated mixin class functionality.
        inst = self.inst

        expected_vectors = []
        actual_vectors = list(inst.generate_arrays([]))
        assert actual_vectors == expected_vectors

        # Complete iteration should cause post-yield method to be called.
        inst._post_iterator_check.assert_called_once()

    def test_generate_elements_empty_iter(self):
        """ Test that we correctly return an empty generator if an empty
        iterable is provided. """
        expected_elems = []
        actual_elems = list(self.inst.generate_elements([]))
        assert actual_elems == expected_elems

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_elements_bad_content_type(self):
        """ Test that a ValueError occurs if one or more data elements passed
        are not considered to have valid content types. """
        d = mock.Mock(spec=smqtk.representation.DataElement)
        d.content_type.return_value = "image/jpeg"

        with pytest.raises(ValueError,
                           match="Data element does not match a content type "
                                 "reported as valid."):
            list(self.inst.generate_elements([d]))

        # Incomplete iteration should have occurred, so post-yield
        # function should not be expected to have been called.
        self.inst._post_iterator_check.assert_not_called()

    def test_generate_elements_impl_over_generate_elements(self):
        """
        Test that an error is thrown when an implementation that returns more
        vectors than data elements (IndexError).
        """
        # Mock data element input
        data_iter = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_iter:
            d.content_type.return_value = 'image/png'

        # Mock generated descriptor elements that *don't* have vectors
        m_descr_elem = mock.MagicMock(spec=DescriptorElement)
        m_descr_elem.has_vector.return_value = False

        # Mock factory to return some descriptor element mock
        m_fact = mock.MagicMock(spec=DescriptorElementFactory)
        m_fact.new_descriptor.return_value = m_descr_elem

        # Mock generator instance to return
        self.inst._generate_arrays = self.inst._generate_too_many_arrays

        # TODO: Check index error message when fail
        with pytest.raises(IndexError):
            list(self.inst.generate_elements(data_iter, descr_factory=m_fact,
                                             overwrite=False))

        # Incomplete iteration should have occurred, so post-yield
        # function should not be expected to have been called.
        self.inst._post_iterator_check.assert_not_called()

    def test_generate_elements_impl_under_generate_elements(self):
        """
        Test that an error is thrown when an implementation that returns less
        vectors than data elements.
        """
        # Mock data element input
        data_iter = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_iter:
            d.content_type.return_value = 'image/png'

        # Mock generated descriptor elements that *don't* have vectors
        m_descr_elem = mock.MagicMock(spec=DescriptorElement)
        m_descr_elem.has_vector.return_value = False
        m_descr_elem.uuid.return_value = "test_uuid"

        # Mock factory to return some descriptor element mock
        m_fact = mock.MagicMock(spec=DescriptorElementFactory)
        m_fact.new_descriptor.return_value = m_descr_elem

        # Mock generator instance to return
        self.inst._generate_arrays = self.inst._generate_too_few_arrays

        with pytest.raises(IndexError):
            list(self.inst.generate_elements(data_iter, descr_factory=m_fact,
                                             overwrite=False))

        # Under-yielding generator should have completed iteration, so the
        # post-yield method should have been called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_elements_non_preexisting(self):
        """ Test generating descriptor elements where none produced by the
        factory have existing vectors, i.e. all data elements are passed to
        underlying generation method. """
        # Mock data element input
        data_iter = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_iter:
            d.content_type.return_value = 'image/png'

        # Mock element type
        m_de_type = mock.MagicMock(name="DescrElemType")

        # Mock factory
        fact = smqtk.representation.DescriptorElementFactory(
            m_de_type, {}
        )

        # Mock element instance
        m_de_inst = m_de_type.from_config()  # from factory
        # !!! Mock that elements all have *no* vector set
        m_de_inst.has_vector.return_value = False

        # Default factory is the in-memory descriptor element.
        list(self.inst.generate_elements(data_iter, descr_factory=fact,
                                         overwrite=False))
        assert m_de_inst.has_vector.call_count == 3
        assert m_de_inst.set_vector.call_count == 3
        # We know the dummy vectors that should have been iterated out
        m_de_inst.set_vector.assert_any_call([0])
        m_de_inst.set_vector.assert_any_call([1])
        m_de_inst.set_vector.assert_any_call([2])

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_elements_all_preexisting(self):
        """ Test that no descriptors are computed if all generated descriptor
        elements report as having a vector and overwrite is False. """
        # Mock data element input
        data_iter = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_iter:
            d.content_type.return_value = 'image/png'

        # Mock element type
        m_de_type = mock.MagicMock(name="DescrElemType")

        # Mock factory
        fact = smqtk.representation.DescriptorElementFactory(
            m_de_type, {}
        )

        # Mock element instance
        m_de_inst = m_de_type.from_config()  # from factory
        # !!! Mock that elements all *have* a vector set
        m_de_inst.has_vector.return_value = True

        # Default factor is the in-memory descriptor element.
        list(self.inst.generate_elements(data_iter, descr_factory=fact,
                                         overwrite=False))
        assert m_de_inst.has_vector.call_count == 3
        assert m_de_inst.set_vector.call_count == 0

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_elements_all_preexisting_overwrite(self):
        """ Test that descriptors are computed even though the generated
        elements (mocked) report as having a vector.
        """
        # Mock data element input
        data_iter = [
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
            mock.Mock(spec=smqtk.representation.DataElement),
        ]
        for d in data_iter:
            d.content_type.return_value = 'image/png'

        # Mock element type
        m_de_type = mock.MagicMock(name="DescrElemType")

        # Mock factory
        fact = smqtk.representation.DescriptorElementFactory(
            m_de_type, {}
        )

        # Mock element instance
        m_de_inst = m_de_type.from_config()  # from factory
        # !!! Mock that elements all *have* a vector set
        m_de_inst.has_vector.return_value = True

        # Default factor is the in-memory descriptor element.
        list(self.inst.generate_elements(data_iter, descr_factory=fact,
                                         overwrite=True))
        # expect no has-vec checks because its after overwrite short-circuit.
        assert m_de_inst.has_vector.call_count == 0
        assert m_de_inst.set_vector.call_count == 3

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_elements_mixed_precomp(self):
        """ Test that a setup of factory-produced elements having and not
        having pre-existing vectors results in all returns. """
        # Mock data/descriptor element pairs, storing state for testing.
        # - content type matching above ``inst.valid_content_types()``
        # - marking indices 2, 3, 5 as NOT having prior vec stored
        data_iter = []
        m_descr_elems = []
        for i in range(8):
            data = mock.Mock(spec=smqtk.representation.DataElement)
            data.uuid.return_value = i
            data.content_type.return_value = 'image/png'
            data_iter.append(data)

            desc = mock.Mock(spec=smqtk.representation.DescriptorElement)
            if i in [2, 3, 5]:  # !!!
                desc.has_vector.return_value = False
            else:
                desc.has_vector.return_value = True
            m_descr_elems.append(desc)

        # Mock factory since we want to control has-vec return logic.
        def m_fact_newdesc(_, uuid):
            return dict(enumerate(m_descr_elems))[uuid]

        m_fact = \
            mock.MagicMock(spec=smqtk.representation.DescriptorElementFactory)
        m_fact.new_descriptor.side_effect = m_fact_newdesc

        actual_ret = list(
            self.inst.generate_elements(data_iter, descr_factory=m_fact,
                                        overwrite=False)
        )

        # Everything should have been checked for has-vec because overwrite was
        # not set.
        for e in m_descr_elems:
            e.has_vector.assert_called_once()

        # Check that the appropriate elements have been set vectors
        m_descr_elems[0].set_vector.assert_not_called()
        m_descr_elems[1].set_vector.assert_not_called()
        m_descr_elems[2].set_vector.assert_called_once_with([0])
        m_descr_elems[3].set_vector.assert_called_once_with([1])
        m_descr_elems[4].set_vector.assert_not_called()
        m_descr_elems[5].set_vector.assert_called_once_with([2])
        m_descr_elems[6].set_vector.assert_not_called()
        m_descr_elems[7].set_vector.assert_not_called()

        # Check that returned elements are the expected elements from
        assert actual_ret == m_descr_elems

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_one_array(self):
        """ Test that the one-array wrapper performs as expected.
        """
        m_d = mock.Mock(spec=smqtk.representation.DataElement)
        m_d.content_type.return_value = 'image/png'

        actual_v = self.inst.generate_one_array(m_d)
        expected_v = [0]
        numpy.testing.assert_allclose(actual_v, expected_v)

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()

    def test_generate_one_element(self):
        """ Test that the one-element wrapper performs as expected.
        Using default factory/overwrite params (memory, False).
        """
        m_d = mock.Mock(spec=smqtk.representation.DataElement)
        m_d.content_type.return_value = 'image/png'

        actual_e = self.inst.generate_one_element(m_d)
        expected_v = [0]
        numpy.testing.assert_allclose(
            actual_e.vector(), expected_v
        )

        # Complete iteration should cause post-yield method to be called.
        self.inst._post_iterator_check.assert_called_once()
