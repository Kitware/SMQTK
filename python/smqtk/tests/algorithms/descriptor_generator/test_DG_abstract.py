from __future__ import division, print_function
import unittest

import mock
import nose.tools as ntools
import numpy

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
import smqtk.representation


class TestGetDescriptorGeneratorImpls (unittest.TestCase):

    def test_get_descriptors(self):
        m = get_descriptor_generator_impls()
        # Currently no types that are guaranteed available
        ntools.assert_is_instance(m, dict, "Should return a dictionary of "
                                           "class label-to-types")


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

    def _compute_descriptor(self, data):
        return


class TestDescriptorGeneratorAbstract (unittest.TestCase):
    """
    Create mock object (look up mock module?) to test abstract super-class
    functionality in isolation.

    Test abstract super-class functionality where there is any
    """

    def test_compute_descriptor_invlaid_type(self):
        cd = DummyDescriptorGenerator()
        cd.valid_content_types = mock.Mock(return_value={'image/png'})

        mDataElement = mock.Mock(spec=smqtk.representation.DataElement)
        m_d = mDataElement()
        m_d.content_type.return_value = 'image/jpeg'

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        m_factory = mDescriptorFactory()
        m_factory.new_descriptor.return_value = mDescrElement()

        ntools.assert_raises(ValueError, cd.compute_descriptor, m_d, m_factory)

    def test_computeDescriptor_validType_newVector(self):
        expected_image_type = 'image/png'
        expected_vector = numpy.random.randint(0, 100, 10)
        expected_uuid = "a unique ID"

        # Set up mock classes/responses
        cd = DummyDescriptorGenerator()
        cd.valid_content_types = mock.Mock(return_value={expected_image_type})
        cd._compute_descriptor = mock.Mock(return_value=expected_vector)

        mDataElement = mock.Mock(spec=smqtk.representation.DataElement)
        m_data = mDataElement()
        m_data.content_type.return_value = expected_image_type
        m_data.uuid.return_value = expected_uuid

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescrElement().has_vector.return_value = False

        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        m_factory = mDescriptorFactory()
        m_factory.new_descriptor.return_value = mDescrElement()

        # Call: matching content types, no existing descriptor for data
        d = cd.compute_descriptor(m_data, m_factory, overwrite=False)

        m_factory.new_descriptor.assert_called_once_with(cd.name, expected_uuid)
        ntools.assert_true(cd._compute_descriptor.called)
        mDescrElement().set_vector.assert_called_once_with(expected_vector)
        ntools.assert_is(d, mDescrElement())

    def test_computeDescriptor_validType_existingVector(self):
        expected_image_type = 'image/png'
        expected_existing_vector = numpy.random.randint(0, 100, 10)
        expected_new_vector = numpy.random.randint(0, 100, 10)
        expected_uuid = "a unique ID"

        # Set up mock classes/responses
        mDataElement = mock.Mock(spec=smqtk.representation.DataElement)
        m_data = mDataElement()
        m_data.content_type.return_value = expected_image_type
        m_data.uuid.return_value = expected_uuid

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescrElement().has_vector.return_value = True
        mDescrElement().vector.return_value = expected_existing_vector

        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        m_factory = mDescriptorFactory()
        m_factory.new_descriptor.return_value = mDescrElement()

        cd = DummyDescriptorGenerator()
        cd.valid_content_types = mock.Mock(return_value={expected_image_type})
        cd._compute_descriptor = mock.Mock(return_value=expected_new_vector)

        # Call: matching content types, existing descriptor for data
        d = cd.compute_descriptor(m_data, m_factory, overwrite=False)

        ntools.assert_true(mDescrElement().has_vector.called)
        ntools.assert_false(cd._compute_descriptor.called)
        ntools.assert_false(mDescrElement().set_vector.called)
        ntools.assert_is(d, mDescrElement())

    def test_computeDescriptor_validType_existingVector_overwrite(self):
        expected_image_type = 'image/png'
        expected_existing_vector = numpy.random.randint(0, 100, 10)
        expected_new_vector = numpy.random.randint(0, 100, 10)
        expected_uuid = "a unique ID"

        # Set up mock classes/responses
        mDataElement = mock.Mock(spec=smqtk.representation.DataElement)
        m_data = mDataElement()
        m_data.content_type.return_value = expected_image_type
        m_data.uuid.return_value = expected_uuid

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescrElement().has_vector.return_value = True
        mDescrElement().vector.return_value = expected_existing_vector

        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        m_factory = mDescriptorFactory()
        m_factory.new_descriptor.return_value = mDescrElement()

        cd = DummyDescriptorGenerator()
        cd.valid_content_types = mock.Mock(return_value={expected_image_type})
        cd._compute_descriptor = mock.Mock(return_value=expected_new_vector)

        # Call: matching content types, existing descriptor for data
        d = cd.compute_descriptor(m_data, m_factory, overwrite=True)

        ntools.assert_false(mDescrElement().has_vector.called)
        ntools.assert_true(cd._compute_descriptor.called)
        cd._compute_descriptor.assert_called_once_with(m_data)
        ntools.assert_true(mDescrElement().set_vector.called)
        mDescrElement().set_vector.assert_called_once_with(expected_new_vector)
        ntools.assert_is(d, mDescrElement())

    def test_computeDescriptorAsync(self):
        # Only using threading because mock.Mock can't be serialized (pickled)
        # for multiprocessing IPC.

        # Mock input data
        m_d0 = mock.Mock(name='data-0',
                         spec=smqtk.representation.DataElement)()
        m_d1 = mock.Mock(name='data-1',
                         spec=smqtk.representation.DataElement)()

        m_d0.uuid.return_value = 'uuid-0'
        m_d1.uuid.return_value = 'uuid-1'

        m_factory = \
            mock.Mock(spec=smqtk.representation.DescriptorElementFactory)()

        def mock_compute(d, *_):
            if d is m_d0:
                return 1
            elif d is m_d1:
                return 2
            else:
                return None

        generator = DummyDescriptorGenerator()
        generator.compute_descriptor = mock.Mock(
            side_effect=mock_compute
        )

        m = generator.compute_descriptor_async([m_d0, m_d1], m_factory,
                                               overwrite=False, use_mp=False)

        ntools.assert_equal(len(m), 2)
        ntools.assert_in(m_d0.uuid(), m)
        ntools.assert_in(m_d1.uuid(), m)
        ntools.assert_equal(m[m_d0.uuid()], 1)
        ntools.assert_equal(m[m_d1.uuid()], 2)

        ntools.assert_true(generator.compute_descriptor.called)
        ntools.assert_equal(generator.compute_descriptor.call_count, 2)
        generator.compute_descriptor.assert_any_call(m_d0, m_factory, False)
        generator.compute_descriptor.assert_any_call(m_d1, m_factory, False)

    def test_computeDescriptorAsync_failure(self):
        # Only using threading because mock.Mock can't be serialized (pickled)
        # for multiprocessing IPC.

        m_d0 = mock.Mock(spec=smqtk.representation.DataElement)()
        m_d1 = mock.Mock(spec=smqtk.representation.DataElement)()

        m_factory = \
            mock.Mock(spec=smqtk.representation.DescriptorElementFactory)()

        generator = DummyDescriptorGenerator()
        generator.compute_descriptor = mock.Mock(
            side_effect=RuntimeError("Intended exception")
        )

        ntools.assert_raises(
            RuntimeError,
            generator.compute_descriptor_async,
            [m_d0, m_d1], m_factory,
            procs=2,
            overwrite=False, use_mp=False
        )
