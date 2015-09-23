import multiprocessing.pool
import unittest
import warnings

import mock
import nose.tools as ntools
import numpy

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.algorithms.descriptor_generator import _async_feature_generator_helper
import smqtk.representation

__author__ = 'purg'


def test_get_descriptors():
    m = get_descriptor_generator_impls()
    ntools.assert_is_instance(m, dict, "Should return a dictionary of class "
                                       "label-to-types")


class DummyDescriptorGenerator (DescriptorGenerator):
    """
    Shell implementation of abstract class in order to test abstract class
    functionality.
    """

    # No base implementation

    @classmethod
    def is_usable(cls):
        return

    def get_config(self):
        return {}

    def valid_content_types(self):
        return

    def _compute_descriptor(self, data):
        return

    # Have base implementations

    def generate_model(self, data_set, **kwargs):
        super(DummyDescriptorGenerator, self).generate_model(data_set, **kwargs)


class TestAsyncHelper (unittest.TestCase):

    mDataElement = mock.Mock(spec=smqtk.representation.DataElement)

    def test_valid_data(self):
        expected_vector = numpy.random.randint(0, 100, 10)

        cd = DummyDescriptorGenerator()
        cd._compute_descriptor = mock.Mock(return_value=expected_vector)

        v = _async_feature_generator_helper(cd, self.mDataElement())

        ntools.assert_true(cd._compute_descriptor.called)
        ntools.assert_equal(cd._compute_descriptor.call_count, 1)
        cd._compute_descriptor.assert_called_once_with(self.mDataElement())
        ntools.assert_true(numpy.array_equal(v, expected_vector))

    def test_nan_data(self):
        # Makes a vector of NaN values. A vector of not-zeros makes a vector of
        # inf values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected_vector = numpy.zeros(10) / 0

        cd = DummyDescriptorGenerator()
        cd._compute_descriptor = mock.Mock(return_value=expected_vector)

        v = _async_feature_generator_helper(cd, self.mDataElement())

        ntools.assert_is_none(v)

    def test_inf_data(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected_vector = numpy.arange(1, 10) / 0.

        cd = DummyDescriptorGenerator()
        cd._compute_descriptor = mock.Mock(return_value=expected_vector)

        v = _async_feature_generator_helper(cd, self.mDataElement())

        ntools.assert_is_none(v)

    @mock.patch('smqtk.algorithms.descriptor_generator.numpy')
    def test_exception(self, mNumpy):
        cd = DummyDescriptorGenerator()
        cd._compute_descriptor = mock.Mock(side_effect=Exception('Some error'))

        v = _async_feature_generator_helper(cd, self.mDataElement())

        ntools.assert_false(mNumpy.isnan.called)
        ntools.assert_is_none(v)


class TestContentDescriptorAbstract (unittest.TestCase):
    """
    Create mock object (look up mock module?) to test abstract super-class
    functionality in isolation.

    Test abstract super-class functionality where there is any
    """

    def test_generate_model(self):
        cd = DummyDescriptorGenerator()
        mDataElement = mock.Mock(spec=smqtk.representation.DataElement)
        mDataElement().content_type.return_value = 'image/png'
        m_dataset = [mDataElement(), mDataElement()]

        # When descriptor takes PNGs
        cd.valid_content_types = mock.Mock(return_value={'image/png'})
        cd.generate_model(m_dataset)
        ntools.assert_equal(cd.valid_content_types.call_count, 1)

        # When descriptor doesn't take PNGs
        cd.valid_content_types = mock.Mock(return_value={'image/jpeg'})
        ntools.assert_raises(ValueError, cd.generate_model, m_dataset)
        ntools.assert_equal(cd.valid_content_types.call_count, 1)

        # Mixed acceptance
        d1 = mDataElement()
        d2 = mDataElement()
        d1.content_type.return_value = "image/jpeg"
        d2.content_type.return_value = 'image/png'

        cd.valid_content_types = mock.Mock(return_value={'image/png',
                                                         'image/jpeg'})
        cd.generate_model(m_dataset)
        ntools.assert_equal(cd.valid_content_types.call_count, 1)

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

    @mock.patch('smqtk.algorithms.descriptor_generator.multiprocessing.Pool')
    def test_computeDescriptorAsync(self, mPool):
        expected_new_descriptors = [numpy.random.randint(0, 100, 10),
                                    numpy.random.randint(0, 100, 10)]
        expected_uuids = [1, 2]

        # Set up mocks
        mAsyncResult = mock.Mock(spec=multiprocessing.pool.ApplyResult)
        mAsyncResult().get.side_effect = expected_new_descriptors

        mPool().apply_async.return_value = mAsyncResult()

        mDataElement0 = mock.Mock(spec=smqtk.representation.DataElement)
        m_d0 = mDataElement0()
        m_d0.uuid.return_value = expected_uuids[0]

        mDataElement1 = mock.Mock(spec=smqtk.representation.DataElement)
        m_d1 = mDataElement1()
        m_d1.uuid.return_value = expected_uuids[1]

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescrElement().has_vector.return_value = False

        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        mDescriptorFactory().new_descriptor.return_value = mDescrElement()

        # The call
        cd = DummyDescriptorGenerator()
        de_map = cd.compute_descriptor_async([m_d0, m_d1],
                                             mDescriptorFactory())

        # Check mocks
        ntools.assert_equals(mPool().apply_async.call_count, 2)
        mPool().apply_async.assert_has_calls([
            mock.call(_async_feature_generator_helper, args=(cd, m_d0)),
            mock.call(_async_feature_generator_helper, args=(cd, m_d1)),
        ])
        ntools.assert_equals(mAsyncResult().get.call_count, 2)
        ntools.assert_equals(mDescrElement().set_vector.call_count, 2)
        mDescrElement().set_vector.assert_has_calls([
            mock.call(expected_new_descriptors[0]),
            mock.call(expected_new_descriptors[1]),
        ], any_order=True)

        ntools.assert_in(expected_uuids[0], de_map)
        ntools.assert_in(expected_uuids[1], de_map)

    @mock.patch('smqtk.algorithms.descriptor_generator.multiprocessing.Pool')
    def test_computeDescriptorAsync_failure(self, mPool):
        expected_uuids = [1, 2]

        # Set up mocks
        mAsyncResult = mock.Mock(spec=multiprocessing.pool.ApplyResult)
        mAsyncResult().get.return_value = None

        mPool().apply_async.return_value = mAsyncResult()

        mDataElement0 = mock.Mock(spec=smqtk.representation.DataElement)
        m_d0 = mDataElement0()
        m_d0.uuid.return_value = expected_uuids[0]

        mDataElement1 = mock.Mock(spec=smqtk.representation.DataElement)
        m_d1 = mDataElement1()
        m_d1.uuid.return_value = expected_uuids[1]

        mDescrElement = mock.Mock(spec=smqtk.representation.DescriptorElement)
        mDescrElement().has_vector.return_value = False

        mDescriptorFactory = mock.Mock(spec=smqtk.representation.DescriptorElementFactory)
        mDescriptorFactory().new_descriptor.return_value = mDescrElement()

        # The call
        cd = DummyDescriptorGenerator()
        ntools.assert_raises(RuntimeError, cd.compute_descriptor_async,
                             [m_d0, m_d1], mDescriptorFactory())

        # Check mocks
        ntools.assert_equals(mPool().apply_async.call_count, 2)
        mPool().apply_async.assert_has_calls([
            mock.call(_async_feature_generator_helper, args=(cd, m_d0)),
            mock.call(_async_feature_generator_helper, args=(cd, m_d1)),
        ])
        ntools.assert_equals(mAsyncResult().get.call_count, 2)
        ntools.assert_false(mDescrElement().set_vector.called)
