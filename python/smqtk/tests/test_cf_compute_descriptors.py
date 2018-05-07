from __future__ import print_function
import collections

import numpy
import pytest

import smqtk
from smqtk.compute_functions import (compute_many_descriptors,
                                     compute_transformed_descriptors,
                                     _CountedGenerator)

from six.moves import range
from six import add_move, MovedModule
add_move(MovedModule('mock', 'mock', 'unittest.mock'))
from six.moves import mock  # NOQA E402


NUM_BASE_ELEMENTS = 3

mock_DescriptorGenerator = mock.Mock(
    spec=smqtk.algorithms.descriptor_generator.DescriptorGenerator)
mock_DescriptorFactory = mock.Mock(
    spec=smqtk.representation.DescriptorElementFactory)
mock_DescriptorIndex = mock.Mock(spec=smqtk.representation.descriptor_index)


def dummy_transform(data_element):
    """Transform function for testing augmented descriptor computation"""
    new_elements = [mock.Mock(spec=smqtk.representation.DataElement)
                    for i in range(data_element.uuid() + 1)]
    for index, de in enumerate(new_elements):
        de.uuid.return_value = data_element.uuid()**2 + index
    return new_elements


@pytest.fixture
def data_elements():
    """Mock data elements"""
    data_elements = [mock.Mock(spec=smqtk.representation.DataElement)
                     for i in range(NUM_BASE_ELEMENTS)]
    for index, de in enumerate(data_elements):
        de.uuid.return_value = index
    return data_elements


@pytest.fixture
def descr_generator():
    """Mock descriptor generator"""
    mock_vector = numpy.random.randint(0, 100, 10)

    def dummy_cd_async(elems, *args, **kwargs):
        # Note: Cannot simply mock this because we must run through iterator
        collections.deque(elems, maxlen=0)
        return {i: mock_vector for i in range(NUM_BASE_ELEMENTS)}

    descr_generator = mock_DescriptorGenerator()
    descr_generator.is_usable = mock.Mock(return_value=True)
    descr_generator.compute_descriptor_async = mock.Mock(
        side_effect=dummy_cd_async)

    return descr_generator


@pytest.fixture
def mock_de():
    """Mock descriptor element"""
    mock_DescriptorElement = mock.Mock(
        spec=smqtk.representation.DescriptorElement)
    mock_de = mock_DescriptorElement()
    mock_de.has_vector.return_value = False
    return mock_de


@pytest.fixture
def descr_factory():
    descr_factory = mock_DescriptorFactory()
    descr_factory.new_descriptor.return_value = mock_de
    return descr_factory


@pytest.fixture
def descr_index():
    """Mock descriptor index"""
    descr_index = mock.Mock(smqtk.representation.DescriptorIndex)
    return descr_index


def test_compute_many_descriptors(data_elements, descr_generator, mock_de,
                                  descr_factory, descr_index):
    descriptors = compute_many_descriptors(data_elements,
                                           descr_generator,
                                           descr_factory, descr_index,
                                           batch_size=None)

    descriptors_count = 0
    for desc, uuid in zip(descriptors, range(NUM_BASE_ELEMENTS)):
        # Make sure order is preserved
        assert desc[0].uuid() == uuid
        descriptors_count += 1
    # Make sure correct number of elements returned
    assert descriptors_count == NUM_BASE_ELEMENTS

    # Since batch_size is None, these should only be called once
    assert descr_generator.compute_descriptor_async.call_count == 1
    assert descr_index.add_many_descriptors.call_count == 1


def test_compute_many_descriptors_batched(data_elements, descr_generator,
                                          mock_de, descr_factory, descr_index):
    batch_size = 2
    descriptors = compute_many_descriptors(data_elements, descr_generator,
                                           descr_factory, descr_index,
                                           batch_size=batch_size)

    descriptors_count = 0
    for desc, uuid in zip(descriptors, range(NUM_BASE_ELEMENTS)):
        # Make sure order is preserved
        assert desc[0].uuid() == uuid
        descriptors_count += 1
    # Make sure correct number of elements returned
    assert descriptors_count == NUM_BASE_ELEMENTS

    # Check number of calls
    num_calls = NUM_BASE_ELEMENTS / batch_size + [0, 1][
        bool(NUM_BASE_ELEMENTS % batch_size)]
    assert descr_generator.compute_descriptor_async.call_count == num_calls
    assert descr_index.add_many_descriptors.call_count == num_calls


def test_CountedGenerator():
    test_data = range(NUM_BASE_ELEMENTS)
    lengths = []
    test_counted_generator = _CountedGenerator(test_data, lengths)()
    for stored_element, value in zip(test_counted_generator, test_data):
        assert value == stored_element
    assert lengths == [NUM_BASE_ELEMENTS]


def test_compute_transformed_descriptors(data_elements, descr_generator,
                                         mock_de, descr_factory, descr_index):
    descriptors = compute_transformed_descriptors(data_elements,
                                                  descr_generator,
                                                  descr_factory, descr_index,
                                                  dummy_transform,
                                                  batch_size=None)

    # Make sure order is preserved
    for desc, uuids in zip(descriptors, [[0], [1, 2], [4, 5, 6]]):
        for i, uuid_ in enumerate(uuids):
            assert desc[0].uuid()**2 + i == uuid_
            collections.deque(desc[1], maxlen=0)  # Run through iterator

    # Since batch_size is None, these should only be called once
    assert descr_generator.compute_descriptor_async.call_count == 1
    assert descr_index.add_many_descriptors.call_count == 1
