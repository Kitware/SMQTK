import abc
from collections import deque

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.descriptor_element.local_elements import \
    DescriptorMemoryElement
from smqtk.utils import ContentTypeValidator


DFLT_DESCRIPTOR_FACTORY = DescriptorElementFactory(DescriptorMemoryElement, {})


class DescriptorGenerator (SmqtkAlgorithm, ContentTypeValidator):
    """
    Base abstract Feature Descriptor interface.
    """

    @abc.abstractmethod
    def _generate_arrays(self, data_iter):
        """
        Inner template method that defines the generation of descriptor vectors
        for a given iterable of data elements.

        Pre-conditions:
          - Data elements input to this method have been validated to be of at
            least one of this class's reported ``valid_content_types``.

        :param collections.Iterable[smqtk.representation.DataElement] data_iter:
            Iterable of data element instances to be described.

        :raises RuntimeError: Descriptor extraction failure of some kind.

        :return: Iterable of numpy arrays in parallel association with the
            input data elements.
        :rtype: collections.Iterable[numpy.ndarray]
        """

    def generate_arrays(self, data_iter):
        """
        Generate descriptor vector elements for **all** input data elements.

        Descriptor arrays yielded out will be parallel in association with
        the data elements input.

        **Selective Iteration**
        For situations when it is desired to access specific generator returns,
        like when only one data element is provided in order to get a single
        array out, it is strongly recommended to expand the returned generator
        into a sequence type first. For example, expanding out the generator's
        returns into a list (``list(g.generate_arrays([e]))[0]``) is
        recommended over just getting the "next" element of the returned
        generator (``next(g.generate_arrays([e]))``).
        Expansion into a sequence allows the generator to fully execute, which
        includes any functionality after the final ``yield`` statement in any
        of the underlying iterators.

        :param collections.Iterable[smqtk.representation.DataElement] data_iter:
            Iterable of DataElement instances to be described.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor generator implementation.

        :return: Iterator of result numpy.ndarray instances.
        :rtype: collections.Iterator[numpy.ndarray]
        """
        # Intermediate iterator for testing that content types are valid
        # TODO: Use an order-preserving call to parallel_map()?
        validated_data_iter = (self.raise_valid_element(d) for d in data_iter)
        return self._generate_arrays(validated_data_iter)

    def generate_elements(self, data_iter,
                          descr_factory=DFLT_DESCRIPTOR_FACTORY,
                          overwrite=False):
        """
        Generate DescriptorElement instances for the input data elements,
        generating new descriptors for those elements that need them, or
        optionally all input data elements.

        Descriptor elements yielded out will be parallel in association with
        the data elements input. Descriptor element UUIDs are inherited from
        the data element it was generated from.

        **Selective Iteration**
        For situations when it is desired to access specific generator returns,
        like when only one data element is provided in order to get a single
        element out, it is strongly recommended to expand the returned
        generator into a sequence type first.
        For example, expanding out the generator's returns into a list
        (``list(g.generate_elements([e]))[0]``) is recommended over just
        getting the "next" element of the returned generator
        (``next(g.generate_elements([e]))``).
        Expansion into a sequence allows the generator to fully execute, which
        includes any functionality after the final ``yield`` statement in any
        of the underlying iterators that may perform required clean-up.

        **Non-redundant Processing**
        Certain descriptor element implementations, as dictated by the input
        factory, may be connected to persistent storage in the background.
        Because of this, some descriptor elements may already "have" a vector
        on construction.
        This method, by default, only computes new descriptor vectors for data
        elements whose associated descriptor element does not report as already
        containing a vector.
        If the ``overwrite`` flag is True then descriptors are computed for all
        input data elements and are set to their respective descriptor elements
        regardless of existing vector storage.

        :param collections.Iterable[smqtk.representation.DataElement] data_iter:
            Iterable of DataElement instances to be described.
        :param smqtk.representation.DescriptorElementFactory descr_factory:
            DescriptorElementFactory instance to drive the generation of
            element instances with some parametrization.
        :param bool overwrite:
            By default, if a factory-produced DescriptorElement reports as
            containing a vector, we do not compute a descriptor again for the
            associated data element. If this is ``True``, however, we will
            generate descriptors for all input data elements, overwriting the
            vectors previously stored in the factory-produces descriptor
            elements.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor generator implementation.
        :raises IndexError: Underlying vector-producing generator either under
            or over produced vectors.

        :return: Iterator of result DescriptorElement instances. UUIDs of
            generated DescriptorElement instances will reflect the UUID of the
            DataElement it was generated from.
        :rtype: collections.Iterator[smqtk.representation.DescriptorElement]
        """
        log_debug = self._log.debug

        # Parallel lists of (uuid, DescriptorElement, already-computed) triples
        #   for formulating the return yielding.
        # Using deques so we can efficiently popleft off of them in the below
        #   for-loop. This way we do not retain elements and booleans for
        #   things we have yielded that would otherwise build up if this method
        #   iterated for a long time.
        #: :type: deque[(smqtk.representation.DescriptorElement, bool)]
        elem_and_status_q = deque()

        # Flag for end of data iteration. When not None will be the index of
        # the last data/descriptor element to be yielded. This will NOT be the
        # number of elements to be yielded, that would be ``end_of_iter[0]+1``.
        # NOTE: When moving to python3+ only support, we can use the
        #       ``nonlocal end_of_iter`` statement within the
        #       ``iter_tocompute_data`` function instead of imitating a state
        #       object here.
        #: :type: list[int|None]
        end_of_iter = [None]

        # TODO: Make generator threadsafe?
        # See: https://anandology.com/blog/using-iterators-and-generators/
        def iter_tocompute_data():
            """ Yield data elements that need descriptor computation.

            Populate parallel lists as we traverse ``data_iter``, yielding data
            elements that do not have an associated descriptor element with a
            vector set. Alternatively, if ``overwrite`` is True, all data
            elements are yielded and descriptor elements are marked for vector
            setting.

            :returns: Generator over data elements that need descriptor
                generation.
            :rtype: typing.Generator[smqtk.representation.DataElement]
            """
            # Running var for the index of final data element in input
            # iterator. This will be -1 or the value of the final index in the
            # parallel lists.
            last_i = -1
            for d_i, data in enumerate(data_iter):
                data_uuid_ = data.uuid()
                descr_elem_ = \
                    descr_factory.new_descriptor(self.name, data_uuid_)
                already_computed = not overwrite and descr_elem_.has_vector()
                elem_and_status_q.append((descr_elem_, already_computed))
                if not already_computed:
                    # Descriptor should be computed for this element
                    log_debug("Yielding DataElement with UUID {} for "
                              "generation".format(data_uuid_))
                    yield data
                else:
                    log_debug("Descriptor already computed for UUID {}"
                              .format(data_uuid_))
                last_i = d_i

            end_of_iter[0] = last_i

        descr_vec_iter = self.generate_arrays(iter_tocompute_data())
        for v_i, v in enumerate(descr_vec_iter):
            # These pops would fail with an IndexError if there is nothing left
            #   left from parallel allocation within ``iter_tocompute_data``.
            # This usually means that the ``self.generate_arrays`` is
            #   generating more vectors than there are descr element slots to
            #   fill.
            v_descr_elem, v_already_computed = elem_and_status_q.popleft()

            # Forwarding the iterator of the ``descr_vec_iter`` generator will,
            # probably, forward the ``iter_tocompute_data`` iterator, thus
            # populating the ``elem_and_status_q`` to some degree. The current
            # ``v`` should be be used to populate the next DescriptorElement
            # with an associated "already_computed" flag of False.
            while v_already_computed:
                yield v_descr_elem
                # We clearly have a descriptor vector from the result of
                # computation so there should logically be some future element
                # in which to store this result.
                v_descr_elem, v_already_computed = elem_and_status_q.popleft()

            # Assign the current computed descriptor vector to the current
            # element that should be set to.
            log_debug("Setting computed vector {} to element UUID {}"
                      .format(v_i, v_descr_elem.uuid()))
            v_descr_elem.set_vector(v)
            yield v_descr_elem

        # At this point, the ``iter_tocompute_data()`` iterator should have
        #   completed due to the ``self.generate_arrays`` method iterating
        #   through it completely, assigning a value to ``end_of_iter[0]``.
        # This also indicates that nothing more should be being added to the
        #   deques.
        assert end_of_iter[0] is not None, \
            "EoI value has not yet been assigned a value even though " \
            "``generate_arrays`` completed. Implementation may not have " \
            "fully iterated through the input DataElement iterable."

        # Finish yielding any already-computed descriptor elements that are
        # past the last computed element index.
        for e, already_comp in elem_and_status_q:
            # If an element's already computed state is False at this point,
            # then this implementation's ``_generate_arrays`` method must not
            # have yielded enough arrays to fill vacancies that needed to be
            # filled.
            if not already_comp:
                raise IndexError(
                    "Implementation generator under-produced vectors to fill "
                    "descriptor elements in need of a vector (UUID `{}` not "
                    "filled).".format(e.uuid())
                )
            yield e

    #
    # Single-element-based convenience methods
    #

    def generate_one_array(self, data_elem):
        """
        Convenience wrapper around ``generate_arrays`` for the single-input
        case.

        See the documentation for the
        :meth:`DescriptorGenerator.generate_arrays` method for more
        information.

        :param smqtk.representation.DataElement data_elem:
            DataElement instance to be described.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor generator implementation.

        :return: Descriptor vector the given data as a ``numpy.ndarray``
            instance.
        :rtype: numpy.ndarray
        """
        return list(
            self.generate_arrays([data_elem])
        )[0]

    def generate_one_element(self, data_elem,
                             descr_factory=DFLT_DESCRIPTOR_FACTORY,
                             overwrite=False):
        """
        Convenience wrapper around ``generate_elements`` for the single-input
        case.

        See documentation for the
        :meth:`DescriptorGenerator.generate_elements` method for more
        information

        :param smqtk.representation.DataElement data_elem:
            DataElement instance to be described.
        :param smqtk.representation.DescriptorElementFactory descr_factory:
            DescriptorElementFactory instance to drive the generation of
            element instances with some parametrization.
        :param bool overwrite:
            By default, if a factory-produced DescriptorElement reports as
            containing a vector, we do not compute a descriptor again for the
            associated data element. If this is ``True``, however, we will
            generate descriptors for all input data elements, overwriting the
            vectors previously stored in the factory-produces descriptor
            elements.
        :raises IndexError: Underlying vector-producing generator either under
            or over produced vectors.

        :raises RuntimeError: Descriptor extraction failure of some kind.
        :raises ValueError: Given data element content was not of a valid type
            with respect to this descriptor generator implementation.

        :return: Result DescriptorElement instance. UUID of the generated
            DescriptorElement instance will reflect the UUID of the
            DataElement it was generated from.
        :rtype: smqtk.representation.DescriptorElement
        """
        return list(
            self.generate_elements([data_elem], descr_factory=descr_factory,
                                   overwrite=overwrite)
        )[0]
