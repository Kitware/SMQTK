Examples
========


Simple Feature Computation
--------------------------

The following is a concrete example of performing feature computation for a set of ten butterfly images. It assumes you have set up the colordescriptor executable and python library in your `PATH` and `PYTHONPATH`.

.. code-block:: python

    # Import some butterfly data
    urls = ["http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/examples/{:03d}.jpg".format(i) for i in range(1,11)]
    from smqtk.representation.data_element.url_element import DataUrlElement
    el = [DataUrlElement(d) for d in urls]

    # Create a model. This assumes you have set up the colordescriptor executable.
    from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
    cd = get_descriptor_generator_impls()['ColorDescriptor_Image_csift'](model_directory='data', work_directory='work')
    cd.generate_model(el)

    # Set up a factory for our vector (here in-memory storage)
    from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
    from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement
    factory = DescriptorElementFactory(DescriptorMemoryElement, {})

    # Compute features on the first image
    result = cd.compute_descriptor(el[0], factory)
    result.vector()
    # array([ 0.        ,  0.01254855,  0.        , ...,  0.0035853 ,
    #         0.        ,  0.00388408])
