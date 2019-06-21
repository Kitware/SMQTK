import numpy as np
import PIL.Image
import six


def compute_saliency_map(base_image, descriptor_generator, augmenter,
                         blackbox):
    """
    Compute the saliency map for an image with respect to some black-box
    algorithm that transforms an image descriptor vector into a scalar
    value.

    :param np.ndarray base_image:
        Image to generate the saliency map over.
    :param smqtk.algorithms.DescriptorGenerator descriptor_generator:
        Some descriptor generation algorithm to transform an image to a
        feature space.
    :param smqtk.algorithms.ImageSaliencyAugmenter augmenter:
        Image augmentation algorithm to generate augmentation matrices
    :param smqtk.algorithms.SaliencyBlackbox blackbox:
        Black-box interface to generating saliency scores for input
        descriptors.

    :return: Saliency heat-map matrix with the same shape as the input image.
    :rtype: np.ndarray
    """
    # NOTE: Its possible that an augmenter may desire to create an absurd
    #       number of variations, possibly creating a memory issue. If that
    #       ever becomes an issue we may need to investigate passing
    #       DataElements in place of images here.
    augs, masks = augmenter.augment(base_image)

    # Mapping of element UUID to augmented image index
    #: :type: list[collections.Hashable]
    idx_to_uuid = []

    def iter_aug_img_data_elements():
        for a in augs:
            buff = six.BytesIO()
            # Choosing BMP format because there is little/no encoding
            # processing (raw format)
            PIL.Image.fromarray(a).save(buff, format="BMP")
            de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/bmp')
            idx_to_uuid.append(de.uuid())
            yield de

    # TODO: compute_descriptor_async really aught to just iterate in order
    #       of input data elements... Would remove the need for the
    #       book-keeping.
    uuid_to_desc = descriptor_generator.compute_descriptor_async(
        iter_aug_img_data_elements()
    )
    scalar_vec = blackbox.transform(
        (uuid_to_desc[uuid] for uuid in idx_to_uuid)
    )

    # TODO: Complete saliency map generation map and return
