from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import six
import numpy as np
import PIL
import copy
import cv2
import logging
from smqtk.algorithms.saliency import ImageSaliencyMapGenerator
from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement

__author__ = "alina.barnett@kitware.com"


def overlay_saliency_map(sa_map, org_img):
    """
    overlay the saliency map on top of original image
    :param sa_map: saliency map
    :type sa_map: numpy.array
    :param org_img: Original image
    :type org_img: numpy.array
    :return: Overlayed image
    :rtype: PIL Image
    """
    plt.switch_backend('agg')
    sizes = np.shape(sa_map)
    height = float(sizes[0])
    width = float(sizes[1])
    sa_map.resize(sizes[0],sizes[1])
    fig = plt.figure(dpi=int(height))
    fig.set_size_inches((width / height), 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(org_img)
    ax.imshow(sa_map, cmap='jet', alpha=0.5)
    fig.canvas.draw()
    np_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    im = PIL.Image.fromarray(np_data)
    im_size = np.shape(im)
    org_h =  im_size[1]
    org_w = im_size[0]
    im = im.resize((org_w, org_h), PIL.Image.BILINEAR)
    plt.close()
    return im


def highest_saliency_indices(scalar_scores, l):
    #returns the indices of the l (proportion) most salient masks
    second_sweep_size = int(np.floor(l * len(scalar_scores)))
    diffss = np.array(scalar_scores)
    indices = (-diffss).argsort()[:second_sweep_size]
    return indices


def highest_saliency_indices_of_subs(scalar_scores, l):
    #find the indices of the submasks under the highest saliency masks
    indices = highest_saliency_indices(scalar_scores, l)
    rel_subs = [submasks_from_mask(i,6,4) for i in indices]
    rel_subs = np.unique(np.asarray(rel_subs)).tolist()
    return rel_subs


def find_intersection(masks, region):
    #returns the masks in masks that intersect with region
    masked_masks = []
    for mask in masks:
        intersection = np.max(np.multiply((1 - mask), (1 - region)))
        if intersection > 0:
            masked_masks.append(mask)

    return masked_masks


class Fast_ImageSaliencyMapGenerator(ImageSaliencyMapGenerator):
    """
    Interface for the method of generation of a saliency map given an image
    augmentation and blackbox algorithms.
    """
    def __init__(self, threshold=0.2):
        
        self.thresh = threshold
   
    def get_config(self):

        return {
            'threshold': 0.2,
        }

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        #TODO:appropriate returns
        return plt and np
    
    def generate(self, base_image, augmenter, descriptor_generator,
                 blackbox):
        """
        Generate an image saliency heat-map matrix given a blackbox's behavior
        over the descriptions of an augmented base image.
        :param numpy.ndarray base_image:
            Numpy matrix of the format [height, width [,channel]] that is to be augmented.
        :param ImageSaliencyAugmenter augmenter:
            Augmentation algorithm following
            the :py:class:`ImageSaliencyAugmenter` interface.
        :param smqtk.algorithms.DescriptorGenerator descriptor_generator:
            A descriptor generation algorithm following
            the :py:class:`smqtk.algorithms.DescriptorGenerator` interface.
        :param SaliencyBlackbox blackbox:
            Blackbox algorithm implementation following
            the :py:class:`SaliencyBlackbox` interface.
        :return: A :py:class:`PIL.Image` of the same [height, width]
            shape as the input image matrix but of floating-point type within
            the range of [0,1], where areas of higher value represent more
            salient regions according to the given blackbox algorithm.
        :rtype: PIL.Image
        """
        def weighted_avg(scalar_vec, masks):
            masks = masks.reshape(-1,224,224,1)
            cur_filters = copy.deepcopy(masks[:,:,:,0])
            count = masks.shape[0] - np.sum(cur_filters, axis=0)
            cur_filters = cur_filters.astype(np.float64)

            for i in range(len(cur_filters)):
                cur_filters[i] = np.multiply((1 - cur_filters[i]), np.clip(scalar_vec[i], a_min=0.0, a_max=None), casting='unsafe')
            
            filter_sum = np.sum(cur_filters, axis=0)
            res_sa = np.divide(filter_sum, count, casting='unsafe')
            res_sa = np.nan_to_num(res_sa, copy=False) #sets nan to 0
            sa_max = np.max(res_sa)
            res_sa = np.clip(res_sa, a_min=sa_max * self.thresh, a_max = None)

            return res_sa

        org_hw = np.shape(base_image)[0:2]
        base_image_resized = cv2.resize(base_image,(224,224),interpolation=cv2.INTER_NEAREST)
        augs, masks = augmenter.augment_roughpass(np.array(base_image_resized))
        idx_to_uuid = []
        def iter_aug_img_data_elements():
            for a in augs:
               buff = six.BytesIO()
               (a).save(buff, format="png")
               de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/png')
               idx_to_uuid.append(de.uuid())
               yield de

        uuid_to_desc = descriptor_generator.compute_descriptor_async(iter_aug_img_data_elements())

        scalar_vec = blackbox.transform((uuid_to_desc[uuid] for uuid in idx_to_uuid))
        
        #find relevant region
        l = 0.3
        rel_masks_indices = highest_saliency_indices(scalar_vec, l)
        rel_masks = np.take(masks, rel_masks_indices, axis=0)
        region = np.ones(masks[0].shape, dtype=np.int64)
        for mask in rel_masks:
            region = np.multiply(region, mask)

        augs, masks = augmenter.augment(base_image_resized)

        #find relevant submasks
        rel_submasks = find_intersection(masks, region)
        augs = augmenter.generate_masked_imgs(rel_submasks, np.asarray(base_image_resized))
        #print("{} out of {} masks are relevant.".format(len(rel_submasks), len(masks)))

        idx_to_uuid = []
        def iter_aug_img_data_elements():
            for a in augs:
               buff = six.BytesIO()
               (a).save(buff, format="png")
               de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/png')
               idx_to_uuid.append(de.uuid())
               yield de

        uuid_to_desc = descriptor_generator.compute_descriptor_async(iter_aug_img_data_elements())

        scalar_vec = blackbox.transform((uuid_to_desc[uuid] for uuid in idx_to_uuid))
        final_sal_map = weighted_avg(scalar_vec, np.asarray(rel_submasks))
        final_sal_map_resized = cv2.resize(final_sal_map,(org_hw),interpolation=cv2.INTER_NEAREST)
        sal_map_ret= overlay_saliency_map(final_sal_map_resized, base_image)
        return sal_map_ret

IMG_SALIENCY_GENERATOR_CLASS=Fast_ImageSaliencyMapGenerator
