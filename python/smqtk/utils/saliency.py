"""
This utility generates saliency maps.

## Double has indicates a note for Bhavan.

## This version does not have torch as a dependency. I have plans for another version that uses torch.
"""
import os
import numpy as np
import pdb
import copy
import PIL.Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element import from_uri
#from smqtk.utils.image_utils import overlay_saliency_map ##could use this function instead of what's written here for overlaying saliency map.

__author__ = "alina.barnett@kitware.com"

def generate_block_masks_from_gridsize(image_size, grid_size=(5,5)): 
    """
    Generating the sliding window style masks.
    
    :param image_size: the mask size which should be the same as the image size
    :type image_size: tuple 
    
    :param grid_size: the number of rows and columns
    :type grid_size: tuple of ints (default: (5, 5))
    
    :return: the sliding window style masks
    :rtype: numpy array 
    """
    window_size = (image_size[0]//grid_size[0], image_size[1]//grid_size[1])
    stride = window_size
    
    grid_num_r = (image_size[0] - window_size[0]) // stride[0] + 1
    grid_num_c = (image_size[1] - window_size[1]) // stride[1] + 1
    mask_num = grid_num_r * grid_num_c
    print('mask_num {}'.format(mask_num))
    masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
    i = 0
    for r in np.arange(0, image_size[0] - window_size[0] + 1, stride[0]):
        for c in np.arange(0, image_size[1] - window_size[1] + 1, stride[1]):
            masks[i, r:r + window_size[0], c:c + window_size[1]] = 0.0
            i += 1

    masks = masks.reshape(-1, *image_size, 1)

    return masks

def generate_block_masks(window_size, stride, image_size):
    """
    Generating the sliding window style masks.
    
    :param window_size: the block window size (with value 0, other areas with value 1)
    :type window_size: int
    
    :param stride: the sliding step
    :type stride: int
    
    :param image_size: the mask size which should be the same to the image size
    :type image_size: tuple
    
    :return: the sliding window style masks
    :rtype: numpy array
    """
    ##I took out the parts where we fetch from file because generating the masks was so fast. Could be put back in I guess.
    grid_num_r = (image_size[0] - window_size) // stride + 1
    grid_num_c = (image_size[1] - window_size) // stride + 1
    mask_num = grid_num_r * grid_num_c
    print('mask_num {}'.format(mask_num))
    masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
    i = 0
    for r in tqdm(np.arange(0, image_size[0] - window_size + 1, stride), total=grid_num_r, desc="Generating mask rows..."):
        for c in np.arange(0, image_size[1] - window_size + 1, stride):
            masks[i, r:r + window_size, c:c + window_size] = 0.0
            i += 1

    masks = masks.reshape(-1, *image_size, 1)
    
    return masks

def generate_masked_imgs(masks, img):
    """
    Apply the N filters/masks onto one input image
    :param index: mask index
    :return: masked images
    """
    masked_imgs = []
    for mask in masks:
        masked_img = np.multiply(mask, img)
        masked_imgs.append(masked_img)

    return masked_imgs

def overlay_saliency_map(sa_map, org_img): #future: rewrite this to be scipy instead of PIL
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
    #im = np.asarray(im)

    return im


def print_img(img, path='/home/local/KHQ/alina.barnett/AlinaCode/imgs/sa_imgs/output.jpg'):
    img = PIL.Image.fromarray(img.astype(np.uint8))
    img.save(path)

def generate_saliency_map(T_img, descriptor_generator, relevancy_index, ADJs):
    """
    Find the saliency map for an image. The context for the saliency map is 
    score in the relevancy index.

    :param T_img: An image for which we want to generate a saliency map.
    :type T_img: PIL Image #may instead want to make this a uid? probably in the future, yes

    :param descriptor_generator: The descriptor generator used by the relevancy 
    index. 
    :type descriptor_generator: DescriptorGenerator, a custom class
    
    :param relevancy_index: The relevancy index item.
    :type relevancy_index: RelevancyIndex, a custom class
    
    :param ADJs: Adjudicated images to build the relevancy index with.
    :type ADJs: tuple containing 2 lists. First list is an iterable of positive 
    exemplar DescriptorElement instances (type pos: 
    collections.Iterable[smqtk.representation.DescriptorElement]). Second list is 
    an iterable of negative exemplar DescriptorElement instances.

    :return: An saliency map image which has saliency added onto `T_img`. 
    Same size as T_img.
    :rtype: PIL image #may instead want to make this a uid? At some point.

    [1] Note: 
    """
    #temp holding path
    path = "/home/local/KHQ/alina.barnett/AlinaCode/imgs/TEMP/masked_imgs"
        
    #resize T_img
    #T_img = (PIL.Image.fromarray(T_img))
    T_img = T_img.resize((224,224),resample=PIL.Image.BICUBIC)
    unmasked_img_path = os.path.join(path, "unmasked_img.png")
    T_img.save(unmasked_img_path)
    T_img = np.array(T_img)
    
    #masks = generate_block_masks_from_gridsize(image_size=(T_img.shape[1],T_img.shape[0]), grid_size=(15,15))
    masks = generate_block_masks(window_size=56, stride=14, image_size=(T_img.shape[1],T_img.shape[0]))
    masked_imgs = generate_masked_imgs(masks, T_img)
    masked_img_paths = []
    
    
    print("Masks file i/o")
    for i, masked_img in enumerate(masked_imgs):
        img = PIL.Image.fromarray(masked_img.astype(np.uint8))
        save_path = os.path.join(path, "masked_img_{:04d}.png".format(i))
        img.save(save_path)
        masked_img_paths.append(save_path)
    
    
    print("Computing descriptors and ranking...") 
    img_fs = [descriptor_generator.compute_descriptor(from_uri(path)) for path in masked_img_paths] ##Need to redo this part so that it uses compute_descriptor_async instead for better speed up.
    img_fs.append(descriptor_generator.compute_descriptor(from_uri(unmasked_img_path)))
    relevancy_index.build_index(img_fs) ##to get Bo's method: there is no need for this and the following line because the relveancy index isn't used
    RI_scores = relevancy_index.rank(*ADJs) 
    
    print("Adding up saliency maps...")
    cur_filters = copy.deepcopy(masks)
    count = masks.shape[0] - np.sum(cur_filters, axis=0)
    #count = np.ones(count.shape)
    # apply the dis diff onto the corresponding masks
    for i in range(len(cur_filters)):
        diff = RI_scores[img_fs[i]] - RI_scores[img_fs[-1]] ##to get to Bo's method: instead of subtracting relevancy scores, instead take the descriptors/feature vector differences here
        cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(diff, a_min=0.0, a_max=None)

    res_sa = np.sum(cur_filters, axis=0) / count
    sa_threshhold = 0.2 ##I picked this value to get better looking images.
    sa_max = np.max(res_sa)
    res_sa = np.clip(res_sa, a_min=sa_max * sa_threshhold, a_max = None)
    print("Overlaying saliency map...")
    S_img = overlay_saliency_map(res_sa, T_img)
    
    return S_img
