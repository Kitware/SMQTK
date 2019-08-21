"""
This utility generates saliency maps.

## Double has indicates a note for Bhavan.

## This version does not have torch as a dependency. I have plans for another version that uses torch.
"""
import os
import numpy as np
import copy
import PIL.Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import six
import matplotlib.pyplot as plt

from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.representation.data_element import from_uri
#from smqtk.utils.image_utils import overlay_saliency_map ##could use this function instead of what's written here for overlaying saliency map.


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
    masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.int64)
    i = 0
    for r in np.arange(0, image_size[0] - window_size[0] + 1, stride[0]):
        for c in np.arange(0, image_size[1] - window_size[1] + 1, stride[1]):
            masks[i, r:r + window_size[0], c:c + window_size[1]] = 0
            i += 1

    masks = masks.reshape(-1, *image_size, 1)

    return masks


def get_rc(index, grid_size):
    
    r = index // grid_size
    c = index % grid_size
    
    return r, c


def get_index(row, col, grid_size):
    
    index = grid_size * row + col
    
    return index


def submasks_from_mask(mask_num, grid_size, subgrid_size):
    """
    Given the index of the larger mask, finds the submasks beneath it.
    
    :param mask_num: the index of the larger mask
    :type mask_num: int 
    
    :param grid_size: the side length of the larger grid size
    :type grid_size: int 
    
    :param subgrid_size: the side length of the subgrid within each segment of the big grid
    :type subgrid_size: int
    
    :return: indices of the submasks
    :rtype: list of ints
    """

    m = mask_num
    g = grid_size
    d = subgrid_size
    
    if m >= g*g or m < 0:
        raise ValueError("The mask number is not on the specified grid.")
    
    r_m, c_m = get_rc(m,g)
    
    r_n0 = d * r_m
    c_n0 = d * c_m

    submasks = []
    
    for sub_row in range(d):
        for sub_col in range(d):
            #print("row: {}, col: {}, index: {}".format(r_n0 + sub_row * 1, c_n0 + sub_col * 1, get_index(r_n0 + sub_row * 1, c_n0 + sub_col * 1, g)))
            submasks.append(get_index(r_n0 + sub_row * 1, c_n0 + sub_col * 1, g * d))
    
    return submasks


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

    grid_num_r = (image_size[0] - window_size) // stride + 1
    grid_num_c = (image_size[1] - window_size) // stride + 1
    mask_num = grid_num_r * grid_num_c
    print('mask_num: {}'.format(mask_num))
    masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.int64)
    i = 0
    for r in np.arange(0, image_size[0] - window_size + 1, stride):
        for c in np.arange(0, image_size[1] - window_size + 1, stride):
            masks[i, r:r + window_size, c:c + window_size] = 0
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
        masked_img = np.multiply(mask, img, casting='unsafe')
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


def print_img(img, path='/home/local/KHQ/alina.barnett/AlinaCode/imgs/sa_imgs/print_from_debugger.jpg'):
    img = PIL.Image.fromarray(img.astype(np.uint8))
    img.save(path)
    

def generate_saliency_map_Bo(T_img, descriptor_generator, query_img):
    """
    Find the saliency map for an image. The context for the saliency map is 
    score in the relevancy index.

    :param T_img: An image for which we want to generate a saliency map.
    :type T_img: PIL Image #may instead want to make this a uid? probably in the future, yes

    :param descriptor_generator: The descriptor generator used by the relevancy 
    index. 
    :type descriptor_generator: DescriptorGenerator, a custom class
    
    :query_img: The query image.
    :type query_img: PIL Image

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
    
    Q_img = query_img.resize((224,224),resample=PIL.Image.BICUBIC)
    query_img_path = os.path.join(path, "query_img.png")
    Q_img.save(query_img_path)
    
    start=datetime.now()
    #masks = generate_block_masks_from_gridsize(image_size=(T_img.shape[1],T_img.shape[0]), grid_size=(15,15))
    masks = generate_block_masks(window_size=56, stride=14, image_size=(T_img.shape[1],T_img.shape[0]))
    masked_imgs = generate_masked_imgs(masks, T_img)
    masked_img_paths = []
    print(datetime.now()-start)
    
    
    print("Masks file i/o...")
    start=datetime.now()
    for i, masked_img in enumerate(masked_imgs):
        img = PIL.Image.fromarray(masked_img.astype(np.uint8))
        save_path = os.path.join(path, "masked_img_{:04d}.png".format(i))
        img.save(save_path)
        masked_img_paths.append(save_path)
    
    print(datetime.now()-start)
    
    print("Computing descriptors...") 
    start=datetime.now()
    des = [from_uri(path) for path in masked_img_paths]
    m = descriptor_generator.compute_descriptor_async(des)
    print(datetime.now()-start)
    print("Put descriptors into list...") 
    start = datetime.now()
    img_fs = [m[de.uuid()] for de in des]
    print(datetime.now()-start)
    img_fs.append(descriptor_generator.compute_descriptor(from_uri(unmasked_img_path)))
    img_fs.append(descriptor_generator.compute_descriptor(from_uri(query_img_path))) ##Uses query_img instead of unmasked
    
    print("Adding up saliency maps...")
    start=datetime.now()
    cur_filters = copy.deepcopy(masks)
    count = masks.shape[0] - np.sum(cur_filters, axis=0)
    #count = np.ones(count.shape)
    # apply the dis diff onto the corresponding masks
    for i in range(len(cur_filters)):
        diff = np.sum(img_fs[i].vector() - img_fs[-1].vector()) - np.sum(img_fs[-2].vector() - img_fs[-1].vector()) ##Bo's method
        cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(diff, a_min=0.0, a_max=None)

    res_sa = np.sum(cur_filters, axis=0) / count
    sa_threshhold = 0.2 ##I picked this value to get better looking images.
    sa_max = np.max(res_sa)
    res_sa = np.clip(res_sa, a_min=sa_max * sa_threshhold, a_max = None)
    print(datetime.now()-start)
    print("Overlaying saliency map...")
    start=datetime.now()
    S_img = overlay_saliency_map(res_sa, T_img)
    print(datetime.now()-start)
    
    return S_img


def save_imgs_to_file(imgs_list, path, rootname):
    filepaths = []
    for i, masked_img in enumerate(imgs_list):
        img = PIL.Image.fromarray(masked_img.astype(np.uint8))
        save_path = os.path.join(path, "{}{:04d}.png".format(rootname,i))
        img.save(save_path)
        filepaths.append(save_path)
    return filepaths


def combine_maps(masks, img_fs, RI_scores):
    cur_filters = copy.deepcopy(masks)
    count = masks.shape[0] - np.sum(cur_filters, axis=0)
    cur_filters = cur_filters.astype(np.float64)
    
    for i in range(len(cur_filters)):
        diff = RI_scores[img_fs[i]] - RI_scores[img_fs[-1]]
        cur_filters[i] = np.multiply((1 - cur_filters[i]), np.clip(diff, a_min=0.0, a_max=None), casting='unsafe')
    
    filter_sum = np.sum(cur_filters, axis=0)
    res_sa = np.divide(filter_sum, count, casting='unsafe')
    res_sa = np.nan_to_num(res_sa, copy=False) #sets nan to 0
    sa_threshhold = 0.4 ##Picked this value to get better looking images.
    sa_max = np.max(res_sa)
    res_sa = np.clip(res_sa, a_min=sa_max * sa_threshhold, a_max = None)

    return res_sa


def combine_maps_subs(masks, rel_submasks, img_fs, subimg_fs, RI_scores, sub_RI_scores):
    all_masks = np.concatenate((masks, rel_submasks),axis=0)
    cur_filters = copy.deepcopy(all_masks)
    count = all_masks.shape[0] - np.sum(cur_filters, axis=0)
    cur_filters = cur_filters.astype(np.float64)

    for i in range(len(masks)):
        diff = RI_scores[img_fs[i]] - RI_scores[img_fs[-1]]
        cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(diff, a_min=0.0, a_max=None)
    for i in range(len(rel_submasks)):
        j = i + len(masks)
        diff = sub_RI_scores[subimg_fs[i]] - sub_RI_scores[subimg_fs[-1]]
        cur_filters[j] = (1.0 - cur_filters[j]) * np.clip(16*diff, a_min=0.0, a_max=None)
    #pdb.set_trace()
    filter_sum = np.sum(cur_filters, axis=0)
    res_sa = np.divide(filter_sum, count, casting='unsafe')
    res_sa = np.nan_to_num(res_sa, copy=False) #sets nan to 0
    sa_threshhold = 0.2 ##Picked this value to get better looking images.
    sa_max = np.max(res_sa)
    res_sa = np.clip(res_sa, a_min=sa_max * sa_threshhold, a_max = None)

    return res_sa


def highest_saliency_indices(img_fs, RI_scores, l):
    #returns the indices of the l (proportion) most salient masks
    diffs = []
    for i in range(len(img_fs) - 1):
        diffs.append(RI_scores[img_fs[i]] - RI_scores[img_fs[-1]])
    second_sweep_size = int(np.floor(l * len(diffs)))
    diffss = np.array(diffs)
    indices = (-diffss).argsort()[:second_sweep_size]

    return indices


def highest_saliency_indices_of_subs(img_fs, RI_scores, l):
    #find the indices of the submasks under the highest saliency masks
    indices = highest_saliency_indices(img_fs, RI_scores, l)
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


def generate_saliency_map(T_img, descriptor_generator, relevancy_index, ADJs):
    """
    Find the saliency map for an image. The context for the saliency map is 
    score in the relevancy index.

    :param T_img: An image for which we want to generate a saliency map.
    :type T_img: PIL Image #may instead want to make this a uid? probably in the future, yes

    :param descriptor_generator: The descriptor generator used by the relevancy 
    index. 
    :type descriptor_generator: DescriptorGenerator, a custom class
    (
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
    big_start = datetime.now()
    path = "/home/local/KHQ/alina.barnett/AlinaCode/imgs/TEMP/masked_imgs"
    
    T_img = T_img.resize((224,224),resample=PIL.Image.BICUBIC)
    unmasked_img_path = os.path.join(path, "unmasked_img.png")
    T_img.save(unmasked_img_path)
    T_img = np.array(T_img)
    
    print("Generating masks and masked imgs")
    start=datetime.now()
    #masks = generate_block_masks_from_gridsize(image_size=(T_img.shape[1],T_img.shape[0]), grid_size=(15,15))
    masks = generate_block_masks(window_size=56, stride=9, image_size=(T_img.shape[1],T_img.shape[0]))
    masked_imgs = generate_masked_imgs(masks, T_img)
    print(datetime.now()-start)
    
    print("Masks file i/o...")
    start=datetime.now()
    masked_img_paths = save_imgs_to_file(masked_imgs, path, "masked_img_")
    print(datetime.now()-start)
    
    print("Computing descriptors...") 
    start=datetime.now()
    masked_img_des = [from_uri(path) for path in masked_img_paths]
    m = descriptor_generator.compute_descriptor_async(masked_img_des)
    print(datetime.now()-start)

    print("Put descriptors into list...") 
    start = datetime.now()
    img_fs = [m[de.uuid()] for de in masked_img_des]
    print(datetime.now()-start)
    img_fs.append(descriptor_generator.compute_descriptor(from_uri(unmasked_img_path)))

    print("Ranking...")
    start = datetime.now()
    relevancy_index.build_index(img_fs)
    RI_scores = relevancy_index.rank(*ADJs) 
    print(datetime.now()-start)
    
    print("Adding up saliency maps...")
    start = datetime.now()
    res_sa = combine_maps(masks, img_fs, RI_scores)
    print(datetime.now()-start)

    print("Overlaying saliency map...")
    start=datetime.now()
    S_img = overlay_saliency_map(res_sa, T_img)
    print(datetime.now()-start)
    
    print("Total time: ")
    print(datetime.now()-big_start)
    return S_img


def generate_saliency_map_fast(T_img, descriptor_generator, relevancy_index, ADJs):
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
    big_start = datetime.now()
    path = "/home/local/KHQ/alina.barnett/AlinaCode/imgs/TEMP/masked_imgs"
    
    T_img = T_img.resize((224,224),resample=PIL.Image.BICUBIC)
    unmasked_img_path = os.path.join(path, "unmasked_img.png")
    T_img.save(unmasked_img_path)
    T_img = np.array(T_img)
    
    print("Masks and masked img generation for first sweep...")
    start=datetime.now()
    #masks = generate_block_masks_from_gridsize(image_size=(T_img.shape[1],T_img.shape[0]), grid_size=(6,6))
    masks = generate_block_masks(window_size=56, stride=35, image_size=(T_img.shape[1],T_img.shape[0]))
    masked_imgs = generate_masked_imgs(masks, T_img)
    print(datetime.now()-start)
    
    print("Masks file i/o...")
    start=datetime.now()
    masked_img_paths = save_imgs_to_file(masked_imgs, path, "masked_img_")
    print(datetime.now()-start)
    
    print("Data elements from URIs...") 
    start=datetime.now()
    des = [from_uri(path) for path in masked_img_paths]
    print(datetime.now()-start)

    print("Computing descriptors for first sweep...") 
    start=datetime.now()
    m = descriptor_generator.compute_descriptor_async(des)
    print(datetime.now()-start)
    
    print("Put descriptors into list...") 
    start = datetime.now()
    img_fs = [m[de.uuid()] for de in des]
    print(datetime.now()-start)
    img_fs.append(descriptor_generator.compute_descriptor(from_uri(unmasked_img_path)))
    
    print("Ranking...")
    start = datetime.now()
    relevancy_index.build_index(img_fs)
    RI_scores = relevancy_index.rank(*ADJs)
    print(datetime.now()-start)
    
    print("Selecting elaboration region...")
    start=datetime.now()
    l = 0.05
    #rel_subs = highest_saliency_indices_of_subs(img_fs, RI_scores, l)
    rel_masks_indices = highest_saliency_indices(img_fs, RI_scores, l)
    rel_masks = np.take(masks, rel_masks_indices, axis=0)
    region = np.ones(masks[0].shape, dtype=np.int64)
    for mask in rel_masks:
        region = np.multiply(region, mask)
    print(datetime.now()-start)
    
    print("Generate submasks for second sweep...")
    start=datetime.now()
    submasks = generate_block_masks(window_size=56, stride=9, image_size=(T_img.shape[1],T_img.shape[0]))
    print(datetime.now()-start)

    print("Find intersection...")
    start=datetime.now()
    rel_submasks = find_intersection(submasks, region)
    submasked_imgs = generate_masked_imgs(rel_submasks, T_img)
    print("{} out of {} masks are relevant.".format(len(rel_submasks), len(submasks)))
    print(datetime.now()-start)
    
    print("Submasks file i/o...")
    start=datetime.now()
    submasked_img_paths = save_imgs_to_file(submasked_imgs, path, "submasked_img_")
    print(datetime.now()-start)
    
    #pdb.set_trace()

    print("Data elements from URIs...") 
    start=datetime.now()
    des = [from_uri(path) for path in submasked_img_paths]
    print(datetime.now()-start)

    print("Computing descriptors for second sweep...") 
    start=datetime.now()
    m = descriptor_generator.compute_descriptor_async(des)
    print(datetime.now()-start)
    
    print("Put descriptors into list...") 
    start = datetime.now()
    subimg_fs = [m[de.uuid()] for de in des]
    print(datetime.now()-start)
    subimg_fs.append(descriptor_generator.compute_descriptor(from_uri(unmasked_img_path)))
    
    print("Subs Reranking...")
    start = datetime.now()
    relevancy_index.build_index(subimg_fs)
    sub_RI_scores = relevancy_index.rank(*ADJs)
    print(datetime.now()-start)
    
    print("Adding up saliency maps...")
    start=datetime.now()
    #res_sa = combine_maps(masks, img_fs, RI_scores)
    res_sa = combine_maps(np.asarray(rel_submasks), subimg_fs, sub_RI_scores)
    #res_sa = combine_maps_subs(masks, rel_submasks, img_fs, subimg_fs, RI_scores, sub_RI_scores)
    print(datetime.now()-start)

    print("Overlaying saliency map...")
    start=datetime.now()
    S_img = overlay_saliency_map(res_sa, T_img)
    print(datetime.now()-start)
    
    print("Total time: ")
    print(datetime.now()-big_start)

    return S_img


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

    :return: Saliency heat-map image  with the same shape as the input image.
    :rtype: PIL Image
    """

    org_hw = base_image.size
    base_image_PIL = base_image.resize((224,224) ,PIL.Image.BILINEAR)
    base_image_np = np.array(base_image_PIL)
    augs, masks = augmenter.augment(base_image_np)
    
       
    idx_to_uuid = []
    def iter_aug_img_data_elements():
        for a in augs:
            buff = six.BytesIO()
            (a).save(buff, format="png")
            de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/png')
            idx_to_uuid.append(de.uuid())
            yield de
         
    uuid_to_desc=descriptor_generator.compute_descriptor_async(iter_aug_img_data_elements())

    scalar_vec = blackbox.transform((uuid_to_desc[uuid] for uuid in idx_to_uuid))
    
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

    def weighted_avg(scalar_vec,masks):
        masks = masks.reshape(-1,224,224,1)
        cur_filters = copy.deepcopy(masks[:,:,:,0])
        count = masks.shape[0] - np.sum(cur_filters, axis=0)
        count = np.ones(count.shape)

        for i in range(len(cur_filters)):
            cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(scalar_vec[i], a_min=0.0, a_max=None)
        res_sa = np.sum(cur_filters, axis=0) / count
        sa_threshhold = 0.2 
        sa_max = np.max(res_sa)
        res_sa = np.clip(res_sa, a_min=sa_max * sa_threshhold, a_max = None)
        return res_sa
    
    final_sal_map=weighted_avg(scalar_vec,masks)
    print("Overlaying saliency map...")
    sal_map_ret=overlay_saliency_map(final_sal_map,base_image_np)
    #sal_map_ret=Image.fromarray(final_sal_map)
    sal_map_ret=sal_map_ret.resize((org_hw), PIL.Image.BILINEAR)
   
    return sal_map_ret

