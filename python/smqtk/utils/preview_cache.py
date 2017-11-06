import imageio
import logging
import os
import shutil

import six

from smqtk.utils import file_utils, video_utils
from smqtk.utils.mimetype import get_mimetypes


MIMETYPES = get_mimetypes()


class PreviewCache (object):
    """
    Create and cache saved located of preview images for data elements.
    """

    # Preview generation methods based on content type
    # - types can either be specific types or just type classes (i.e. "image" or
    #   "video"), which are fall-backs when a data element's specific content
    #   type is not found in the mapping.
    # - This should be populated with methods that take two arguments, the first
    #   being a DataElement instance, and the second being the directory to
    #   place generated files under. These methods should return the full path
    #   to the generated preview image.
    #: :type: dict[collections.Hashable, collections.Callable]
    PREVIEW_GEN_METHOD = {}

    @property
    def _log(self):
        return logging.getLogger('.'.join([self.__module__,
                                           self.__class__.__name__]))

    def __init__(self, cache_dir):
        """
        :param cache_dir: Directory to cache preview image elements into.
        :type cache_dir: str
        """
        self._cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        # Cache of preview images for data elements encountered.
        #: :type: dict[collections.Hashable, str]
        self._preview_cache = {}
        self._video_work_dir = os.path.join(cache_dir, 'tmp_video_work')

    def __del__(self):
        """
        Cleanup after ourselves.
        """
        for fp in six.itervalues(self._preview_cache):
            os.remove(fp)

    def get_preview_image(self, elem):
        """
        Get the filepath to the preview image for the given data element.

        :raises ValueError: Do not know how to generate a preview image for the
            given element's content type.

        :param elem: Data element to generate a preview image for.
        :type elem: smqtk.representation.DataElement

        :return: Path to the preview image for the given data element.
        :rtype: str

        """
        if elem.uuid() in self._preview_cache:
            return self._preview_cache[elem.uuid()]

        # else, generate preview image based on content type / content class
        if elem.content_type() in self.PREVIEW_GEN_METHOD:
            self._log.debug("Generating preview image based on content type: "
                            "%s", elem.content_type)
            file_utils.safe_create_dir(self._cache_dir)
            fp = self.PREVIEW_GEN_METHOD[elem.content_type()](self, elem,
                                                              self._cache_dir)
        else:
            content_class = elem.content_type().split('/', 1)[0]
            if content_class in self.PREVIEW_GEN_METHOD:
                self._log.debug("Generating preview image based on content "
                                "class: %s", content_class)
                file_utils.safe_create_dir(self._cache_dir)
                fp = self.PREVIEW_GEN_METHOD[content_class](self, elem,
                                                            self._cache_dir)
            else:
                raise ValueError("No preview generation method for the data "
                                 "element provided, of content type '%s'."
                                 % elem.content_type())
        self._preview_cache[elem.uuid()] = fp
        return fp

    # noinspection PyMethodMayBeStatic
    def gen_image_preview(self, elem, output_dir):
        """
        Copy temporary image to specified output filepath.

        :param elem: Data element to get the preview image for.
        :type elem: smqtk.representation.DataElement

        :param output_dir: Directory to save generated image to.
        :type output_dir: str

        """
        output_fp = os.path.join(
            output_dir,
            "%s%s" % (str(elem.uuid()),
                      MIMETYPES.guess_extension(elem.content_type()))
        )
        if not os.path.isfile(output_fp):
            tmp_img_fp = elem.write_temp()
            shutil.copy(tmp_img_fp, output_fp)
            elem.clean_temp()
        return output_fp

    def gen_video_preview(self, elem, output_dir):
        """
        Copy temporary image to specified output filepath.

        :param elem: Data element to get the preview image for.
        :type elem: smqtk.representation.DataElement

        :param output_dir: Directory to save generated image to.
        :type output_dir: str

        """
        output_fp = os.path.join(output_dir,
                                 "%s.gif" % elem.uuid())
        if not os.path.isfile(output_fp):
            tmp_vid_fp = elem.write_temp()
            interval = 0.5  # ~2fps gif
            fm = video_utils.ffmpeg_extract_frame_map(
                self._video_work_dir, tmp_vid_fp,
                second_interval=interval
            )
            img_arrays = []
            for frm_num in sorted(fm.keys()):
                img_arrays.append(imageio.imread(fm[frm_num]))
            imageio.mimwrite(output_fp, img_arrays, duration=interval)
            elem.clean_temp()
        return output_fp


PreviewCache.PREVIEW_GEN_METHOD = {
    "image": PreviewCache.gen_image_preview,
    "video": PreviewCache.gen_video_preview,
}
