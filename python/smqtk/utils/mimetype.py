import mimetypes as pymimetypes
import threading


MT_CACHE_LOCK = threading.RLock()
MT_CACHE = None


def get_mimetypes():
    """
    Get the singleton SMQTK-adjusted mimetypes instance.

    This is desired due to the presence of some odd mimetype and extension
    conversion, e.g. image/jpeg -> .jfif.

    :return: SMQTK singleton MimeTypes instance.
    :rtype: mimetypes.MimeTypes

    """
    global MT_CACHE
    with MT_CACHE_LOCK:
        if MT_CACHE is None:
            MT_CACHE = pymimetypes.MimeTypes()
            # Remove .jfif
            MT_CACHE.types_map_inv[1]['image/jpeg'].remove('.jfif')
            # Remove .jpe
            MT_CACHE.types_map_inv[1]['image/jpeg'].remove('.jpe')
    return MT_CACHE
