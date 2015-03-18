from .csift import (
    ColorDescriptor_CSIFT_Image,
    ColorDescriptor_CSIFT_Video,
    ColorDescriptor_TCH_Image,
    ColorDescriptor_TCH_Video
)

FEATURE_DESCRIPTOR_CLASS = [
    ColorDescriptor_CSIFT_Image,
    ColorDescriptor_CSIFT_Video,
    ColorDescriptor_TCH_Image,
    ColorDescriptor_TCH_Video
]