from .csift2 import (
    ColorDescriptor_Image_csift,
    ColorDescriptor_Image_transformedcolorhistogram,
    ColorDescriptor_Video_csift,
    ColorDescriptor_Video_transformedcolorhistogram
)

FEATURE_DESCRIPTOR_CLASS = [
    ColorDescriptor_Image_csift,
    ColorDescriptor_Image_transformedcolorhistogram,
    ColorDescriptor_Video_csift,
    ColorDescriptor_Video_transformedcolorhistogram,
]
