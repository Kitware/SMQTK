Additional required binaries for various SMQTK system components:

- ffmpeg -> video metadata probing
- frame_extractor -> build by this repository but additionally requiring Boost
                     and VXL dependencies, this is required for frame
                     extraction from video files
- colorDescriptor -> This executable is required for the CSIFT image descriptor
                     (as well as the `video` CSIFT descriptor which works over
                     extracted images). This can be downloaded from:
                         http://koen.me/research/colordescriptors
                     Note that a license is required for commercial purposes.
