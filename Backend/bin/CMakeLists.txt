# Install pre-compiled binaries

install(
  PROGRAMS colorDescriptor colorDescriptorCompatible
           ffmpeg ffprobe ffserver frame_extractor
           unify_videos.py vcd_mpi_scheduler.py
  DESTINATION bin
  )
