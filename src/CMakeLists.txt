option(SMQTK_BUILD_FRAME_EXTRACTOR
  "Enable building of the frame_extractor tool. This requires VXL."
  OFF
  )
if( SMQTK_BUILD_FRAME_EXTRACTOR )
  add_subdirectory( frame_extractor )
endif()
