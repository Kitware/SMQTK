option(SMQTK_BUILD_FRAME_EXTRACTOR
  "Enable building of the frame_extractor tool. This requires VXL."
  OFF
  )
if( SMQTK_BUILD_FRAME_EXTRACTOR )
  add_subdirectory( frame_extractor )
endif()

option(SMQTK_BUILD_CNN_FEATURE_EXTRACTOR
  "Enable building of the deep learning feature extraction. This requires Caffe."
  OFF
  )
if( SMQTK_BUILD_CNN_FEATURE_EXTRACTOR )
  add_subdirectory( cnn_feature_extractor )
endif()
