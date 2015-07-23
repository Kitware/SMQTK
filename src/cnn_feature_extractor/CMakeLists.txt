project(smqtk_cnn_feature_extractor)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(Caffe REQUIRED)
add_definitions( ${Caffe_DEFINITIONS} )
include_directories(SYSTEM ${Caffe_INCLUDE_DIRS})

set(cnn_feature_extractor_srcs
    cnn_feature_extractor.cxx
    )

add_executable(cnn_feature_extractor
    ${cnn_feature_extractor_srcs})
target_link_libraries(cnn_feature_extractor
    ${Caffe_LIBRARIES})

set_property(
    TARGET cnn_feature_extractor
    PROPERTY
        COMPILE_FLAGS "${flags}")

install(
  TARGETS     cnn_feature_extractor
  DESTINATION bin
  COMPONENT   tools
  )

configure_file( "${SMQTK_SOURCE_DIR}/data/caffenet/file_list.txt.in"
                "${SMQTK_BINARY_DIR}/data/caffenet/file_list.txt" )
