project(smqtk_frame_extractor)
cmake_minimum_required(VERSION 2.8)

find_package(Boost 1.50 REQUIRED) #COMPONENTS
add_definitions(-DBOOST_ALL_NO_LIB)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(VXL REQUIRED)
include("${VXL_CMAKE_DIR}/UseVXL.cmake")

set(frame_extractor_srcs
    frame_extractor.cxx)

add_executable(frame_extractor
    ${frame_extractor_srcs})
target_link_libraries(frame_extractor
    vidl vul)

set_property(
    TARGET frame_extractor
    PROPERTY
        COMPILE_FLAGS "${flags}")

install(
  TARGETS     frame_extractor
  DESTINATION bin
  COMPONENT   tools
  )
