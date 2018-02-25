# Add and build TPL elements
include(ExternalProject)

option(SMQTK_BUILD_LIBSVM "Enable building of libSVM." ON)
option(SMQTK_BUILD_FLANN "Enable building of FLANN." ON)

# Variables defined by parent module:
#   TPL_BUILD_PREFIX  - Where build files will go
#   TPL_LOCAL_INSTALL - Local install tree
#   TPL_PYTHON_SP     - relative path of standarad Python site-packages dir


###
# libSVM
#
if (SMQTK_BUILD_LIBSVM)

  # Required a make command to build. Required its existance.
  # Pick "make" executable based on generator chosen or environment variable
  # set.
  if (CMAKE_GENERATOR MATCHES ".*Makefiles")
    set(MAKE_EXECUTABLE "$(MAKE)")
  elseif(WIN32)
    set(MAKE_EXECUTABLE nmake -f Makefile.win)
  elseif(NOT "#@$ENV{MAKE}" STREQUAL "#@")
    set(MAKE_EXECUTABLE $ENV{MAKE})
  else()
    find_program(MAKE_EXECUTABLE make)
  endif()

  # Set the libsvm we want. Can't just use a suffix, since linux uses *.so.2
  if(WIN32)
    set(LIBSVM_NAME libsvm.dll)
  else()
    set(LIBSVM_NAME libsvm.so.2)
  endif()


  if( NOT MAKE_EXECUTABLE )
    message( FATAL_ERROR "Could not find 'make', required to build libsvm." )
  endif()

  set(libSVM_DIR "${CMAKE_CURRENT_LIST_DIR}/libsvm-3.1-custom")
  ExternalProject_Add(libSVM
    PREFIX            "${TPL_BUILD_PREFIX}"
    SOURCE_DIR        "${libSVM_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE   1
    BUILD_COMMAND     ${MAKE_EXECUTABLE} lib
    INSTALL_DIR       "${TPL_LOCAL_INSTALL}"
    INSTALL_COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_LOCAL_INSTALL}/lib"
            COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_LOCAL_INSTALL}/include"
            COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_PYTHON_SP}"
            COMMAND   "${CMAKE_COMMAND}" -E copy ${LIBSVM_NAME} "${TPL_LOCAL_INSTALL}/lib"
            COMMAND   "${CMAKE_COMMAND}" -E copy svm.h "${TPL_LOCAL_INSTALL}/include"
            COMMAND   "${CMAKE_COMMAND}" -E copy python/svm.py "${TPL_PYTHON_SP}"
            COMMAND   "${CMAKE_COMMAND}" -E copy python/svmutil.py "${TPL_PYTHON_SP}"
  )
endif()


###
# FLANN
#
# The install step likes to try to install the python bindings into the place
# where the python used for the installation is located. This is not desired.
# We would like to put those bindings in our local install tree ONLY. In order
# to do this, we turn off the internal python bindings switch and manually
# copy over the bindings directory as a post-install step. These bindings in
# their raw form are sufficiently able to locate the local install's flann
# library, and thus are sufficiently functional.
#
if (SMQTK_BUILD_FLANN)
  set(FLANN_DIR "${CMAKE_CURRENT_LIST_DIR}/flann-1.8.4")
  ExternalProject_Add(flann
    PREFIX            "${TPL_BUILD_PREFIX}"
    SOURCE_DIR        "${FLANN_DIR}"
    BUILD_IN_SOURCE   1
    CMAKE_GENERATOR   ${gen}
    CMAKE_ARGS
      -DBUILD_MATLAB_BINDINGS:BOOL=OFF
      -DBUILD_CUDA_LIB:BOOL=OFF
      -DBUILD_PYTHON_BINDINGS:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=${SMQTK_BUILD_EXAMPLES}
      #-DLATEX_OUTPUT_PATH:PATH=${CMAKE_CURRENT_LIST_DIR}/flann-1.8.4
      -DLATEX_OUTPUT_PATH:PATH=build
      "-DCMAKE_INSTALL_PREFIX:PATH=${TPL_LOCAL_INSTALL}"
    )

  ExternalProject_Add_Step(flann install_python
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${TPL_PYTHON_SP}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${FLANN_DIR}/src/python/pyflann" "${TPL_PYTHON_SP}/pyflann"
    DEPENDEES install
    )
endif()


###
# Install rules for TPL-installed material
#
# trailing slash means "Everything under this directory"
if (SMQTK_BUILD_LIBSVM OR SMQTK_BUILD_FLANN)
  install(DIRECTORY "${TPL_LOCAL_INSTALL}/"
          DESTINATION .
          USE_SOURCE_PERMISSIONS
          )
endif()
