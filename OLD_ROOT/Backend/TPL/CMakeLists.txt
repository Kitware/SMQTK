
# Add and build the sub TPL elements
include(ExternalProject)


###
# libSVM
#
set(libSVM_DIR "${CMAKE_CURRENT_LIST_DIR}/libsvm-3.17")
ExternalProject_Add(libSVM
  PREFIX            "${TPL_BUILD_PREFIX}"
  SOURCE_DIR        "${libSVM_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE   1
  BUILD_COMMAND     $(MAKE) lib
  INSTALL_DIR       "${TPL_LOCAL_INSTALL_PREFIX}"
  INSTALL_COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_LOCAL_INSTALL_PREFIX}/lib"
          COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_LOCAL_INSTALL_PREFIX}/include"
          COMMAND   "${CMAKE_COMMAND}" -E make_directory "${TPL_PYTHON_SP}"
          COMMAND   "${CMAKE_COMMAND}" -E copy libsvm.so.2 "${TPL_LOCAL_INSTALL_PREFIX}/lib"
          COMMAND   "${CMAKE_COMMAND}" -E copy svm.h "${TPL_LOCAL_INSTALL_PREFIX}/include"
          COMMAND   "${CMAKE_COMMAND}" -E copy python/svm.py "${TPL_PYTHON_SP}"
          COMMAND   "${CMAKE_COMMAND}" -E copy python/svmutil.py "${TPL_PYTHON_SP}"
  )


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
    #-DLATEX_OUTPUT_PATH:PATH=${CMAKE_CURRENT_LIST_DIR}/flann-1.8.4
    -DLATEX_OUTPUT_PATH:PATH=build
    "-DCMAKE_INSTALL_PREFIX:PATH=${TPL_LOCAL_INSTALL_PREFIX}"
  )

ExternalProject_Add_Step(flann install_python
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${TPL_PYTHON_SP}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory "${FLANN_DIR}/src/python/pyflann" "${TPL_PYTHON_SP}/pyflann"
  DEPENDEES install
  )


###
# Install rules for TPL-installed material
#
# trailing slash means "Everything under this directory"
install(DIRECTORY "${TPL_LOCAL_INSTALL_PREFIX}/"
        DESTINATION .
        USE_SOURCE_PERMISSIONS
        )
