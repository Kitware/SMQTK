project(SMQTK)
cmake_minimum_required(VERSION 2.8)


###
# Versioning
#
set(SMQTK_VERSION_MAJOR 0)
set(SMQTK_VERSION_MINOR 6)
set(SMQTK_VERSION_PATCH 1)
set(SMQTK_VERSION ${SMQTK_VERSION_MAJOR}.${SMQTK_VERSION_MINOR}.${SMQTK_VERSION_PATCH})


###
# Determine where to install python things in our install tree
#
find_package(PythonInterp REQUIRED)
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig; print distutils.sysconfig.get_python_lib(prefix='')"
  RESULT_VARIABLE proc_success
  OUTPUT_VARIABLE python_site_packages
  )

if(NOT ${proc_success} EQUAL 0)
  message(FATAL_ERROR "Request for python site-packages location failed with error code: ${proc_success}")
else()
  string(STRIP "${python_site_packages}" python_site_packages)
  message(STATUS "Python site-packages to install into: ${python_site_packages}")
endif()


###
# TPL
#
# local installation control variables
set(TPL_BUILD_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/TPL/build")
set(TPL_LOCAL_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/TPL/install")
set(TPL_PYTHON_SP "${TPL_LOCAL_INSTALL}/${python_site_packages}")
add_subdirectory( TPL )


###
# SMQTK Library
#
add_subdirectory( src )
add_subdirectory( python )
add_subdirectory( bin )


###
# File configuration
#
# Environment setup scripts
configure_file("${SMQTK_SOURCE_DIR}/setup_env.build.sh.in"
               "${SMQTK_BINARY_DIR}/setup_env.build.sh"
               @ONLY
               )
configure_file("${SMQTK_SOURCE_DIR}/setup_env.install.sh.in"
               "${SMQTK_BINARY_DIR}/setup_env.install.sh"
               @ONLY
               )


###
# System Installation
#

install(DIRECTORY DESTINATION var/data)

if(UNIX)
  # Due to super-user privileges being required to run things like groupadd,
  # these commands will only be executed when we have such permissions (like
  # installing to the system via sudo, etc.)
  install(CODE
          "execute_process(COMMAND groupadd -r SMQTK
                           COMMAND chown -R :SMQTK var
                           COMMAND chmod -R 2775 var
                           WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\")"
          )
endif()

install(DIRECTORY   docs/
        DESTINATION share/doc/SMQTK
        )

install(DIRECTORY   etc/
        DESTINATION etc
        )

install(FILES requirements.conda.txt
        DESTINATION .
        )
install(FILES requirements.pip.txt
        DESTINATION .
        )

install(FILES "${SMQTK_BINARY_DIR}/setup_env.install.sh"
        DESTINATION .
        RENAME setup_smqtk.sh
        )

###
# CPack stuff
#

# TODO: Define package dependencies here
#       - just non-python? or include system python deps here, too?
set(SMQTK_DEPS "")

if(EXISTS /etc/redhat-release)
  file(READ /etc/redhat-release RHEL_VERSION)
  string(REGEX REPLACE ".*release ([^\\. ]*).*" "\\1" RHEL_VERSION "${RHEL_VERSION}")
  set(CPACK_SYSTEM_NAME "el${RHEL_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_RPM_PACKAGE_AUTOREQPROV "no")
  set(CPACK_RPM_PACKAGE_REQUIRES "${SMQTK_DEPS}")
  set(CPACK_RPM_PACKAGE_RELOCATABLE TRUE)
else()
  set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CPACK_PACKAGE_NAME "SMQTK")
set(CPACK_PACKAGE_VENDOR "Kitware, Inc.")
set(CPACK_PACKAGE_CONTACT "kitware@kitware.com")
#set(CPACK_MONOLITHIC_INSTALL true)
set(CPACK_PACKAGE_VERSION_MAJOR ${SMQTK_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${SMQTK_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${SMQTK_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION       ${SMQTK_VERSION})
#set(CPACK_RESOURCE_FILE_LICENSE [NONE YET])
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")

include(CPack)
