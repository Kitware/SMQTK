get_filename_component(PLUGIN ${CMAKE_CURRENT_LIST_DIR} NAME)

add_web_client_test(
  tasks "${PROJECT_SOURCE_DIR}/plugins/${PLUGIN}/plugin_tests/gallerySpec.js" PLUGIN ${PLUGIN}
  SETUP_DATABASE "${CMAKE_CURRENT_LIST_DIR}/plugin_tests/fixtures/images.yml")

add_python_test(utils PLUGIN ${PLUGIN})
