
install(
  PROGRAMS run_masir.py
           colorDescriptor
           FeatureManagerServer.py
           generate_data_models_from_ingest.py
           ingest_images.py
  DESTINATION bin
  )

