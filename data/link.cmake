#  Linker CMAKE settings for the data shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010
  
# Directorios para el linkeo.
LINK_DIRECTORIES(${data_BINARY_DIR} ${utils_BINARY_DIR} )


# Shared/Static Libraries a linkear:
SET(LinkLibs utils)



