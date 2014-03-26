#  Linker CMAKE settings for the recon shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Linking directories.
LINK_DIRECTORIES(${recon_BINARY_DIR} ${data_BINARY_DIR} ${utils_BINARY_DIR})

# Shared/Static Libraries a linkear:
SET(LinkLibs data utils)



