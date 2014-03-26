#  Linker CMAKE settings for the reconGPU shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Directorios para el linkeo.
LINK_DIRECTORIES(${CUDA_CUT_LIBRARIES} ${reconGPU_BINARY_DIR} ${recon_BINARY_DIR} ${data_BINARY_DIR} ${utils_BINARY_DIR})

# Shared/Static Libraries a linkear:
SET(LinkLibs ${CUDA_CUT_LIBRARIES} recon data utils)
message("Linklibs: ${LinkLibs}")



