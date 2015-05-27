#SET(LinkLibs CUDA_Reconstruction ${LinkLibs})
#  Linker Cmake settings for cmd executables commands
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Linking Directories. Shared libraries: recon, utils and data.
LINK_DIRECTORIES(${cmdGPU_BINARY_DIR} ${reconGPU_BINARY_DIR} ${utils_BINARY_DIR} ${data_BINARY_DIR} ${CUDA_CUT_LIBRARY} ${recon_BINARY_DIR})
SET(LinkLibs data reconGPU.lib utils data recon)


