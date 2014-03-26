#  Linker Cmake settings for cmd executables commands
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Linking Directories. Shared libraries: recon, utils and data.
LINK_DIRECTORIES(${cmd_BINARY_DIR} ${recon_BINARY_DIR} ${utils_BINARY_DIR} ${data_BINARY_DIR})

# Shared/Static Libraries a linkear:
SET(LinkLibs recon data utils)
