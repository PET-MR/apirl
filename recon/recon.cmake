#  CMAKE source file configuration for the recon shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

INCLUDE_DIRECTORIES(${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir})

#FILE(GLOB recon_Sources RELATIVE ${recon_SOURCE_DIR}/src ${recon_SOURCE_DIR}/src/*.cpp)
#FILE(GLOB recon_Headers RELATIVE ${recon_SOURCE_DIR}/inc ${recon_SOURCE_DIR}/inc/*.h)
# Viejo:
# No pongo el flag RELATIVE porque me devuelve solo el nombre del archivo.
# Ahora: Pongo el relative para que kdevelop me lo tome, el truco está en poner ${data_SOURCE_DIR}
# y no ${data_SOURCE_DIR}/src
FILE(GLOB recon_Sources RELATIVE ${recon_SOURCE_DIR} ${recon_SOURCE_DIR}/src/*.cpp)
FILE(GLOB recon_Headers RELATIVE ${recon_SOURCE_DIR} ${recon_SOURCE_DIR}/inc/*.h)


SOURCE_GROUP("recon\\Sources" FILES ${recon_Sources})
message(STATUS "my sources: ${recon_Sources}")
SOURCE_GROUP("recon\\Headers" FILES ${recon_Headers})
message(STATUS "my headers: ${recon_Headers}")
