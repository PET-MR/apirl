#  CMAKE source file configuration for the utils shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

INCLUDE_DIRECTORIES(${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir})

# Viejo:
# No pongo el flag RELATIVE porque me devuelve solo el nombre del archivo.
# Ahora: Pongo el relative para que kdevelop me lo tome, el truco está en poner ${data_SOURCE_DIR}
# y no ${data_SOURCE_DIR}/src
FILE(GLOB utils_Sources RELATIVE ${utils_SOURCE_DIR} ${utils_SOURCE_DIR}/src/*.cpp)
FILE(GLOB utils_Headers RELATIVE ${utils_SOURCE_DIR} ${utils_SOURCE_DIR}/inc/*.h)


SOURCE_GROUP("utils\\Sources" FILES ${utils_Sources})
message(STATUS "my sources: ${utils_Sources}")
SOURCE_GROUP("utils\\Headers" FILES ${utils_Headers})
message(STATUS "my headers: ${utils_Headers}")
