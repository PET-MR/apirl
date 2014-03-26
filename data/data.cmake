#  CMAKE source file configuration for the data shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

INCLUDE_DIRECTORIES(${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir} ${Medcon_Dir})

# Viejo:
# No pongo el flag RELATIVE porque me devuelve solo el nombre del archivo.
# Ahora: Pongo el relative para que kdevelop me lo tome, el truco está en poner ${data_SOURCE_DIR}
# y no ${data_SOURCE_DIR}/src
FILE(GLOB data_Sources RELATIVE ${data_SOURCE_DIR} ${data_SOURCE_DIR}/src/*.cpp)
FILE(GLOB data_Headers RELATIVE ${data_SOURCE_DIR} ${data_SOURCE_DIR}/inc/*.h)
FILE(GLOB Medcon_Sources RELATIVE ${data_SOURCE_DIR} ${data_SOURCE_DIR}/medcon/*.h ${data_SOURCE_DIR}/medcon/*.c)

SOURCE_GROUP("data\\Sources" FILES ${data_Sources})
message(STATUS "my sources: ${data_Sources}")
SOURCE_GROUP("data\\Headers" FILES ${data_Headers})
message(STATUS "my headers: ${data_Headers}")
SOURCE_GROUP("data\\Medcon" FILES ${Medcon_Sources})
message(STATUS "Medcon sources: ${Medcon_Sources}")


