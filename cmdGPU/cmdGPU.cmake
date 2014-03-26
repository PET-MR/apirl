#  CMAKE source file configuration for the cmdGPU commands of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

#Directorios de la Librería de Reconstrucción de Imágenes
INCLUDE_DIRECTORIES(${reconGPU_Headers_Dir} ${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir} ${CUDA_CUT_INCLUDE_DIR})

#  Tengo que generar un grupo de archivos por cada comando.

#FILE(GLOB cmd_Sources RELATIVE ${cmd_SOURCE_DIR} ${cmd_SOURCE_DIR}/*.cpp)
SET(cuMLEM_Sources "src/cuMLEM.cpp")
#SET(cuOSEM_Sources "src/cuOSEM.cpp")


SOURCE_GROUP("cmdGPU\\cuMLEM" FILES ${cuMLEM_Sources})
#SOURCE_GROUP("cmdGPU\\cuOSEM" FILES ${cuOSEM_Sources})
