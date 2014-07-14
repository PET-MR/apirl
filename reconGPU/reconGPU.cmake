#  CMAKE source file configuration for the reconGPU shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Add current directory to the nvcc include line.

CUDA_INCLUDE_DIRECTORIES(
  ${reconGPU_SOURCE_DIR}
  ${FIND_CUDA_DIR}
  ${CUDA_CUT_INCLUDE_DIR}
  ${recon_Headers_Dir}
  ${reconGPU_Headers_Dir}
  ${data_Headers_Dir} 
  ${utils_Headers_Dir}
)
INCLUDE_DIRECTORIES(${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir} ${reconGPU_Headers_Dir})

message(STATUS "reconGPU_Headers_Dir: ${reconGPU_Headers_Dir}")
#FILE(GLOB recon_Sources RELATIVE ${recon_SOURCE_DIR}/src ${recon_SOURCE_DIR}/src/*.cpp)
#FILE(GLOB recon_Headers RELATIVE ${recon_SOURCE_DIR}/inc ${recon_SOURCE_DIR}/inc/*.h)
# Viejo:
# No pongo el flag RELATIVE porque me devuelve solo el nombre del archivo.
# Ahora: Pongo el relative para que kdevelop me lo tome, el truco está en poner ${data_SOURCE_DIR}
# y no ${data_SOURCE_DIR}/src
FILE(GLOB reconGPU_Sources RELATIVE ${reconGPU_SOURCE_DIR} ${reconGPU_SOURCE_DIR}/src/*.cpp)
FILE(GLOB reconGPU_cuSources RELATIVE ${reconGPU_SOURCE_DIR} ${reconGPU_SOURCE_DIR}/src/*.cu)
FILE(GLOB reconGPU_Headers RELATIVE ${reconGPU_SOURCE_DIR} ${reconGPU_SOURCE_DIR}/inc/*.h)
FILE(GLOB reconGPU_cuHeaders RELATIVE ${reconGPU_SOURCE_DIR} ${reconGPU_SOURCE_DIR}/inc/*.cuh)


SOURCE_GROUP("reconGPU\\Sources" FILES ${reconGPU_Sources} ${reconGPU_cuSources})
message(STATUS "my sources: ${reconGPU_Sources} ${reconGPU_cuSources}")
SOURCE_GROUP("reconGPU\\Headers" FILES ${reconGPU_Headers} ${reconGPU_cuHeaders})
message(STATUS "my headers: ${reconGPU_Headers} ${reconGPU_cuHeaders}")


