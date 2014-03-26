#  CMAKE source file configuration for the cmd commands of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010
INCLUDE_DIRECTORIES(${cmd_Headers_Dir} ${recon_Headers_Dir} ${data_Headers_Dir} ${utils_Headers_Dir})

#  Tengo que generar un grupo de archivos por cada comando.

#FILE(GLOB cmd_Sources RELATIVE ${cmd_SOURCE_DIR} ${cmd_SOURCE_DIR}/*.cpp)
# MLEM:
SET(MLEM_Sources "src/MLEM.cpp")
#OSEM:
SET(OSEM_Sources "src/OSEM.cpp")
# GenerateImage:
SET(GenerateImage_Sources "src/generateImage.cpp")
# GenerateSystemMatrix:
SET(GenerateSystemMatrix_Sources "src/generateSystemMatrix.cpp")
# Backprojection:
SET(Backproject_Sources "src/backproject.cpp")
# Projection:
SET(Project_Sources "src/project.cpp")
# ACF:
SET(GenerateACFs_Source "src/generateACFs.cpp")


SOURCE_GROUP("cmd\\MLEM" FILES ${MLEM_Sources})
SOURCE_GROUP("cmd\\OSEM" FILES ${OSEM_Sources})
SOURCE_GROUP("cmd\\GenerateImage" FILES ${GenerateImage_Sources})
SOURCE_GROUP("cmd\\GenerateSystemMatrix" FILES ${GenerateSystemMatrix_Sources})
SOURCE_GROUP("cmd\\Backproject" FILES ${Backproject_Sources})
SOURCE_GROUP("cmd\\Project" FILES ${Project_Sources})
SOURCE_GROUP("cmd\\GenerateACFs" FILES ${GenerateACFs_Source})
message(STATUS "cmd sources: ${MLEM_Sources} ${GenerateImage_Sources} ${Backproject_Sources} ${Project_Sources}")
#SOURCE_GROUP("cmd\\OSEM" FILES ${OSEM_Sources})

