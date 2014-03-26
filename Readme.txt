Debe estar definida la variable de entorno CUDA_SDK_INSTALL_PATH con el directorio de la SDK para que encuentre la biblioteca cutil.

Para crear el proyecto para KDE con cmake realizar los siguiente pasos:

1- Dentro del directorio donde se encuentran los códigos fuentes crear un nuevo directorio "build": mkdir build
2- Entrar a ese directorio: cd build
3- Ejecutar: cmake -DCMAKE_BUILD_TYPE="debug" -GKDevelop3 .. 


De esta forma los archivos del proyecto KDE se encuentran en Sources/build
Además de esa forma se crea el makefile, por lo que para compilar alcanza con ejecutar el make.



ENCODING
Los códigos están codificados en utf-8, y el EOF de Windows. Todo esto es facilmente configurable en el Kdevelop. Ir a Editor->Ajustes->Configurar Editor->Open/Save


CHEQUEO DE MEMORIA CON VALGRIND
valgrind --tool=memcheck --leak-check=full  --track-origins=yes --log-file=salida.log Ejecutable

Ejemplo:
valgrind --tool=memcheck --leak-check=full  --track-origins=yes --log-file=TestMlem.log MLEM reconSinogram2D_Siddon_TestProjector.par
