#include <CUDA_Geometry.h>
#include <Utilities.h>
#include <Michelogram.h>
#include <CUDA_Siddon.h>
//#include <math.h>
#include <Images.h>
#include <CudaMlemMichelogram.h>
//#include <cuda_ZEL_utils.h>	// Hay un par de funciones para la inicializaciïṡẄn del Device y el llmado seguro a kernels
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>
#include <string>
#include <Logger.h>
//#include <math_functions.h>

//#define __cplusplus
// This implementation of MLEM calculates online the System Matrix. Also calculates the sensitivity
// image before starting the iterative reconstruction. The sensitivity image is saved in the same directory
// of execution.
//#ifdef __cplusplus
//	extern "C" 
//#endif
__global__ void UpdatePixelValue(float *RawImage, float *FactorRawImage, float* SumAij);
__global__ void CUDA_Forward_Projection (float* volume, float* michelogram);
__global__ void CUDA_Forward_Projection (float* volume, float* michelogram, float* michelogram_measured);
__global__ void CUDA_Back_Projection (float* michelogram, float* volume);
__global__ void CUDA_Sensibility_Image (float* volume);
__global__ void CUDA_Divide_Projection (float* michelogram_measured, float* michelogram_estimated);
__global__ void CUDA_Calcular_Likelihood (float* estimated_michelogram, float* measured_michelogram);
//void SaveRawFile(float* array, unsigned int N, char* path);


__device__ __constant__ float cuda_values_phi[NPROJ_MICH];
__device__ __constant__ float cuda_values_r[NR_MICH];
__device__ __constant__ float cuda_values_z[NZ_MICH];
__device__ __constant__ unsigned int cuda_threads_per_block;
__device__ __constant__ unsigned int cuda_threads_per_block_update_pixel;
__device__ __constant__ unsigned int cuda_nr_splitter;
__device__ __constant__ unsigned int cuda_rows_splitter;
__device__ __constant__ SizeImage cuda_image_size;
__device__ __constant__ int cudaBinsSino2D;
// Agrego estas dos porque la nueva clase SizeImage no tiene los campos RFOV y ZFOV
__device__  __constant__ float cudaRFOV;
__device__  __constant__ float cudaZFOV;
__device__  __constant__ float OffsetZ;
__device__  __constant__ float cudaRscanner;
//	
__device__  __constant__ SizeMichelogram cuda_michelogram_size;
__device__ float cuda_likelihood;

/// Constructor
CudaMlemMichelogram::CudaMlemMichelogram(Michelogram* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSensitivityImageFromFile, dim3 blockSizeProj, dim3 blockSizeBackproj, dim3 blockSizeIm) : CudaMlem(cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSensitivityImageFromFile, blockSizeProj, blockSizeBackproj, blockSizeIm)
{
  
}

/// Método privado para la reconstrucción de Michelogramas
bool CudaMlemMichelogram::Reconstruct()
{
	/// Punteros donde voy a tener los datos crudos, y de esa forma poder copiarlos al GPU.
	float* SumAij; 
	float* RawMichelogram;
	float* RawImage;
	float* RawAuxImage;
	float* RawProjectedMichelogram;
	float* angleValues_radians;
	/// Variable para el control de error en las funciones de cuda:
	cudaError_t my_cuda_error;
	/// Punteros que serán utilizado con cuda, o sea apuntarán a memoria de video.
	float* cuda_volume;
	float* cuda_aux_volume;
	float* cuda_michelogram;
	float* cuda_projected_michelogram;
	float* cuda_sum_aij;
	/// Varibles que utilizaba en versiones anteriores para identificar los threadid,
	/// pero que por ahora no voy a vovler a usar:
	int NR_Splitter, rowSplitter;
	
	/// String de c para utlizar en los mensajes de logueo.
	char c_string[512];
	/// Puntero del array con los tiempos de reconstrucción por iteración.
	float* timesIteration_mseg;
	/// Puntero del array con los tiempos de backprojection por iteración.
	float* timesBackprojection_mseg;
	/// Puntero del array con los tiempos de forwardprojection por iteración.
	float* timesForwardprojection_mseg;
	/// Puntero del array con los tiempos de pixel update por iteración.
	float* timesPixelUpdate_mseg;
	
	/// Cantidad de bloques que entran
	/// Pido memoria para los arrays, que deben tener tantos elementos como iteraciones:
	timesIteration_mseg = (float*)malloc(sizeof(float)*this->numIterations);
	timesBackprojection_mseg = (float*)malloc(sizeof(float)*this->numIterations);
	timesForwardprojection_mseg = (float*)malloc(sizeof(float)*this->numIterations);
	timesPixelUpdate_mseg = (float*)malloc(sizeof(float)*this->numIterations);

	/// El vector de likelihood puede haber estado alocado previamente por lo que eso realloc. Tiene
	/// un elemento más porque el likelihood es previo a la actualización de la imagen, o sea que inicia
	/// con el initialEstimate y termina con la imagen reconstruida.
	if(this->likelihoodValues == NULL)
	{
		/// No había sido alocado previamente así que utilizo malloc.
		this->likelihoodValues = (float*)malloc(sizeof(float)*(this->numIterations +1));
	}
	else
	{
		/// Ya habí­a sido alocado previamente, lo realoco.
		this->likelihoodValues = (float*)realloc(this->likelihoodValues, sizeof(float)*(this->numIterations + 1));
	}

	/// Termino de generar los tamaños de ejecución. Los tamaños
	/// de los bloques ya fueron cargados, hay que cargar los de las grillas, a partir de los bloques y el tamaño de los datos.
	gridSizeProjector = dim3((int)((inputProjection->NProj * inputProjection->NR) / blockSizeProjector.x)+1, inputProjection->NZ * inputProjection->NZ, 1);

	gridSizeBackprojector = gridSizeProjector;

	gridSizeImageUpdate = dim3(sizeReconImage.nPixelsY,  sizeReconImage.nPixelsZ, (int)(sizeReconImage.nPixelsX / blockSizeImageUpdate.x + 1));
	
	/// En principio esto no lo necesitaría, trabajo directamente con punteros.
	/// Michelograma auxiliar donde se van calculando las sucesivas proyecciones
	/// Michelograma auxiliar donde se van calculando las sucesivas proyecciones
	Michelogram* ProjectedMichelogram = new Michelogram(inputProjection->NProj, inputProjection->NR, inputProjection->NZ,
		inputProjection->Span, inputProjection->MaxRingDiff, inputProjection->RFOV, inputProjection->ZFOV);
	ProjectedMichelogram->FillConstant(0);
	Image* BackprojectedImagen = new Image(sizeReconImage);
	BackprojectedImagen->fillConstant(0);
	/// BackprojectedImagen->RFOV = Imagen->RFOV;
	/// BackprojectedImagen->ZFOV = Imagen->ZFOV;

	/// Hago el log de la reconstrucción:
	Logger* logger = new Logger(logFileName);
	/// Escribo el título y luego los distintos parámetros de la reconstrucción:
	logger->writeLine("######## CUDA ML-EM Reconstruction #########");
	logger->writeValue("Name", this->outputFilenamePrefix);
	logger->writeValue("Type", "ML-EM");
	sprintf(c_string, "%d", this->numIterations);
	logger->writeValue("Iterations", c_string);
	logger->writeValue("Input Projections", "Michelogram");
	sprintf(c_string, "%d[r] x %d[ang]", inputProjection->NR, inputProjection->NProj);
	logger->writeValue("Size of Sinogram2D",c_string);
	sprintf(c_string, "%d", inputProjection->NZ);
	logger->writeValue("Rings", c_string);
	sprintf(c_string, "%d", sizeReconImage.nDimensions);
	logger->writeValue("Image Dimensions", c_string);
	sprintf(c_string, "%d[x] x %d[y] x %d[z]", this->sizeReconImage.nPixelsX, this->sizeReconImage.nPixelsY, this->sizeReconImage.nPixelsZ);
	logger->writeValue("Image Size", c_string);

	/// Inicializo el volumen a reconstruir con un valor cte
	reconstructionImage->fillConstant(0.5);
	
	/// Genero los datos en vectores crudos en el CPU para poder copiarlos a memoria de GPU.
	Image* AuxImage = new Image(sizeReconImage);
	AuxImage->fillConstant(0);	// Inicializo el volumen en cero.
	RawImage = reconstructionImage->getPixelsPtr();
	RawAuxImage = AuxImage->getPixelsPtr();
	SumAij = sensitivityImage->getPixelsPtr();
	RawMichelogram = inputProjection->RawData();
	RawProjectedMichelogram = ProjectedMichelogram->RawData();
	
	// Number of Pixels
	unsigned int NPixels = reconstructionImage->getPixelCount();
	// Number of bins in Michelogram
	unsigned int NBins = inputProjection->NZ * inputProjection->NZ * inputProjection->NProj * inputProjection->NR;
		
	// Initialization of SizeMichelogram
	SizeMichelogram MySizeMichelogram;
	MySizeMichelogram.NProj = inputProjection->NProj;
	MySizeMichelogram.NR = inputProjection->NR;
	MySizeMichelogram.NZ = inputProjection->NZ;
	MySizeMichelogram.RFOV = reconstructionImage->getFovRadio();
	MySizeMichelogram.ZFOV = reconstructionImage->getFovHeight();
	///////////// INICALIZACIÓN DE GPU 
	if(!initCuda (0, logger))
	{
		return false;
	}
	/// Las variables splitter que creo que no voy a voler a usar:
	NR_Splitter = MySizeMichelogram.NR / blockSizeProjector.x;
	rowSplitter = reconstructionImage->getSize().nPixelsX / blockSizeImageUpdate.x;
	
	///////////// LOCACIÓN DE MEMORIA EN GPU\\\\\\\\\\\\	

	my_cuda_error = cudaMalloc((void**) &cuda_sum_aij, sizeof(float)*NPixels);
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_volume, sizeof(float)*NPixels));
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_aux_volume, sizeof(float)*NPixels));
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_michelogram, sizeof(float)*NBins));
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_projected_michelogram, sizeof(float)*NBins));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_volume, RawImage,sizeof(float)*NPixels,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_aux_volume, RawAuxImage, sizeof(float)*NPixels,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_michelogram, RawMichelogram, sizeof(float)*NBins,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_projected_michelogram, RawProjectedMichelogram, sizeof(float)*NBins,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemset(cuda_sum_aij, 0,sizeof(float)*NPixels));
	// Load constant memory with geometrical and other useful values
	/// Los ángulos los cargo en radianes, para evitar hacer la conversión en cada thread.
	angleValues_radians = inputProjection->Sinograms2D[0]->getAnglesInRadians();
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_phi, angleValues_radians, sizeof(float)*inputProjection->NProj));
	float* rValues_mm = (float*)malloc(sizeof(float)*(inputProjection->NR));
	for(int i = 0; i < inputProjection->NR; i++)
	{
	  rValues_mm[i] = inputProjection->Sinograms2D[0]->getRValue(i);
	}
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_r, rValues_mm, sizeof(float)*inputProjection->NR));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_z, inputProjection->ZValues, sizeof(float)*inputProjection->NZ));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_threads_per_block, &(blockSizeProjector.x), sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_threads_per_block_update_pixel, &(blockSizeImageUpdate.x), sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_rows_splitter, &rowSplitter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_image_size, &sizeReconImage, sizeof(SizeImage)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaRFOV, &MySizeMichelogram.RFOV, sizeof(MySizeMichelogram.RFOV)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaZFOV, &MySizeMichelogram.ZFOV, sizeof(MySizeMichelogram.ZFOV)));
	int binsSino2D = MySizeMichelogram.NR * MySizeMichelogram.NProj;
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaBinsSino2D, &binsSino2D, sizeof(binsSino2D)));
	
	/// Esto después hay que cambiarlo! Tiene que ir en la clase Michelogram!!!!!!!!!!!!!
	/// Necesito tener el dato del zfov del michelograma, que no lo tengo accesible ahora. Lo pongo a mano, pero
	/// cambiarlo de maner aurgente lo antes posible.!!!
	/// Ente el offsetZ lo calculaba en base al FOV del sinograma, ahora que fov es el de la imagen adquirida. Debo
	/// centrar dicho FOV en el FOV del sinograma y calcular el offsetZ relativo. Esto sería el valor mínimo de Z de la
	/// imagen a reconstruir. Lo puedo obtener del zFOV de la imagen o del sizePixelZ_mm.
	/// Lo mejor sería que el slice central sea el z=0, entonces no deberíamos modificar nada. Pero habría que cambiar
	/// varias funciones para que así sea. Por ahora queda así.
	float offsetZvalue = (SCANNER_ZFOV - inputProjection->ZFOV)/2;
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(OffsetZ, &offsetZvalue, sizeof(offsetZvalue)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaRscanner, &(inputProjection->rScanner), sizeof(inputProjection->rScanner)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_michelogram_size, &MySizeMichelogram, sizeof(SizeMichelogram)));

	
	/// Guardo en el Log la Estructura de Kernels Usada:
	/// Arranco con el log de los resultados:
	strcpy(c_string,"_______CUDA CONFIG_______");
	logger->writeLine(c_string, strlen(c_string));
	strcpy(c_string," Dimensiones kernels Proj y BackProj:");
	logger->writeLine(c_string, strlen(c_string));
	sprintf(c_string, "%d x %d [threads]",  blockSizeProjector.x,  blockSizeProjector.y);
	logger->writeValue("  Blocks Dimensions", c_string);
	sprintf(c_string, "%d x %d x %d [blocks]",  gridSizeProjector.x, gridSizeProjector.y, gridSizeProjector.z);
	logger->writeValue("  Grids Dimensions", c_string);
	strcpy(c_string, " Dimensiones kernel Pixel Update:");
	logger->writeLine(c_string, strlen(c_string));
	sprintf(c_string, "%d x %d [threads]",  blockSizeImageUpdate.x,  blockSizeImageUpdate.y);
	logger->writeValue("  Blocks Dimensions", c_string);
	sprintf(c_string, "%d x %d x %d [blocks]",  gridSizeImageUpdate.x, gridSizeImageUpdate.y, gridSizeImageUpdate.z);
	logger->writeValue("  Grids Dimensions", c_string);

	/// Me fijo si la sensibility image la tengo que cargar desde archivo o calcularla
	if(sensitivityImageFromFile)
	{
		/// Leo el sensibility volume desde el archivo
		if(sensitivityImage->readFromInterfile((char*) sensitivityFilename.c_str()) == false)
		{
		  logger->writeLine("Error al leer sensitivity image.");
		  logger->writeLine(sensitivityImage->getError());
		  printf("Error al leer sensitivity image: %s", sensitivityImage->getError());
		  return false;
		}
		/// Si lei bien la imagen la cargo en la memoria de la placa. Al leerla de un archivo, se
		/// realoca memoria, así que vuelvo a cargar el puntero SumAij.
		SumAij = sensitivityImage->getPixelsPtr();
		CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_sum_aij, sensitivityImage->getPixelsPtr(), sizeof(float)*NPixels,cudaMemcpyHostToDevice));
	}
	else
	{
		/// Calculo el sensibility volume
		// check if kernel execution generated and error
		/// Generating SumAij matrix
		CUDA_Sensibility_Image<<<gridSizeProjector, blockSizeProjector>>>(cuda_sum_aij);	
		my_cuda_error = cudaThreadSynchronize();
		#ifndef _DEBUG
		CUT_CHECK_ERROR("Kernel execution failed");
		#endif
		/// Copio la memoria de GPU al puntero de la sensitivity image
		CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(sensitivityImage->getPixelsPtr(),cuda_sum_aij,sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
		string sensitivityFileName = outputFilenamePrefix;
		sensitivityFileName.append("_sensitivity");
		/// Guardo la imagen en formato interfile.
		sensitivityImage->writeInterfile((char*)sensitivityFileName.c_str());
	}
	//Start with the iteration
	printf("Iniciando Reconstrucción...\n");
	/// Arranco con el log de los resultados:
	strcpy(c_string, "_______RECONSTRUCCION_______");
	logger->writeLine(c_string, strlen(c_string));
	/// Inicio el clock. La función clock no me anda bien para ejecutar el código de cpu,
	/// pareciera que no cuenta el tiempo en que se están ejecutando los kernels, y solo
	/// considera el tiempo que transcurre en el host (CPU).
	cudaThreadSynchronize();
	//clock_t initialClock = clock();
	cudaEvent_t start, stop, startIteration, stopForwardProj, startBackProj, stopBackProj, stopIteration;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startIteration);
	cudaEventCreate(&stopForwardProj);
	cudaEventCreate(&startBackProj);
	cudaEventCreate(&stopBackProj);
	cudaEventCreate(&stopIteration);
	/// Evento de inicio:
	cudaEventRecord(start, 0);
	for(unsigned int t = 0; t < numIterations; t++)
	{
		// The Forward Projection I do it in every itaration, and it,s pixel independant
		// It's to do the multiplication of SystemMatrix with Column Vector with the initial image
		// For doing this I go throw every LOR with at least one event and Calculate Siddon algorithms
		printf("Iteration No: %d\n", t);
		// Now I process the update of every pixel value thrugh the ML-EM algorithm
		cudaThreadSynchronize();
		//clock_t initialClockIteration = clock();
		/// Evento de inicio de iteración:
		cudaEventRecord(startIteration, 0);
		// For every Pixel I must do the ForwardProjection of that Pixel
 		CUDA_Forward_Projection<<<gridSizeProjector, blockSizeProjector>>>(cuda_volume, cuda_projected_michelogram, cuda_michelogram);
		
		//CUDA_Forward_Projection<<<dimGridProj, dimBlockProj>>>(cuda_volume, cuda_projected_michelogram);
		cudaThreadSynchronize();
		cudaEventRecord(stopForwardProj, 0);
		cudaEventSynchronize(stopForwardProj);
		//clock_t finalClockProjection = clock();
		
		/*CUDA_SAFE_CALL(my_cuda_error =    cudaMemcpy(RawProjectedMichelogram,cuda_projected_michelogram,sizeof(float)*NBins,cudaMemcpyDeviceToHost));
		char strNombreArchivo[512];
		sprintf(strNombreArchivo, "EstimatedProj_%d_%d_%d_Iteration_%d.dat", ProjectedMichelogram->NProj, ProjectedMichelogram->NR, ProjectedMichelogram->NZ,t);
		SaveRawFile(RawProjectedMichelogram, sizeof(float), NBins, strNombreArchivo);*/
		/// Guardo el likelihood (Siempre va una iteración atrás, ya que el likelihhod se calcula a partir de la proyección
		/// estimada, que es el primer paso del algoritmo):
		/// Lo hago en CPU porque necesito el atomicSum que solo está diponible para Compute Capabilities 2.0 o mayor.		
		CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawProjectedMichelogram, cuda_projected_michelogram,sizeof(float)*NBins,cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
		ProjectedMichelogram->FromRawData(RawProjectedMichelogram);
		this->likelihoodValues[t] = this->getLikelihoodValue();
	
		// Calculo del likelihood en GPU:
		/*this->likelihoodValues[t] = 0;
		CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_likelihood, this->likelihoodValues+t, sizeof(float), 0, cudaMemcpyHostToDevice ) );
		CUDA_Calcular_Likelihood<<<dimGridProj, dimBlockProj>>>(cuda_projected_michelogram, cuda_michelogram);
		cudaThreadSynchronize();
		#ifdef _DEBUG
		CUT_CHECK_ERROR("Kernel execution failed");
		#endif
		CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyFromSymbol(ValoresLikelihood+t, cuda_likelihood, sizeof(float), 0, cudaMemcpyDeviceToHost ) );*/

		// And then do the BackProjection of the OriginalProjection divided the
		// result of the ForwardProjection
		/// Evento de inicio de Backprojection:
		cudaEventRecord(startBackProj, 0);
		/// Lleno el backprojected volume con ceros.
		CUDA_SAFE_CALL(my_cuda_error = cudaMemset(cuda_aux_volume, 0,sizeof(float)*NPixels));
		cudaThreadSynchronize();
		CUDA_Divide_Projection<<<gridSizeProjector, blockSizeProjector>>>(cuda_michelogram, cuda_projected_michelogram);
		cudaThreadSynchronize();
		CUDA_Back_Projection<<<gridSizeBackprojector, blockSizeBackprojector>>>(cuda_projected_michelogram, cuda_aux_volume);
		cudaThreadSynchronize();
		cudaEventRecord(stopBackProj, 0);
		cudaEventSynchronize(stopBackProj);
		//clock_t finalClockBackprojection = clock();
		/*CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawAuxImage,cuda_aux_volume,sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
		sprintf(strNombreArchivo, "EstimatedImage_Iteration_%d.dat", t);
		SaveRawFile(RawAuxImage, sizeof(float), NPixels, strNombreArchivo);*/
	
		// BackprojectedImagen->SaveInFile(strNombreArchivo);
		/// Actualización del Pixel
		UpdatePixelValue<<<gridSizeImageUpdate, blockSizeImageUpdate>>>(cuda_volume,cuda_aux_volume,cuda_sum_aij);
		cudaThreadSynchronize();
		cudaEventRecord(stopIteration, 0);
		cudaEventSynchronize(stopIteration);
		/*CUDA_SAFE_CALL(my_cuda_error =cudaMemcpy(RawImage,cuda_volume,sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
		sprintf(strNombreArchivo, "Image_Iteration_%d.dat", t);
		SaveRawFile(RawImage, sizeof(float), NPixels, strNombreArchivo);*/
		// check if kernel execution generated and error
		#ifndef _DEBUG
		CUT_CHECK_ERROR("Kernel execution failed");
		#endif
				
		/// Verifico si debo guardar el resultado de la iteración.
		if(saveIterationInterval != 0)
		{
		  if((t%saveIterationInterval)==0)
		  {
		    sprintf(c_string, "%s_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
		    string outputFilename;
		    outputFilename.assign(c_string);
		    // Get back the result to CPU
		    CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawImage,cuda_volume,sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
		    //reconstructionImage->fromRawData(RawImage);
		    reconstructionImage->writeInterfile((char*)outputFilename.c_str());
		    /// Termino con el log de los resultados:
		    sprintf(c_string, "Imagen de iteración %d guardada en: %s", t, outputFilename.c_str());
		    logger->writeLine(c_string);
		  }
		}
		//cudaThreadSynchronize();
		//clock_t finalClockIteration = clock();
		/// Cargo los tiempos:
		//timesIteration_mseg[t] = (float)(finalClockIteration-initialClockIteration)/(float)CLOCKS_PER_SEC;
		cudaEventElapsedTime(&(timesIteration_mseg[t]), startIteration, stopIteration);
		//timesBackprojection_mseg[t] = (float)(finalClockBackprojection-finalClockProjection)/(float)CLOCKS_PER_SEC;
		cudaEventElapsedTime(&(timesBackprojection_mseg[t]), startBackProj, stopBackProj);
		//timesForwardprojection_mseg[t] = (float)(finalClockProjection-initialClockIteration)/(float)CLOCKS_PER_SEC;
		cudaEventElapsedTime(&(timesForwardprojection_mseg[t]), startIteration, stopForwardProj);
		//timesPixelUpdate_mseg[t] = (float)(finalClockIteration-finalClockBackprojection)/(float)CLOCKS_PER_SEC;
		cudaEventElapsedTime(&(timesPixelUpdate_mseg[t]), stopBackProj, stopIteration);
	}
	//clock_t finalClock = clock();
	/// Evento de finalización:
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float tiempoTotal;
	cudaEventElapsedTime(&tiempoTotal, start, stop);
	/// Elimino todos los eventos:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(startIteration);
	cudaEventDestroy(stopForwardProj);
	cudaEventDestroy(startBackProj);
	cudaEventDestroy(stopIteration);
	// Get back the result to CPU
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawImage,cuda_volume,sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
	reconstructionImage->fromRawData(RawImage);
	string outputFilename = outputFilenamePrefix.append("_final");
	reconstructionImage->writeInterfile((char*)outputFilename.c_str());
	
	/// Calculo la proyección de la última imagen para poder calcular el likelihood final:
	// For every Pixel I must do the ForwardProjection of that Pixel
 	CUDA_Forward_Projection<<<gridSizeProjector, blockSizeProjector>>>(cuda_volume, cuda_projected_michelogram, cuda_michelogram);
	cudaThreadSynchronize();
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawProjectedMichelogram, cuda_projected_michelogram,sizeof(float)*NBins,cudaMemcpyDeviceToHost));
	ProjectedMichelogram->FromRawData(RawProjectedMichelogram);
	this->likelihoodValues[numIterations] = this->getLikelihoodValue();
	/*this->likelihoodValues[numIterations] = 0;
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_likelihood, this->likelihoodValues+numIterations, sizeof(float), 0, cudaMemcpyHostToDevice ) );
	CUDA_Calcular_Likelihood<<<dimGridProj, dimBlockProj>>>(cuda_projected_michelogram, cuda_michelogram);
	cudaThreadSynchronize();
	#ifndef _DEBUG
		CUT_CHECK_ERROR("Kernel execution failed");
	#endif
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyFromSymbol(ValoresLikelihood+numIterations, cuda_likelihood, sizeof(float), 0, cudaMemcpyDeviceToHost ) );*/
	/// Termino con el log de los resultados:
	strcpy(c_string, "_______RESULTADOS DE RECONSTRUCCION_______");
	logger->writeLine(c_string, strlen(c_string));
	sprintf(c_string, "%f", tiempoTotal);
	logger->writeValue("Tiempo Total de Reconstrucción:", c_string);
	/// Ahora guardo los tiempos por iteración y por etapa, en fila de valores.
	strcpy(c_string, "Tiempos de Reconstrucción por Iteración [mseg]");
	logger->writeLine(c_string, strlen(c_string));
	logger->writeRowOfNumbers(timesIteration_mseg, this->numIterations);
	strcpy(c_string, "Tiempos de Forwardprojection por Iteración [mseg]");
	logger->writeLine(c_string, strlen(c_string));
	logger->writeRowOfNumbers(timesForwardprojection_mseg, this->numIterations);
	strcpy(c_string, "Tiempos de Backwardprojection por Iteración [mseg]");
	logger->writeLine(c_string, strlen(c_string));
	logger->writeRowOfNumbers(timesBackprojection_mseg, this->numIterations);
	strcpy(c_string, "Tiempos de UpdatePixel por Iteración [mseg]");
	logger->writeLine(c_string, strlen(c_string));
	logger->writeRowOfNumbers(timesPixelUpdate_mseg, this->numIterations);
	/// Por último registro los valores de likelihood:
	strcpy(c_string, "Likelihood por Iteración:");
	logger->writeLine(c_string, strlen(c_string));
	logger->writeRowOfNumbers(this->likelihoodValues, this->numIterations + 1);
	
	/// Libero la memoria de los arrays:
	free(timesIteration_mseg);
	free(timesBackprojection_mseg);
	free(timesForwardprojection_mseg);
	free(timesPixelUpdate_mseg);
	free(angleValues_radians);
	/// Libreo la memoria
	/// Estos punteros son los de los objetos Image y Michelogram , por lo que no debería liberarlos. Tal vez
	/// crear y llamar a un destructor.
	/*free(SumAij);
	free(RawImage);
	free(RawAuxImage);
	free(RawMichelogram);
	free(RawProjectedMichelogram);*/
	
	// free CUDA memory
	CUDA_SAFE_CALL(cudaFree(cuda_volume));
	CUDA_SAFE_CALL(cudaFree(cuda_aux_volume));
	CUDA_SAFE_CALL(cudaFree(cuda_michelogram));
	CUDA_SAFE_CALL(cudaFree(cuda_projected_michelogram));
	CUDA_SAFE_CALL(cudaFree(cuda_sum_aij));
	//CUDA_SAFE_CALL(cudaFree(cuda_values_phi));
	//CUDA_SAFE_CALL(cudaFree(cuda_values_r));
	//CUDA_SAFE_CALL(cudaFree(cuda_values_z));
	cudaThreadExit();
	return true;

}

bool ReconstruirSinogram3DGPU()
{
    return false;
}


__global__ void UpdatePixelValue(float *RawImage, float *FactorRawImage, float* SumAij)
{
  // Global Pixel index
  int indexPixel = blockIdx.y * ( cuda_image_size.nPixelsX * cuda_image_size.nPixelsY) + blockIdx.x * cuda_image_size.nPixelsX + threadIdx.x;
  // Memory is in contiguous address, so it can be coalesce while accessing memory.
  // Each block should have the size of a 2D Image
  if(SumAij[indexPixel] >= 1)
    RawImage[indexPixel] = RawImage[indexPixel] * FactorRawImage[indexPixel] / SumAij[indexPixel];
  else if(SumAij[indexPixel] != 0)
    RawImage[indexPixel] = RawImage[indexPixel];
  else
    RawImage[indexPixel] = 0;
  //__syncthreads();
}


// Function that shows the pattern originated from Siddon algorithm for every LOR
// For this, it makes the Projection of a Constant Image using online Siddon coefficients
/*bool CUDA_SaveSiddonPattern(SizeMichelogram MySizeMichelogram, SizeImage MySizeImage, char* pathFile)
{
	Image* MyImage = new Image(MySizeImage);
	MyImage->fillConstant(1);
	float* RawImage = MyImage->getPixelsPtr();
	unsigned int NPixels = MySizeImage.nPixelsX * MySizeImage.nPixelsY * MySizeImage.nPixelsZ;
		// I must do the forward projection for every bin. In the function ForwardProjection, the
	// forward projection is made only for the bins values that are filled with elements
	// different to 0 in the Michelogram MyMichelogram. So now we need a Michelogram that has
	// all its elements different from zero
	unsigned int NBins = MySizeMichelogram.NR*MySizeMichelogram.NProj*MySizeMichelogram.NZ*MySizeMichelogram.NZ;
	float* RawMichelogram = (float*) malloc(sizeof(float)*NBins);
	for(unsigned int i = 0; i < MySizeMichelogram.NZ * MySizeMichelogram.NZ; i++)
	{
		for(unsigned int j = 0; j < MySizeMichelogram.NProj; j++)
		{
			for(unsigned int k = 0; k < MySizeMichelogram.NR; k++)
			{
				RawMichelogram[i *(MySizeMichelogram.NR * MySizeMichelogram.NProj) + j*MySizeMichelogram.NR + k] = 1;
			}
		}
	}
	
	//////////////////////////////////// CUDA CONFIG /////////////////////////////
	////Threads organization////////////
	unsigned int NR_Splitter = 2;	// We divide NR into blocks of NR/NR_Splitter
	unsigned int threads_per_block = MySizeMichelogram.NR / NR_Splitter;
	int Resto = MySizeMichelogram.NR % NR_Splitter;
	if(threads_per_block > MAX_THREADS_PER_BLOCK)
	{
		printf("Too much r values for this implementation. \n");
		return false;
	}
	else if(Resto != 0)
	{
		// 
		printf("The threads_per_block is not a divisor of the amount of values of r.\n");
		return false;
	}
	unsigned int NumberOfBlocksX = MySizeMichelogram.NProj * NR_Splitter;
	unsigned int NumberOfBlocksY = MySizeMichelogram.NZ * MySizeMichelogram.NZ;
	unsigned int NumberOfBlocksZ = 1;
	unsigned int NumberThreadsX = threads_per_block;
	unsigned int NumberThreadsY = 1;
	dim3 dimBlockProj(NumberThreadsX,NumberThreadsY);
	dim3 dimGridProj(NumberOfBlocksX,NumberOfBlocksY,NumberOfBlocksZ);
	////////// CUDA Memory Allocation //////////////////////////////////////
	float* cuda_volume;
	float* cuda_michelogram;
	cudaError_t my_cuda_error;
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_volume, sizeof(float)*NPixels));
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_michelogram, sizeof(float)*NBins));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_volume, RawImage, sizeof(float)*NPixels,cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(cuda_michelogram, RawMichelogram, sizeof(float)*NBins,cudaMemcpyHostToDevice));
	
	Michelogram* MyMichelogram = new Michelogram(MySizeMichelogram.NProj,MySizeMichelogram.NR,MySizeMichelogram.NZ,1,1,MySizeMichelogram.RFOV,MySizeMichelogram.ZFOV);
	// Load constant memory with geometrical and other useful values
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_phi, MyMichelogram->Sinograms2D[0]->PhiValues, sizeof(float)*MyMichelogram->NProj));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_r, MyMichelogram->Sinograms2D[0]->RValues, sizeof(float)*MyMichelogram->NR));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_z, MyMichelogram->ZValues, sizeof(float)*MyMichelogram->NZ));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_threads_per_block, &NumberThreadsX, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_image_size, &MySizeImage, sizeof(SizeImage)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_michelogram_size, &MySizeMichelogram, sizeof(SizeMichelogram)));
	free(MyMichelogram);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/////////////////////
	//Siddon Test
	// I can use the same Michelogram to save the pattern
 	CUDA_Forward_Projection<<<dimGridProj, dimBlockProj>>>(cuda_volume,cuda_michelogram);
 	CUT_CHECK_ERROR("Execution failed.\n");
 	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(RawMichelogram, cuda_michelogram, sizeof(float)*NBins,cudaMemcpyDeviceToHost));
	FILE* fileMichelogram = fopen(pathFile,"wb");
	fwrite(RawMichelogram, sizeof(float), NBins , fileMichelogram);
	fclose(fileMichelogram);
	// free CUDA memory
	CUDA_SAFE_CALL(cudaFree(cuda_volume));
	CUDA_SAFE_CALL(cudaFree(cuda_michelogram));
	// free CPU memory
	free(RawMichelogram);
	free(RawImage);
	return true;
}
*/
__global__ void CUDA_Divide_Projection (float* michelogram_measured, float* michelogram_estimated)
{
  //if((threadIdx.x + (blockIdx.x % cuda_nr_splitter))<cuda_michelogram_size.NR)
  //{
  int indexSino2D = threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int indiceMichelogram = indexSino2D + blockIdx.y * cudaBinsSino2D;
  if(michelogram_estimated[indiceMichelogram]!=0)
    michelogram_estimated[indiceMichelogram] = michelogram_measured[indiceMichelogram] / michelogram_estimated[indiceMichelogram];
  else if(michelogram_measured[indiceMichelogram] != 0)
  {
    /// Si estimated = 0 e Input != 0, esto tiende a infinito. Pero no me conviene, simplemente le dejo el valor del Input. O sea considero Estimated = 1. 
    michelogram_estimated[indiceMichelogram] = michelogram_measured[indiceMichelogram];
  }
  else
  {
    /// Los bins de los sinogramas Input y Estimates son 0, o sea tengo el valor indeterminado 0/0.
    /// Lo más lógico pareciera ser dejarlo en 0 al cociente, para que no sume al backprojection.
    /// Sumarle 0 es lo mismo que nada.
  }
  //}
}

// Kernel that calculates the Forward Projection of a Image. The volume must be passed as a 
// parameter in a float*, and the result will saved in float* michelogram. Also it receives a measured
// michelogram, which is used to calculate justa those bins where thw michelogram is not zero.
__global__ void CUDA_Back_Projection (float* michelogram, float* volume)
{
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int indexSino2D = threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  /// int iR = threadIdx.x + (blockIdx.x % cuda_nr_splitter) * cuda_threads_per_block;	// ThreadX is the indexX in the Row determined by column bx on image2d bby
		
//	if((iR)<cuda_michelogram_size.NR)
//	{
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  ///int iProj = blockIdx.x / cuda_nr_splitter;	// Block index x, index of the angle prjection in the 2d sinogram
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  // Thread Index
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (unsigned int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas	
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
	  + iZ * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  // En blockIdx.y tengo
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;

  CUDA_GetPointsFromLOR(cuda_values_phi[iProj], cuda_values_r[iR], cuda_values_z[indexRing1], cuda_values_z[indexRing2], cudaRscanner, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  CUDA_Siddon (&LOR, &P1, michelogram, volume, BACKPROJECTION, indiceMichelogram);
//	}
}

// Kernel that calculates the Forward Projection of a Image. The volume must be passed as a 
// parameter in a float*, and the result will saved in float* michelogram.
__global__ void CUDA_Forward_Projection (float* volume, float* michelogram)
{
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  /// int iR = threadIdx.x + (blockIdx.x % cuda_nr_splitter) * cuda_threads_per_block;	// ThreadX is the indexX in the Row determined by column bx on image2d bby
//	if((iR)<cuda_michelogram_size.NR)
//	{
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  ///int iProj = blockIdx.x / cuda_nr_splitter;	// Block index x, index of the angle prjection in the 2d sinogram
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d sinogram
  //unsigned int Resto =  blockIdx.x % cuda_nr_splitter;
  //unsigned int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  // Thread Index
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (unsigned int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas	
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
    + blockIdx.y * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  CUDA_GetPointsFromLOR(cuda_values_phi[iProj], cuda_values_r[iR], cuda_values_z[indexRing1], cuda_values_z[indexRing2], cudaRscanner, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  CUDA_Siddon (&LOR, &P1, volume, michelogram, PROJECTION, indiceMichelogram);
//	}
}

__global__ void CUDA_Forward_Projection (float* volume, float* michelogram, float* michelogram_measured)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas	
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
	  + iZ * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  if(michelogram_measured[indiceMichelogram] != 0)
  {
    CUDA_GetPointsFromLOR(cuda_values_phi[iProj], cuda_values_r[iR], cuda_values_z[indexRing1], cuda_values_z[indexRing2], cudaRscanner, &P1, &P2);
    LOR.x = P2.x - P1.x;
    LOR.y = P2.y - P1.y;
    LOR.z = P2.z - P1.z;
    CUDA_Siddon (&LOR, &P1, volume, michelogram, PROJECTION, indiceMichelogram);
  }	
}

/// El ángulo de GetPointsFromLOR debe estar en radianes.
__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2)
{
	float auxValue = sqrtf(cudaRscanner * cudaRscanner - r * r);
	float sinValue, cosValue;
	sincosf(PhiAngle, &sinValue, &cosValue);
	P1->x = r * cosValue + sinValue * auxValue;
	P1->y = r * sinValue - cosValue * auxValue;
	P1->z = Z1;
	P2->x = r * cosValue - sinValue * auxValue;
	P2->y = r * sinValue + cosValue * auxValue;
	P2->z = Z2;
}

__device__ void CUDA_GetPointsFromLOR2 (float PhiAngle, float r, float Z1, float Z2, float Rscanner, float4* P1, float4* P2)
{
	float auxValue = sqrtf(cudaRscanner * cudaRscanner - r * r);
	float sinValue, cosValue;
	sincosf(PhiAngle, &sinValue, &cosValue);
	P1->x = r * cosValue - sinValue * auxValue;
	P1->y = r * sinValue + cosValue * auxValue;
	P1->z = Z1;
	P2->x = r * cosValue + sinValue * auxValue;
	P2->y = r * sinValue - cosValue * auxValue;
	P2->z = Z2;
}

bool MyCudaInitialization (int device)
{
	int deviceCount;
	//Check that I can multiplie the matrix
	cudaGetDeviceCount(&deviceCount);
	if(device>=deviceCount)
		return false;
	
	for(int i=0;i<deviceCount;i++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("\nCuda Number %d:",i+1);
		printf("\nNombre: %s", deviceProp.name);
		printf("\nTotal Global Memory: %d", deviceProp.totalGlobalMem);
		printf("\nShared Memory per Block: %d", deviceProp.sharedMemPerBlock);
		printf("\nRegister pero Block: %d", deviceProp.regsPerBlock);
		printf("\nWarp Size: %d", deviceProp.warpSize);
		printf("\nMax Threads Per Block: %d", deviceProp.maxThreadsPerBlock);
		printf("\nMax Threads Dimension: %dx%dx%d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("\nMax Grid Size: %dx%dx%d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("\nTotal Constant Memory: %d", deviceProp.totalConstMem);
		printf("\nMajor: %d", deviceProp.major);
		printf("\nMinor: %d", deviceProp.minor);
		printf("\nClock Rate: %d", deviceProp.clockRate);
		printf("\nTexture Alginment: %d", deviceProp.textureAlignment);
		printf("\nDevice Overlap: %d", deviceProp.deviceOverlap);
		printf("\nMultiProcessors count: %d\n", deviceProp.multiProcessorCount);
	}
		///////////////////////////////////////////////////////////
	// Initialisation of the GPU device
	///////////////////////////////////////////////////////////
	//CUT_DEVICE_INIT();
	//CHECK_CUDA_ERROR();
	if(cudaSetDevice(device)!=cudaSuccess)
		return false;
	else
		return true;
}




// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR direction vector in
// a float4*, the first point of the lor in a float4, a float* where a posible input must be loaded, 
// a float* where the result will be stored, and a int that says in which mode are we working. 
// The modes availables are: SENSIBILITY_IMAGE -> It doesn't need any input, the output is a Image
//							 PROJECTIO -> The input is a Image, and the output is a Michelogram
//							 BACKPROJECTION -> The input is a Michelogram and the output is a Image
// The size of the volume must be loaded first in the global and constant variable named cuda_image_size
// and the size of the michelogram in cuda_michelogram_size
__device__ void CUDA_Siddon (float4* LOR, float4* P0, float* Input, float* Result, int Mode, int indiceMichelogram)
{
	// se podria usar la funcion: IntersectionLinePlane(LOR, PlaneX, Point3D* IntersectionPoint);
	// para calcular los distintos puntos de interseccion, pero para hcerlo mas eficiente, lo vamos
	// a calcular como Siddon
	/// Modificación! 21/09/09: Antes de esta corrección con el circulo que se hacia de MinValues y MaxValues
	/// se consideraba un FOV cuadrado, cuando enrealdiad se desea obtener uno cricular. Por lo que dejo de
	/// de hacer intersecciïṡẄn con las rectas que delimitan el FOV cïṡẄbico. Entonces el MinValueX y MinValueY lo
	/// hago con la intersecciïṡẄn de un RFOV circular.
	/// Lo calculo como la intersecciïṡẄn entre la recta y una circunferencia de radio RFOV. La ecuaciïṡẄn a resolver es:
	/// (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=RFOV.^2
	/// alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
	//float c = LOR.P0.X*LOR.P0.X + LOR.P0.Y*LOR.P0.Y - InputImage->RFOV*InputImage->RFOV;
	float SegundoTermino = sqrt(4*(LOR->x*LOR->x*(cudaRFOV*cudaRFOV-P0->y*P0->y)
		+LOR->y*LOR->y*(cudaRFOV*cudaRFOV-P0->x*P0->x)) + 8*LOR->x*P0->x*LOR->y*P0->y);
	/// Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
	/// Como la deberïṡẄa cruzar en dos puntos hay dos soluciones.
	float alpha_xy_1 = (-2*(LOR->x*P0->x+LOR->y*P0->y) + SegundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
	float alpha_xy_2 = (-2*(LOR->x*P0->x+LOR->y*P0->y) - SegundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));

	
	/// Ahora calculo los dos puntos (X,Y)
	// float X_Circ_1 = LOR.P0.X + alpha_xy_1*LOR.Vx;
	// float Y_Circ_1 = LOR.P0.Y + alpha_xy_1*LOR.Vy;
	// float Z_Circ_1 = LOR.P0.Z + alpha_xy_1*LOR.Vz;
	float3 Circ1 = make_float3(P0->x + alpha_xy_1*LOR->x,P0->y + alpha_xy_1*LOR->y, P0->z + alpha_xy_1*LOR->z);	
	// float X_Circ_2 = LOR.P0.X + alpha_xy_2*LOR.Vx;	
	// float Y_Circ_2 = LOR.P0.Y + alpha_xy_2*LOR.Vy;
	// float Z_Circ_2 = LOR.P0.Z + alpha_xy_2*LOR.Vz;
	float3 Circ2 = make_float3(P0->x + alpha_xy_2*LOR->x,P0->y + alpha_xy_2*LOR->y, P0->z + alpha_xy_2*LOR->z);	

	// const float MinValueX = min(X_Circ_1, X_Circ_2);
	// const float MinValueY = min(Y_Circ_1, Y_Circ_2);
	// const float MinValueZ = max((float)0, min(Z_Circ_1,Z_Circ_2));
	// I use float4 instead fo float 3 to make sure the alginment is ok.
	const float3 MinValue = make_float3(min(Circ1.x, Circ2.x),min(Circ1.y, Circ2.y), max((float)OffsetZ, min(Circ1.z, Circ2.z)));
	
	// const float MaxValueX = max(X_Circ_1, X_Circ_2);
	// const float MaxValueY = max(Y_Circ_1, Y_Circ_2);
	// const float MaxValueZ = min(SCANNER_ZFOV - (SCANNER_ZFOV - zFov)/2,max(Z_Circ_1,Z_Circ_2));
	const float3 MaxValue = make_float3(max(Circ1.x, Circ2.x),max(Circ1.y, Circ2.y), min((SCANNER_ZFOV + cudaZFOV)/2, max(Circ1.z, Circ2.z)));	// Maximum coordinates values for the FOV
	
	/// Si el valor de MinValueZ es mayor que el de MaxValueZ, significa que esa lor no
	/// corta el fov de reconstrucción:
	if(MinValue.z>MaxValue.z)
	{
	  return;
	}

	// Calculates alpha values for the inferior planes (entry planes) of the FOV
	float3 alpha1 = make_float3((MinValue.x - P0->x) / LOR->x,(MinValue.y - P0->y) / LOR->y,(MinValue.z - P0->z) / LOR->z);	// LOR has the vector direction f the LOR> LOR->x = P1.x - P0->x
	// Calculates alpha values for superior planes ( going out planes) of the fov
	float3 alpha2 = make_float3((MaxValue.x - P0->x) / LOR->x,(MaxValue.y - P0->y) / LOR->y,(MaxValue.z - P0->z) / LOR->z);	// ValuesX has one more element than pixels in X, thats we can use FOVSize.nPixelsX as index for the las element

	//alpha fminf
	float3 alphas_min = make_float3(fminf(alpha1.x, alpha2.x),fminf(alpha1.y, alpha2.y),fminf(alpha1.z, alpha2.z));	// Minimum values of alpha in each direction
	//alphas_min.z = fmaxf((float)0, alphas_min.z);
	float alpha_min = fmaxf(alphas_min.x, fmaxf(alphas_min.y, alphas_min.z)); // alpha_min is the maximum values
							// bewtween the three alpha values. Because this means that we our inside the FOV
	
	//alpha fmaxf
	float3 alphas_max = make_float3(fmaxf(alpha1.x, alpha2.x), fmaxf(alpha1.y, alpha2.y), fmaxf(alpha1.z, alpha2.z));
	float alpha_max = fminf(alphas_max.x, fminf(alphas_max.y, alphas_max.z));

	// Calculus of coordinates of the first pixel (getting in pixel)
	// For x indexes de value in x increases from left to righ in Coordinate System,
	// and also in Pixel indexes. So the reference (offset) is ValueX[0].
	// On the other hand, Y and Z coordinates increase from down to up, and from bottom to top.
	// But the pixel indexes do it in the oposite way, so now the reference ( offset)
	// is ValuesY[FOVSize.nPixelsY] and ValuesZ[FOVSize.nPixelsZ] respectively.
	/// Antes la ecuaciïṡẄn se calculaba respecto del MinValueX, ahora que el MinValue es la entrada al FOV
	/// pero los ïṡẄndices de los pïṡẄxeles siguen siendo referenciados a una imagen cuadrada, por lo que utilizo
	/// los valores de RFOV que me la limitan-
	float3 indexes_min = make_float3(0,0,0);	// Starting indexes for the pixels indexes_min.x = 0, indexes_min.y = 0, indexes_min.z = 0;
	/// Verifico que estïṡẄ dentro del FOV, para eso x^2+y^2<RFOV, le doy un +1 para evitar probelmas numïṡẄricos.
	/*if((sqrt(MinValue.x*MinValue.x+MinValue.y+MinValue.y)>(cudaRFOV))||(sqrt(MaxValue.x*MaxValue.x+MaxValue.y+MaxValue.y)>(cudaRFOV))
		||(MinValue.z<0)||(MaxValue.z<0)||(MinValue.z>(cudaZFOV))||(MaxValue.z>(cudaZFOV)))
		return;	// Salgo porque no es una lor vïṡẄlida
*/
	indexes_min.x = floorf((P0->x + LOR->x * alpha_min + cudaRFOV)/cuda_image_size.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
	indexes_min.y = floorf((P0->y + LOR->y * alpha_min + cudaRFOV)/cuda_image_size.sizePixelY_mm); 
	indexes_min.z = floorf((P0->z + LOR->z * alpha_min - OffsetZ)/cuda_image_size.sizePixelZ_mm);



	// Calculus of end pixel
	float3 indexes_max = make_float3(0,0,0);
	indexes_max.x = floorf((P0->x + LOR->x * alpha_max + cudaRFOV)/cuda_image_size.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
	indexes_max.y = floorf((P0->y + LOR->y * alpha_max + cudaRFOV)/cuda_image_size.sizePixelY_mm); // 
	indexes_max.z = floorf((P0->z + LOR->z * alpha_max - OffsetZ)/cuda_image_size.sizePixelZ_mm);
	
	/// Descomentar esto!
	/// EstïṡẄ dentro del FOV? Para eso verifico que el rango de valores de i, de j y de k estïṡẄ al menos parcialmente dentro de la imagen.
	/*if(((indexes_min.x<0)&&(indexes_max.x<0))||((indexes_min.y<0)&&(indexes_max.y<0))||((indexes_min.z<0)&&(indexes_max.z<0))||((indexes_min.x>=cuda_image_size.nPixelsX)&&(indexes_max.x>=cuda_image_size.nPixelsX))
		||((indexes_min.y>=cuda_image_size.nPixelsY)&&(indexes_max.y>=cuda_image_size.nPixelsY))||((indexes_min.z>=cuda_image_size.nPixelsZ)&&(indexes_max.z>=cuda_image_size.nPixelsZ)))
	{
		return;
	}*/
	/// Incremento en píxeles en cada dirección, lo inicio en 1. Si la pendente es negativa, le cambio el signo.
	int3 incr = make_int3(1,1,1); 
	if(LOR->x < 0)
	  incr.x = -incr.x;
	if(LOR->y < 0)
	  incr.y = -incr.y;
	if(LOR->z < 0)
	  incr.z = -incr.z;

	// Amount of pixels intersected
	float Np =  fabsf(indexes_max.x - indexes_min.x) + fabsf(indexes_max.y - indexes_min.y) + fabsf(indexes_max.z - indexes_min.z) + 1; // +1 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2
	

	//Distance between thw two points of the LOR, the LOR has to be set in such way that
	// P0 is P1 of the LOR and the point represented by a=1, is P2 of the LOR
	float RayLength = sqrt(((P0->x + LOR->x) - P0->x) * ((P0->x + LOR->x) - P0->x) 
		+ ((P0->y + LOR->y) - P0->y) * ((P0->y + LOR->y) - P0->y)
		+ ((P0->z + LOR->z) - P0->z) * ((P0->z + LOR->z) - P0->z));
	//Alpha increment per each increment in one plane
	float3 alpha_u = make_float3(fabsf(cuda_image_size.sizePixelX_mm / (LOR->x)),fabsf(cuda_image_size.sizePixelY_mm / (LOR->y)),fabsf(cuda_image_size.sizePixelZ_mm / (LOR->z))); //alpha_u.x = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
	//Now we go through by every pixel crossed by the LOR
	//We get the alpha values for the startin pixel
	float3 alpha;

	//if (LOR.Vx>0)
	//	alpha_x = ( -InputImage->RFOV + (i_min + i_incr) * dx - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
	//else if (LOR.Vx<0)
	//	alpha_x = ( -InputImage->RFOV + (i_min) * dx - LOR.P0.X ) / LOR.Vx;	// Limit to the left
	//else
	//	alpha_x = numeric_limits<float>::max();
	/// Si considero el FOV circular puede tener un tamaïṡẄo lo suficientemente grande que el alpha de negativo
	/// y estïṡẄ dentro del FOV. Ya que los i_min se consideran para una imagen cuadarada. Por lo tanto, lo que me fijo
	/// que el alpha no sea menor
	if (LOR->x>0)
		alpha.x = ( -cudaRFOV + (indexes_min.x + incr.x) * cuda_image_size.sizePixelX_mm - P0->x ) / LOR->x;	//The formula is (indexes_min.x+incr.x) because que want the limit to the next change of pixel
	else if (LOR->x<0)
		alpha.x = ( -cudaRFOV + (indexes_min.x) * cuda_image_size.sizePixelX_mm - P0->x ) / LOR->x;	// Limit to the left
	else
		alpha.x = 1000000;
	if	(alpha.x <0)		// If its outside the FOV o set to a big value so it doesn't bother
		alpha.x = 1000000;

	if(LOR->y > 0)
		alpha.y = ( -cudaRFOV + (indexes_min.y + incr.y) * cuda_image_size.sizePixelY_mm - P0->y ) / LOR->y;
	else if (LOR->y < 0)
		alpha.y = ( -cudaRFOV + (indexes_min.y) * cuda_image_size.sizePixelY_mm - P0->y ) / LOR->y;
	else
		alpha.y = 1000000;
	if	(alpha.y <0)
		alpha.y = 1000000;

	if(LOR->z > 0)
		alpha.z = ( OffsetZ + (indexes_min.z + incr.z) * cuda_image_size.sizePixelZ_mm - P0->z ) / LOR->z;
	else if (LOR->z < 0)
		alpha.z = ( OffsetZ + (indexes_min.z) * cuda_image_size.sizePixelZ_mm - P0->z ) / LOR->z;
	else	// Vz = 0 -> The line is paralles to z axis, I do alpha.z the fmaxf value
		alpha.z = 1000000;
	if	(alpha.z <0)
		alpha.z = 1000000;

	float alpha_c = alpha_min;	// Auxiliar alpha value for save the latest alpha vlaue calculated
	//Initialization of first alpha value and update
	//Initialization of indexes.x,indexes.y,indexes.z values with alpha_min
	uint3 indexes = make_uint3(indexes_min.x,indexes_min.y,indexes_min.z);
	
	float4 Weight = make_float4(0,0,0,0);	// Weight for every pixel
	
	// Para calcular el indice del michelograma, necesito el índice del sino2D 
	// y el de z, que se calculan:
	// int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
	// int iZ = blockIdx.y;
	// Lo hago todo en una operación:
	/*int indiceMichelogram = threadIdx.x + (blockIdx.x * cuda_threads_per_block)
	  + blockIdx.y * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);*/
	//Result[indiceMichelogram] = 0;	// No sirve para el backprojection ni el sensibility image, hayq ue ponerlo en cero antes
	//We start going through the ray following the line directon
	for(unsigned int m = 0; m < Np; m++)
	{
	  Weight.x = indexes.x;
	  Weight.y = indexes.y;
	  Weight.z = indexes.z;
	  if((alpha.x <= alpha.y) && (alpha.x <= alpha.z))
	  {
	    // Crossing an x plane
	    Weight.w = (alpha.x - alpha_c) * RayLength;
	    indexes.x += incr.x;
	    alpha_c = alpha.x;
	    alpha.x += alpha_u.x;
	  }
	  else if((alpha.y <= alpha.x) && (alpha.y <= alpha.z))
	  {
	    // Crossing y plane
	    Weight.w = (alpha.y - alpha_c) * RayLength;
	    indexes.y += incr.y;
	    alpha_c = alpha.y;
	    alpha.y += alpha_u.y;
	  }
	  else
	  {
	    // Crossing z plane
	    Weight.w = (alpha.z - alpha_c) * RayLength;
	    indexes.z += incr.z;
	    alpha_c = alpha.z;
	    alpha.z += alpha_u.z;
	  }
	  /// Si estïṡẄ dentro de la imagen lo contabilizo. Todos los puntos
	  /// deberïṡẄan estar dentro de la imagen, pero por errores de cïṡẄlculo
	  /// algunos quedan afuera. Por lo tanto lo verifico.
	  if((Weight.x<cuda_image_size.nPixelsX)&&(Weight.y<cuda_image_size.nPixelsY)&&(Weight.z<cuda_image_size.nPixelsZ))
	  {
	    
		  switch(Mode)
		  {  
		    case SENSIBILITY_IMAGE:
			    Result[(int)(Weight.x + Weight.y * cuda_image_size.nPixelsX + Weight.z * (cuda_image_size.nPixelsX * cuda_image_size.nPixelsY))] 
				    += Weight.w;
			    break;
		    case PROJECTION:
				    Result[indiceMichelogram] += Weight.w * Input[(int)(Weight.x + Weight.y * cuda_image_size.nPixelsX + Weight.z * (cuda_image_size.nPixelsX * cuda_image_size.nPixelsY))];
			    break;
		    case BACKPROJECTION:
			    Result[(int)(Weight.x + Weight.y * cuda_image_size.nPixelsX + Weight.z * (cuda_image_size.nPixelsX * cuda_image_size.nPixelsY))] 
				    += Weight.w * Input[indiceMichelogram];
			    break;
		  }
	  }
	}
}





// Function that shows the pattern originated from Siddon algorithm for every LOR
// For this, it makes the Projection of a Constant Image using online Siddon coefficients
bool CUDA_SaveSensibilityImage(SizeMichelogram MySizeMichelogram, SizeImage MySizeImage, char* pathFile)
{
	int NPixels = MySizeImage.nPixelsX * MySizeImage.nPixelsY * MySizeImage.nPixelsZ;
	int NBins = MySizeMichelogram.NR*MySizeMichelogram.NProj*MySizeMichelogram.NZ*MySizeMichelogram.NZ;
	//////////////////////////////////// CUDA CONFIG /////////////////////////////
	////Threads organization////////////
	int NR_Splitter = 1;	// We divide NR into blocks of NR/NR_Splitter
	int threads_per_block = MySizeMichelogram.NR / NR_Splitter;
	int Resto = MySizeMichelogram.NR % NR_Splitter;
	if(threads_per_block > MAX_THREADS_PER_BLOCK)
	{
		printf("Too much r values for this implementation. \n");
		return false;
	}
	else if(Resto != 0)
	{
		// 
		printf("The threads_per_block is not a divisor of the amount of values of r.\n");
		return false;
	}
	int NumberOfBlocksX = MySizeMichelogram.NProj * NR_Splitter;
	int NumberOfBlocksY = MySizeMichelogram.NZ * MySizeMichelogram.NZ;
	int NumberOfBlocksZ = 1;
	int NumberThreadsX = threads_per_block;
	int NumberThreadsY = 1;
	dim3 dimBlockProj(NumberThreadsX,NumberThreadsY);
	dim3 dimGridProj(NumberOfBlocksX,NumberOfBlocksY,NumberOfBlocksZ);
	////////// CUDA Memory Allocation //////////////////////////////////////
	float* cuda_sensibility_volume;
	cudaError_t my_cuda_error;
	CUDA_SAFE_CALL(my_cuda_error = cudaMalloc((void**) &cuda_sensibility_volume, sizeof(float)*NPixels));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemset((void*) cuda_sensibility_volume, 0, sizeof(float)*NPixels));
	
	Michelogram* MyMichelogram = new Michelogram(MySizeMichelogram.NProj,MySizeMichelogram.NR,MySizeMichelogram.NZ,1,1,MySizeMichelogram.RFOV,MySizeMichelogram.ZFOV);
	// Load constant memory with geometrical and other useful values
	float* angleValues_radians = MyMichelogram->Sinograms2D[0]->getAnglesInRadians();
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_phi, angleValues_radians, sizeof(float)*MyMichelogram->NProj));
	float* rValues_mm = (float*)malloc(sizeof(float)*(MyMichelogram->NR));
	for(int i = 0; i < MyMichelogram->NR; i++)
	{
	  rValues_mm[i] = MyMichelogram->Sinograms2D[0]->getRValue(i);
	}
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_r, rValues_mm, sizeof(float)*MyMichelogram->NR));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_values_z, MyMichelogram->ZValues, sizeof(float)*MyMichelogram->NZ));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_threads_per_block, &NumberThreadsX, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_image_size, &MySizeImage, sizeof(SizeImage)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaRFOV, &MySizeMichelogram.RFOV, sizeof(SizeImage)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cudaZFOV, &MySizeMichelogram.ZFOV, sizeof(SizeImage)));
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpyToSymbol(cuda_michelogram_size, &MySizeMichelogram, sizeof(SizeMichelogram)));
	free(MyMichelogram);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/////////////////////
	//Siddon Test
	CUDA_Sensibility_Image<<<dimGridProj, dimBlockProj>>>(cuda_sensibility_volume);
	CUT_CHECK_ERROR("Execution failed.\n");
 	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	float* Result;
	Result = (float*)malloc(sizeof(float)*NPixels);
	CUDA_SAFE_CALL(my_cuda_error = cudaMemcpy(Result, cuda_sensibility_volume, sizeof(float)*NPixels,cudaMemcpyDeviceToHost));
	FILE* fileSensImage = fopen(pathFile,"wb");
	fwrite(Result, sizeof(float), NPixels, fileSensImage);
	fclose(fileSensImage);

	// free CUDA memory
	CUDA_SAFE_CALL(cudaFree(cuda_sensibility_volume));

	// free CPU memory
	free(Result);

	return true;
}

__global__ void CUDA_Sensibility_Image (float* volume)
{
	int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  ///int iProj = blockIdx.x / cuda_nr_splitter;	// Block index x, index of the angle prjection in the 2d sinogram
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (unsigned int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas	
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
    + iZ * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  CUDA_GetPointsFromLOR(cuda_values_phi[iProj], cuda_values_r[iR], cuda_values_z[indexRing1], cuda_values_z[indexRing2], cudaRscanner, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  CUDA_Siddon (&LOR, &P1, NULL, volume, SENSIBILITY_IMAGE, indiceMichelogram);
}

__global__ void CUDA_Calcular_Likelihood (float* estimated_michelogram, float* measured_michelogram)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (unsigned int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas		
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
    + iZ * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  if(estimated_michelogram[indiceMichelogram]>0)
  {
    cuda_likelihood += measured_michelogram[indiceMichelogram] * logf(estimated_michelogram[indiceMichelogram])
      - estimated_michelogram[indiceMichelogram];
  }
}
