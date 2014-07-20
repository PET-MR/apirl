/** @file helper_cuda.h */
#ifndef _HELPER_CUDA_H_
#define _HELPER_CUDA_H_

/** @brief Tama#o del warp de la GPU */
#define CUDA_WARP_SIZE 32
/** @brief Mascara de bits del tama#o del warp de la GPU */
#define CUDA_WARP_MASK ((CUDA_WARP_SIZE) - 1)

/** @brief Tama#o de medio warp de la GPU */
#define CUDA_HALF_WARP_SIZE ((CUDA_WARP_SIZE) / 2)
/** @brief Mascara de bits del tama#o de medio warp de la GPU */
#define CUDA_HALF_WARP_MASK ((CUDA_HALF_WARP_SIZE) - 1)

#ifdef __CUDACC__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/**
 * Helper que imprime un mensaje de error informativo si hubo un problema en una llamada a CUDA.
 *
 * @param result valor de retorno de la funcion de CUDA
 * @param func   string con el nombre de la funcion llamada
 * @param file   archivo en el que se dio el error
 * @param line   linea en la que se dio el error
 */
static
inline
void
_checkCudaReturnValue(cudaError_t result, const char *const func, const char *const file, const int line)
{
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
          file, line, static_cast<int>(result), cudaGetErrorString(result), func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(static_cast<int>(result));
  }
}

/** @brief Macro para verificar valores de retorno de las llamadas a la API de CUDA */
#define checkCudaErrors(val) _checkCudaReturnValue ( (val), #val, __FILE__, __LINE__ )


/**
 * Helper que imprime un mensaje de error informativo si hay un error pendiente de CUDA.
 *
 * @param errorMessage mensaje informativo definido por el usuario
 * @param file   archivo en el que se detecto el error
 * @param line   linea en la que se detecto el error
 */
static
inline
void
_getLastCudaError(const char *errorMessage, const char *file, const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
          file, line, errorMessage, static_cast<int>(err), cudaGetErrorString(err));
    cudaDeviceReset();
    exit(static_cast<int>(err));
  }
}

/** @brief Macro para verificar si hay un error pendiente de CUDA */
#define getLastCudaError(msg) _getLastCudaError (msg, __FILE__, __LINE__)


/** Definiciones para ocultar palabras reservadas de CUDA que dan errores en el compilador del host */
#define HOSTDEVICEFUNCTION __host__ __device__
#define KERNELFUNCTION __global__
#define HOSTFUNCTION __host__
#define DEVICEFUNCTION __device__

#else /* !__CUDACC__ */

#define HOSTDEVICEFUNCTION
#define KERNELFUNCTION
#define HOSTFUNCTION
#define DEVICEFUNCTION

#endif /* __CUDACC__ */

#endif /* _HELPER_CUDA_H_ */
