/**
	\file CuSiddon.cu
	\brief Implementación de funciónes de device del ray tracing con siddon.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.07.11
	\version 1.1.0
*/
#ifndef _CUSIDDONFUNC_H_
#define _CUSIDDONFUNC_H_

#include <CuSiddon.h>
#include <float.h>

// Variables de Memoria constante utilizadas en Siddon. Se debe encargar de cargar los datos de forma rpevia a la reconstrucción.
//__device__ __constant__ float dummy; // Esto lo tuve que agregar porque el cudaMemcpyToSymbol me tiraba error con la primera variable declarada acá, sea cual sea.

extern __device__ __constant__ float d_AxialFov_mm;

extern __device__ __constant__ float d_RadioFov_mm;

extern __device__ __constant__ SizeImage d_imageSize;

extern __device__ __constant__ int d_numPixels;

extern __device__ __constant__ int numPixelsPerSlice;

extern __device__ __constant__ int d_numBinsSino2d;

extern texture<float, 3, cudaReadModeElementType> texImage;
// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR direction vector in
// a float4*, the first point of the lor in a float4, a float* where a posible input must be loaded, 
// a float* where the result will be stored, and a int that says in which mode are we working. 
// The modes availables are: SENSIBILITY_IMAGE -> It doesn't need any input, the output is a Image
//							 PROJECTIO -> The input is a Image, and the output is a Michelogram
//							 BACKPROJECTION -> The input is a Michelogram and the output is a Image
// The size of the volume must be loaded first in the global and constant variable named d_imageSize
// and the size of the michelogram in cuda_michelogram_size
__device__ void CuSiddon (float4* LOR, float4* P0, float* Input, float* Result, int Mode, int indiceMichelogram)

{

  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_x_1, alpha_x_2, alpha_y_1, alpha_y_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_min, alpha_y_min, alpha_x_max, alpha_y_max; //, alpha_z_min, alpha_z_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.

  // Valores de alpha para recorrer la lor:
  float alpha_x = FLT_MAX, alpha_y = FLT_MAX, alpha_z = FLT_MAX;	// Valores de alpha si avanzo en x o en y, siempre debo ir siguiendo al más cercano.
  float alpha_x_u, alpha_y_u, alpha_z_u;	// Valor de incremento de alpha según avance un píxel en x o en y.
  float alpha_c;	// Valor de alhpa actual, cuando se recorre la lor.

  // Variables relacionadas con los índices de píxeles recorridos:
  int i_min = 0, j_min = 0, k_min = 0;	// Índices (i,j,k) del píxel de entrada al fov.
  int i_max = 0, j_max = 0, k_max = 0;	// Índices (i,j,k) del píxel de salida al fov.
  int i, j, k;	// Índices con que recorro los píxeles de la lor.

  // Incrementos en píxeles. Puede ser +-1 según la dirección de la lor.
  int i_incr = 0, j_incr = 0, k_incr = 0;

  // Cantidad de píxeles intersectados:
  int numIntersectedPixels;

  // Punto de entrada y salida al fov trasladado al borde del píxel, ya que en esta versión el píxel
  // de entrada se considera entero, y no desde el punto exacto de itnersección con el círculo:
  float x_1_mm, x_2_mm, y_1_mm, y_2_mm, z_1_mm, z_2_mm;

  // Largo de la lor teniendo en cuenta P0 y P1, y largo de la lor dentro del fov:
  float rayLengthInFov_mm,rayLength_mm;

  
// For Fov cilindrico:
//   // Cálculo de intersección de la lor con un fov cilíndrico.
//   // Las lors siempre interesectan las caras curvas del cilindro y no las tapas. Ya
//   // que el fov de los scanner está limitado por eso.
//   // Lo calculo como la intersección entre la recta y una circunferencia de radio cudaRFOV. La ecuación a resolver es:
//   // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=cudaRFOV.^2
//   // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
//   //float c = P0->x*P0->x + P0->y*P0->y - cudaRFOV*cudaRFOV;
//   float segundoTermino = sqrt(4.0f*(LOR->x*LOR->x*(d_RadioFov_mm*d_RadioFov_mm-P0->y*P0->y)
//     +LOR->y*LOR->y*(d_RadioFov_mm*d_RadioFov_mm-P0->x*P0->x)) + 8.0f*LOR->x*P0->x*LOR->y*P0->y);
// 
//   // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
//   // Como la debería cruzar en dos puntos hay dos soluciones.
//   alpha_xy_1 = (-2*(LOR->x*P0->x+LOR->y*P0->y) + segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
//   alpha_xy_2 = (-2*(LOR->x*P0->x+LOR->y*P0->y) - segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
// 
//   // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
// // se recorre desde P0 a P1.
//   alpha_min = min(alpha_xy_1, alpha_xy_2);
//   alpha_max = max(alpha_xy_1, alpha_xy_2);

    // Para FOV cuadrado:
  // Obtengo la intersección de la lor con las rectas x=-rFov_mm x=rFov_mm y=-rFov_mm y =rFov_mm
  // Para dichos valores verifico que la otra coordenada este dentro de los valores, y obtengo
  // los puntos de entrada y salida de la lor. No me fijo z, porque no debería ingresar por las
  // tapas del cilindro, al menos que haya algún error entre el sinograma y la imagen de entrada.
  float minValueX_mm = -d_RadioFov_mm;
  float minValueY_mm = -d_RadioFov_mm;
  float maxValueX_mm = d_RadioFov_mm;
  float maxValueY_mm = d_RadioFov_mm;
  
  
  // Calculates alpha values for the inferior planes (entry planes) of the FOV
  if(LOR->x == 0) // Parallel to x axis
  {
    alpha_y_1 = (minValueY_mm - P0->y) / LOR->y; 
    alpha_y_2 = (maxValueY_mm - P0->y) / LOR->y;
    if(alpha_y_1 < alpha_y_2)
    {
      alpha_min = alpha_y_1;
      alpha_max = alpha_y_2;
    }
    else
    {
      alpha_min = alpha_y_2;
      alpha_max = alpha_y_1;
    }
  }
  else if(LOR->y == 0) // Parallel to y axis.
  {
    alpha_x_1 = (minValueX_mm - P0->x) / LOR->x;
    alpha_x_2 = (maxValueX_mm - P0->x) / LOR->x;
    if(alpha_x_1 < alpha_x_2)
    {
      alpha_min = alpha_x_1;
      alpha_max = alpha_x_2;
    }
    else
    {
      alpha_min = alpha_x_2;
      alpha_max = alpha_x_1;
    }
  }
  else
  {
    alpha_x_1 = (minValueX_mm - P0->x) / LOR->x;
    alpha_y_1 = (minValueY_mm - P0->y) / LOR->y;  
    // Calculates alpha values for superior planes ( going out planes) of the fov
    alpha_x_2 = (maxValueX_mm - P0->x) / LOR->x;	// ValuesX has one more element than pixels in X, thats we can use InputVolume->SizeX as index for the las element
    alpha_y_2 = (maxValueY_mm - P0->y) / LOR->y;
    //alpha min
    alpha_x_min = min(alpha_x_1, alpha_x_2);
    alpha_y_min = min(alpha_y_1, alpha_y_2);
    //alpha_y_min = max((float)0, alpha_y_min);
    alpha_min = max(alpha_x_min, alpha_y_min); //
    //alpha max
    alpha_x_max = max(alpha_x_1, alpha_x_2);
    alpha_y_max = max(alpha_y_1, alpha_y_2);
    alpha_max = min(alpha_x_max, alpha_y_max);
  }
    
  
  // if the radius of the scanner is less than the diagonal (alpha less than 0), the entry point should be P0
  if ((alpha_min<0)||(alpha_min>1)) // I added (alpha_min>1), because for aprallel lors to an axis, both alphas can be positiver or negative.
    alpha_min = 0;
  // if the radius of the scanner is less than the diagonal (alpha less than 0), the entry point should be P0
  if ((alpha_max>1)||(alpha_max<0)) 
    alpha_max = 1;
  // Fin para Fov Cuadrado.
  
  // Coordenadas dentro de la imagen de los dos puntos de entrada:
  x_1_mm = P0->x + LOR->x * alpha_min;
  y_1_mm = P0->y + LOR->y * alpha_min;
  z_1_mm = P0->z + LOR->z * alpha_min;
  
  x_2_mm = P0->x + LOR->x * alpha_max;
  y_2_mm = P0->y + LOR->y * alpha_max;
  z_2_mm = P0->z + LOR->z * alpha_max;
  
  // To debug the axial coordinates:
  #ifdef __DEBUG__
    if ((blockIdx.x == 0) && (threadIdx.x == 0))
      printf("i_z:%d z1:%f z2:%f\n", threadIdx.x, z_1_mm, z_2_mm);
  #endif
  
  rayLengthInFov_mm = sqrt((x_2_mm-x_1_mm) * (x_2_mm-x_1_mm) + (y_2_mm-y_1_mm) * (y_2_mm-y_1_mm) + (z_2_mm-z_1_mm) * (z_2_mm-z_1_mm));

  // Distancia total de la LOR. Es la distancia entre los puntos P0 y P1, habitualmente, esos son
  // los puntos de la lor sobre el detector.
  rayLength_mm = sqrt(((P0->x + LOR->x) - P0->x) * ((P0->x + LOR->x) - P0->x) 
	  + ((P0->y + LOR->y) - P0->y) * ((P0->y + LOR->y) - P0->y)
	  + ((P0->z + LOR->z) - P0->z) * ((P0->z + LOR->z) - P0->z));
  
  float offsetZ_mm = d_imageSize.sizePixelZ_mm/2;//(SCANNER_ZFOV - cudaZFOV)/2;
  #ifdef __DEBUG__
    if((z_1_mm < offsetZ_mm)||(z_1_mm > (d_AxialFov_mm-offsetZ_mm)) || (z_2_mm < offsetZ_mm)||(z_2_mm > (d_AxialFov_mm-offsetZ_mm)))
    {
      // La lor entra por las tapas del clindro del fov:
      printf("Warning: Lor que entra por las tapas del cilindro del FoV.\n");
    }
  #endif

  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  i_min = floorf((x_1_mm + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = floorf((y_1_mm + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); 
  k_min = floorf((z_1_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm); 
  i_max = floorf((x_2_mm + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = floorf((y_2_mm + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); // 
  k_max = floorf((z_2_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm);
  int nPixelsXY = (d_imageSize.nPixelsX * d_imageSize.nPixelsY);
  int indicePixel;

  // Esta verificación y corrección la saco, porque a veces por error de redondeo puede quedar en el píxel -1 o en sizePixel
  #ifdef __DEBUG__
    // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
    if(((i_min<0)||(i_max<0))||((j_min<0)||(j_max<0))||((k_min<0)||(k_max<0))||((i_min>=d_imageSize.nPixelsX)||(i_max>=d_imageSize.nPixelsX))||
      ((j_min>=d_imageSize.nPixelsY)||(j_max>=d_imageSize.nPixelsY))||((k_min>=d_imageSize.nPixelsZ)||(k_max>=d_imageSize.nPixelsZ)))
    {
      // Por error de redondeo puede caer al límite:
      printf("Indices fuera de imagen. Pixel min: (%d,%d,%d) (%f,%f,%f)mm. Pixel max: (%d,%d,%d) (%f,%f,%f)mm.\n", i_min, j_min, k_min, x_1_mm, y_1_mm, z_1_mm, i_max, j_max, k_max, x_2_mm, y_2_mm, z_2_mm);
      return;
    }
  #endif


  // Cantidad de píxeles intersectados:
  numIntersectedPixels = abs(i_max - i_min) + abs(j_max - j_min) + abs(k_max - k_min) + 2; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2
  
  // Pixels increments
  i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR->x > 0)
    i_incr = 1;
  else if(LOR->x < 0)
    i_incr = -1;
  if(LOR->y > 0)
    j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR->y < 0)
    j_incr = -1;
  if(LOR->z > 0)
    k_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR->z < 0)
    k_incr = -1;
  
  // Incremento en los valores de alpha, según se avanza un píxel en x o en y.
  alpha_x_u = fabsf(d_imageSize.sizePixelX_mm / (LOR->x)); //alpha_x_u = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
  alpha_y_u = fabsf(d_imageSize.sizePixelY_mm / (LOR->y));
  alpha_z_u = fabsf(d_imageSize.sizePixelZ_mm / (LOR->z));

    // A partir del (i_min,j_min) voy recorriendo la lor, para determinar el índice del próximo píxel, o sea
  // saber si avanzo en i o en j, debo ver si el cambio se da en x o en y. Para esto en cada avance se calcula
  // el valor de alpha si avanzar un píxel en i (o x) y el valor de alpha en j (o y). De estos dos valores: alpha_x
  // y alpha_y, el que sea menor indicará en que sentido tengo que avanzar con el píxel.
  if (LOR->x>0)
    alpha_x = ( -d_RadioFov_mm + (i_min + i_incr) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR->x<0)
    alpha_x = ( -d_RadioFov_mm + i_min * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	// Limit to the left
  else
    alpha_x = FLT_MAX;
  if (alpha_x <0)		// If its outside the FOV que get to the maximum value so it doesn't bother
    alpha_x = FLT_MAX;
  
  if(LOR->y > 0)
    alpha_y = ( -d_RadioFov_mm + (j_min + j_incr) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
  else if (LOR->y < 0)
    alpha_y = ( -d_RadioFov_mm + j_min * d_imageSize.sizePixelY_mm - P0->y) / LOR->y;
  else
    alpha_y = FLT_MAX;
  if (alpha_y <0)
    alpha_y = FLT_MAX;
  
  if(LOR->z > 0)
    alpha_z = ( offsetZ_mm + (k_min + k_incr) * d_imageSize.sizePixelZ_mm - P0->z) / LOR->z;
  else if (LOR->z < 0)
    alpha_z = ( offsetZ_mm + k_min * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
  else
    alpha_z = FLT_MAX;
  if (alpha_z <0)
    alpha_z = FLT_MAX;
  
  // En alpha_c voy guardando el valor de alpha con el que voy recorriendo los píxeles.
  alpha_c = alpha_min;
//   // En alpha_x guardo los próximos límites en todas las coordenadas. En alpha_c está el punto actual, que es el mínimo.
//   alpha_x = alpha_x_min + alpha_x_u;
//   alpha_y = alpha_y_min + alpha_y_u;
//   alpha_z = alpha_z_min + alpha_z_u;

  // Inicialización de i,j a sus valores de entrada.
  i = i_min;
  j = j_min;
  k = k_min;

  // Recorro la lor y guardo los segmentos en la lista de salida.
  float siddonWeight = 0;
  for(int m = 0; m < numIntersectedPixels; m++)
  {
    indicePixel = i + j * d_imageSize.nPixelsX + k * nPixelsXY;
    if((alpha_x <= alpha_y) && (alpha_x <= alpha_z))
    {
      // Cruce por el plano x: avanzo en i.
      siddonWeight = (alpha_x - alpha_c) * rayLength_mm;
      i += i_incr;
      alpha_c = alpha_x;
      alpha_x += alpha_x_u;
    }
    else if((alpha_y <= alpha_x) && (alpha_y <= alpha_z))
    {
      siddonWeight = (alpha_y - alpha_c) * rayLength_mm;
      // Cruce por el plano y: avanzo en j.
      j += j_incr;
      alpha_c = alpha_y;
      alpha_y += alpha_y_u;
    }
    else
    {
      // Cruce por el plano y: avanzo en j.
      siddonWeight = (alpha_z - alpha_c) * rayLength_mm;
      k += k_incr;
      alpha_c = alpha_z;
      alpha_z += alpha_z_u;
    }
    //if((weight.x<d_imageSize.nPixelsX)&&(weight.y<d_imageSize.nPixelsY)&&(weight.z<d_imageSize.nPixelsZ)&&(weight.x>=0)&&(weight.y>=0)&&(weight.z>=0)&&(weight.w>=0))
    //{ 
      switch(Mode)
      {  
	case SENSIBILITY_IMAGE:
	  atomicAdd(Result+indicePixel, siddonWeight * d_imageSize.sizePixelX_mm);
	  break;
	case PROJECTION:
	  //Result[indiceMichelogram] += siddonWeight * Input[indicePixel];
	  Result[indiceMichelogram] += siddonWeight * tex3D(texImage,i,j,k);
	  break;
	case BACKPROJECTION:
	  atomicAdd(Result+indicePixel, siddonWeight * Input[indiceMichelogram]);
	  break;
      }
    //}

  }
}

__device__ void CuSiddonBackprojection (float4* LOR, float4* P0, float* image, float* sinogram, int indiceMichelogram)

{

  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_x_1, alpha_x_2, alpha_y_1, alpha_y_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_min, alpha_y_min, alpha_x_max, alpha_y_max;//, alpha_z_min, alpha_z_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.

  // Valores de alpha para recorrer la lor:
  float alpha_x = 10000000.0f, alpha_y = 10000000.0f, alpha_z = 10000000.0f;	// Valores de alpha si avanzo en x o en y, siempre debo ir siguiendo al más cercano.
  float alpha_x_u, alpha_y_u, alpha_z_u;	// Valor de incremento de alpha según avance un píxel en x o en y.
  float alpha_c;	// Valor de alhpa actual, cuando se recorre la lor.

  // Variables relacionadas con los índices de píxeles recorridos:
  int i_min = 0, j_min = 0, k_min = 0;	// Índices (i,j,k) del píxel de entrada al fov.
  int i_max = 0, j_max = 0, k_max = 0;	// Índices (i,j,k) del píxel de salida al fov.
  int i, j, k;	// Índices con que recorro los píxeles de la lor.

  // Incrementos en píxeles. Puede ser +-1 según la dirección de la lor.
  int i_incr = 0, j_incr = 0, k_incr = 0;

  

  // Cantidad de píxeles intersectados:
  int numIntersectedPixels;

  // Punto de entrada y salida al fov trasladado al borde del píxel, ya que en esta versión el píxel
  // de entrada se considera entero, y no desde el punto exacto de itnersección con el círculo:
  float x_1_mm, x_2_mm, y_1_mm, y_2_mm, z_1_mm, z_2_mm;

  // Largo de la lor teniendo en cuenta P0 y P1, y largo de la lor dentro del fov:
  float rayLength_mm;

  
// For Fov cilindrico:
//   // Cálculo de intersección de la lor con un fov cilíndrico.
//   // Las lors siempre interesectan las caras curvas del cilindro y no las tapas. Ya
//   // que el fov de los scanner está limitado por eso.
//   // Lo calculo como la intersección entre la recta y una circunferencia de radio cudaRFOV. La ecuación a resolver es:
//   // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=cudaRFOV.^2
//   // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
//   //float c = P0->x*P0->x + P0->y*P0->y - cudaRFOV*cudaRFOV;
//   float segundoTermino = sqrt(4.0f*(LOR->x*LOR->x*(d_RadioFov_mm*d_RadioFov_mm-P0->y*P0->y)
//     +LOR->y*LOR->y*(d_RadioFov_mm*d_RadioFov_mm-P0->x*P0->x)) + 8.0f*LOR->x*P0->x*LOR->y*P0->y);
// 
//   // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
//   // Como la debería cruzar en dos puntos hay dos soluciones.
//   alpha_xy_1 = (-2*(LOR->x*P0->x+LOR->y*P0->y) + segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
//   alpha_xy_2 = (-2*(LOR->x*P0->x+LOR->y*P0->y) - segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
// 
//   // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
// // se recorre desde P0 a P1.
//   alpha_min = min(alpha_xy_1, alpha_xy_2);
//   alpha_max = max(alpha_xy_1, alpha_xy_2);

    // Para FOV cuadrado:
  // Obtengo la intersección de la lor con las rectas x=-rFov_mm x=rFov_mm y=-rFov_mm y =rFov_mm
  // Para dichos valores verifico que la otra coordenada este dentro de los valores, y obtengo
  // los puntos de entrada y salida de la lor. No me fijo z, porque no debería ingresar por las
  // tapas del cilindro, al menos que haya algún error entre el sinograma y la imagen de entrada.
  float minValueX_mm = -d_RadioFov_mm;
  float minValueY_mm = -d_RadioFov_mm;
  float maxValueX_mm = d_RadioFov_mm;
  float maxValueY_mm = d_RadioFov_mm;
  
  
  // Calculates alpha values for the inferior planes (entry planes) of the FOV
  if(LOR->x == 0) // Parallel to x axis
  {
    alpha_y_1 = (minValueY_mm - P0->y) / LOR->y; 
    alpha_y_2 = (maxValueY_mm - P0->y) / LOR->y;
    if(alpha_y_1 < alpha_y_2)
    {
      alpha_min = alpha_y_1;
      alpha_max = alpha_y_2;
    }
    else
    {
      alpha_min = alpha_y_2;
      alpha_max = alpha_y_1;
    }
  }
  else if(LOR->y == 0) // Parallel to y axis.
  {
    alpha_x_1 = (minValueX_mm - P0->x) / LOR->x;
    alpha_x_2 = (maxValueX_mm - P0->x) / LOR->x;
    if(alpha_x_1 < alpha_x_2)
    {
      alpha_min = alpha_x_1;
      alpha_max = alpha_x_2;
    }
    else
    {
      alpha_min = alpha_x_2;
      alpha_max = alpha_x_1;
    }
  }
  else
  {
    alpha_x_1 = (minValueX_mm - P0->x) / LOR->x;
    alpha_y_1 = (minValueY_mm - P0->y) / LOR->y;  
    // Calculates alpha values for superior planes ( going out planes) of the fov
    alpha_x_2 = (maxValueX_mm - P0->x) / LOR->x;	// ValuesX has one more element than pixels in X, thats we can use InputVolume->SizeX as index for the las element
    alpha_y_2 = (maxValueY_mm - P0->y) / LOR->y;
    //alpha min
    alpha_x_min = min(alpha_x_1, alpha_x_2);
    alpha_y_min = min(alpha_y_1, alpha_y_2);
    //alpha_y_min = max((float)0, alpha_y_min);
    alpha_min = max(alpha_x_min, alpha_y_min); //
    //alpha max
    alpha_x_max = max(alpha_x_1, alpha_x_2);
    alpha_y_max = max(alpha_y_1, alpha_y_2);
    alpha_max = min(alpha_x_max, alpha_y_max);
  }
    
  
  // if the radius of the scanner is less than the diagonal (alpha less than 0), the entry point should be P0
  if ((alpha_min<0)||(alpha_min>1)) // I added (alpha_min>1), because for aprallel lors to an axis, both alphas can be positiver or negative.
    alpha_min = 0;
  // if the radius of the scanner is less than the diagonal (alpha less than 0), the entry point should be P0
  if ((alpha_max>1)||(alpha_max<0)) 
    alpha_max = 1;
  // Fin para Fov Cuadrado.
  
  // Coordenadas dentro de la imagen de los dos puntos de entrada:
  x_1_mm = P0->x + LOR->x * alpha_min;
  y_1_mm = P0->y + LOR->y * alpha_min;
  z_1_mm = P0->z + LOR->z * alpha_min;
  
  x_2_mm = P0->x + LOR->x * alpha_max;
  y_2_mm = P0->y + LOR->y * alpha_max;
  z_2_mm = P0->z + LOR->z * alpha_max;
  
  // Distancia total de la LOR. Es la distancia entre los puntos P0 y P1, habitualmente, esos son
  // los puntos de la lor sobre el detector.
  rayLength_mm = sqrt(((P0->x + LOR->x) - P0->x) * ((P0->x + LOR->x) - P0->x) 
	  + ((P0->y + LOR->y) - P0->y) * ((P0->y + LOR->y) - P0->y)
	  + ((P0->z + LOR->z) - P0->z) * ((P0->z + LOR->z) - P0->z));
  
  float offsetZ_mm = d_imageSize.sizePixelZ_mm/2;//(SCANNER_ZFOV - cudaZFOV)/2;
  #ifdef __DEBUG__
    if((z_1_mm < offsetZ_mm)||(z_1_mm > (d_AxialFov_mm-offsetZ_mm)) || (z_2_mm < offsetZ_mm)||(z_2_mm > (d_AxialFov_mm-offsetZ_mm)))
    {
      // La lor entra por las tapas del clindro del fov:
      printf("Warning: Lor que entra por las tapas del cilindro del FoV.\n");
    }
  #endif

  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  i_min = floorf((x_1_mm + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = floorf((y_1_mm + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); 
  k_min = floorf((z_1_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm); 
  i_max = floorf((x_2_mm + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = floorf((y_2_mm + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); // 
  k_max = floorf((z_2_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm);
  int nPixelsXY = (d_imageSize.nPixelsX * d_imageSize.nPixelsY);
  int indicePixel;


  // Esta verificación y corrección la saco, porque a veces por error de redondeo puede quedar en el píxel -1 o en sizePixel
  #ifdef __DEBUG__
    // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
    if(((i_min<0)||(i_max<0))||((j_min<0)||(j_max<0))||((k_min<0)||(k_max<0))||((i_min>=d_imageSize.nPixelsX)||(i_max>=d_imageSize.nPixelsX))||
      ((j_min>=d_imageSize.nPixelsY)||(j_max>=d_imageSize.nPixelsY))||((k_min>=d_imageSize.nPixelsZ)||(k_max>=d_imageSize.nPixelsZ)))
    {
      // Por error de redondeo puede caer al límite:
      printf("Indices fuera de imagen. Pixel min: (%d,%d,%d) (%f,%f,%f)mm. Pixel max: (%d,%d,%d) (%f,%f,%f)mm.\n", i_min, j_min, k_min, x_1_mm, y_1_mm, z_1_mm, i_max, j_max, k_max, x_2_mm, y_2_mm, z_2_mm);
      return;
    }
  #endif


  // Cantidad de píxeles intersectados:
  numIntersectedPixels = abs(i_max - i_min) + abs(j_max - j_min) + abs(k_max - k_min) + 2; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2
  
  // Pixels increments
  // A partir del (i_min,j_min) voy recorriendo la lor, para determinar el índice del próximo píxel, o sea
  // saber si avanzo en i o en j, debo ver si el cambio se da en x o en y. Para esto en cada avance se calcula
  // el valor de alpha si avanzar un píxel en i (o x) y el valor de alpha en j (o y). De estos dos valores: alpha_x
  // y alpha_y, el que sea menor indicará en que sentido tengo que avanzar con el píxel.
  i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR->x > 0)
  {
    i_incr = 1;
    alpha_x = ( -d_RadioFov_mm + (i_min + i_incr) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;
  }
  else if(LOR->x < 0)
  {
    i_incr = -1;
    alpha_x = ( -d_RadioFov_mm + i_min * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;
  }
  /*else
    alpha_x = FLT_MAX;*/ // I can avoid this because I initialize with this value.
  if(LOR->y > 0)
  {
    j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
    alpha_y = ( -d_RadioFov_mm + (j_min + j_incr) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
  }
  else if(LOR->y < 0)
  {
    j_incr = -1;
    alpha_y = ( -d_RadioFov_mm + j_min * d_imageSize.sizePixelY_mm - P0->y) / LOR->y;
  }
  /*if (alpha_y <0)
    alpha_y = FLT_MAX;
  */
  if(LOR->z > 0)
  {
    k_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
    alpha_z = ( offsetZ_mm + (k_min + k_incr) * d_imageSize.sizePixelZ_mm - P0->z) / LOR->z;
  }
  else if(LOR->z < 0)
  {
    k_incr = -1;
    alpha_z = ( offsetZ_mm + k_min * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
  }
  /*if (alpha_z <0)
    alpha_z = FLT_MAX;
  */
  
  // Incremento en los valores de alpha, según se avanza un píxel en x o en y.
  alpha_x_u = fabsf(d_imageSize.sizePixelX_mm / (LOR->x)); //alpha_x_u = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
  alpha_y_u = fabsf(d_imageSize.sizePixelY_mm / (LOR->y));
  alpha_z_u = fabsf(d_imageSize.sizePixelZ_mm / (LOR->z));
  
  // En alpha_c voy guardando el valor de alpha con el que voy recorriendo los píxeles.
  alpha_c = alpha_min;
//   // En alpha_x guardo los próximos límites en todas las coordenadas. En alpha_c está el punto actual, que es el mínimo.
//   alpha_x = alpha_x_min + alpha_x_u;
//   alpha_y = alpha_y_min + alpha_y_u;
//   alpha_z = alpha_z_min + alpha_z_u;

  // Inicialización de i,j a sus valores de entrada.
  i = i_min;
  j = j_min;
  k = k_min;

  // Recorro la lor y guardo los segmentos en la lista de salida.
  float siddonWeight = 0;
  float binValue = sinogram[indiceMichelogram]; // Copy bin value to a register to accelerate the access.
  int nextIndexPixel; // To try to avoid re computing the pixel index.
  indicePixel = i + j * d_imageSize.nPixelsX + k * nPixelsXY; // starting index pixel
  for(int m = 0; m < numIntersectedPixels; m++)
  {
    if((alpha_x <= alpha_y) && (alpha_x <= alpha_z))
    {
      // Cruce por el plano x: avanzo en i.
      siddonWeight = (alpha_x - alpha_c) * rayLength_mm;
      i += i_incr;
      nextIndexPixel = indicePixel + i_incr;
      alpha_c = alpha_x;
      alpha_x += alpha_x_u;
    }
    else if((alpha_y <= alpha_x) && (alpha_y <= alpha_z))
    {
      siddonWeight = (alpha_y - alpha_c) * rayLength_mm;
      // Cruce por el plano y: avanzo en j.
      j += j_incr;
      nextIndexPixel = indicePixel + j_incr * d_imageSize.nPixelsX;
      alpha_c = alpha_y;
      alpha_y += alpha_y_u;
    }
    else
    {
      // Cruce por el plano y: avanzo en j.
      siddonWeight = (alpha_z - alpha_c) * rayLength_mm;
      k += k_incr;
      nextIndexPixel = indicePixel + k_incr * nPixelsXY;
      alpha_c = alpha_z;
      alpha_z += alpha_z_u;
    }
    if((indicePixel<d_numPixels)&&(indicePixel>=0))
      atomicAdd(image+indicePixel, siddonWeight * binValue);
    indicePixel = nextIndexPixel;
  }
}

#endif

