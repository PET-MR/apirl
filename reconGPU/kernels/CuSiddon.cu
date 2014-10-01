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


// Variables de Memoria constante utilizadas en Siddon. Se debe encargar de cargar los datos de forma rpevia a la reconstrucción.
__device__ __constant__ float dummy; // Esto lo tuve que agregar porque el cudaMemcpyToSymbol me tiraba error con la primera variable declarada acá, sea cual sea.

extern __device__ __constant__ float d_AxialFov_mm;

extern __device__ __constant__ float d_RadioFov_mm;

extern __device__ __constant__ SizeImage d_imageSize;

// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR direction vector in
// a float4*, the first point of the lor in a float4, a float* where a posible input must be loaded, 
// a float* where the result will be stored, and a int that says in which mode are we working. 
// The modes availables are: SENSIBILITY_IMAGE -> It doesn't need any input, the output is a Image
//							 PROJECTIO -> The input is a Image, and the output is a Michelogram
//							 BACKPROJECTION -> The input is a Michelogram and the output is a Image
// The size of the volume must be loaded first in the global and constant variable named d_imageSize
// and the size of the michelogram in cuda_michelogram_size
__device__ void CUDA_Siddon (float4* LOR, float4* P0, float* Input, float* Result, int Mode, int indiceMichelogram)

{

  // Coordenadas (x,y,z) de los dos puntos de intersección de la lor con el fov:
  float xCirc_1_mm, xCirc_2_mm, yCirc_1_mm, yCirc_2_mm, zCirc_1_mm, zCirc_2_mm;

  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_xy_1, alpha_xy_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_1, alpha_y_1, alpha_z_1, alpha_x_2, alpha_y_2, alpha_z_2; // Valores de alpha para el punto de salida y entrada al fov.
  float alpha_x_min, alpha_y_min, alpha_z_min, alpha_x_max, alpha_y_max, alpha_z_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.

  // Valores de alpha para recorrer la lor:
  float alpha_x = 10000000, alpha_y = 10000000, alpha_z = 10000000;	// Valores de alpha si avanzo en x o en y, siempre debo ir siguiendo al más cercano.
  float alpha_x_u, alpha_y_u, alpha_z_u;	// Valor de incremento de alpha según avance un píxel en x o en y.
  float alpha_c;	// Valor de alhpa actual, cuando se recorre la lor.

  // Variables relacionadas con los índices de píxeles recorridos:
  int i_min = 0, j_min = 0, k_min = 0;	// Índices (i,j,k) del píxel de entrada al fov.
  int i_max = 0, j_max = 0, k_max = 0;	// Índices (i,j,k) del píxel de salida al fov.
  int i, j, k;	// Índices con que recorro los píxeles de la lor.

  // Incrementos en píxeles. Puede ser +-1 según la dirección de la lor.
  int i_incr = 0, j_incr = 0, k_incr = 0;

  

  // Cantidad de píxeles intersectados:
  float numIntersectedPixels;

  // Punto de entrada y salida al fov trasladado al borde del píxel, ya que en esta versión el píxel
  // de entrada se considera entero, y no desde el punto exacto de itnersección con el círculo:
  float x_1_mm, x_2_mm, y_1_mm, y_2_mm, z_1_mm, z_2_mm;

  // Largo de la lor teniendo en cuenta P0 y P1, y largo de la lor dentro del fov:
  float rayLength_mm, rayLengthInFov_mm;

  

  // Cálculo de intersección de la lor con un fov cilíndrico.
  // Las lors siempre interesectan las caras curvas del cilindro y no las tapas. Ya
  // que el fov de los scanner está limitado por eso.
  // Lo calculo como la intersección entre la recta y una circunferencia de radio cudaRFOV. La ecuación a resolver es:
  // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=cudaRFOV.^2
  // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
  //float c = P0->x*P0->x + P0->y*P0->y - cudaRFOV*cudaRFOV;
  float segundoTermino = sqrt(4.0f*(LOR->x*LOR->x*(d_RadioFov_mm*d_RadioFov_mm-P0->y*P0->y)
    +LOR->y*LOR->y*(d_RadioFov_mm*d_RadioFov_mm-P0->x*P0->x)) + 8.0f*LOR->x*P0->x*LOR->y*P0->y);

  // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
  // Como la debería cruzar en dos puntos hay dos soluciones.
  alpha_xy_1 = (-2*(LOR->x*P0->x+LOR->y*P0->y) + segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
  alpha_xy_2 = (-2*(LOR->x*P0->x+LOR->y*P0->y) - segundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));

  // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
// se recorre desde P0 a P1.
  alpha_min = min(alpha_xy_1, alpha_xy_2);
  alpha_max = max(alpha_xy_1, alpha_xy_2);

  // Coordenadas dentro de la imagen de los dos puntos de entrada:
  x_1_mm = P0->x + LOR->x * alpha_min;
  y_1_mm = P0->y + LOR->y * alpha_min;
  z_1_mm = P0->z + LOR->z * alpha_min;
  
  x_2_mm = P0->x + LOR->x * alpha_max;
  y_2_mm = P0->y + LOR->y * alpha_max;
  z_2_mm = P0->z + LOR->z * alpha_max;

  float offsetZ_mm = 0;//(SCANNER_ZFOV - cudaZFOV)/2;
  if((z_1_mm < offsetZ_mm)||(z_1_mm > (SCANNER_ZFOV-offsetZ_mm)) || (z_2_mm < offsetZ_mm)||(z_2_mm > (SCANNER_ZFOV-offsetZ_mm)))

  {
	// La lor entra por las tapas del clindro del fov:
	printf("Warning: Lor que entra por las tapas del cilindro del FoV.\n");
  }

  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  float d_RadioFov_aux_mm = d_RadioFov_mm - 0.001; // Manganeta para que por error de redondeo no me quede el índice máx en d_imageSize.nPixelsX o Yo Z
  i_min = abs((x_1_mm + d_RadioFov_aux_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = abs((y_1_mm + d_RadioFov_aux_mm)/d_imageSize.sizePixelY_mm); 
  k_min = abs((z_1_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm); 

  i_max = abs((x_2_mm + d_RadioFov_aux_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = abs((y_2_mm + d_RadioFov_aux_mm)/d_imageSize.sizePixelY_mm); // 
  k_max = abs((z_2_mm - offsetZ_mm)/d_imageSize.sizePixelZ_mm);

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
  numIntersectedPixels = fabsf(i_max - i_min) + fabsf(j_max - j_min) + fabsf(k_max - k_min) + 2; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2

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

  

  // Los alpha min y alpha max los tengo calcular para la entrada
  // y salida al primer y último píxel en vez del punto de intersección del fov circular.
  // Entonces a partir del i_min, i_max, j_min, j_max, y las direcciones de las lors, determino
  // cuales serían los valores alpha para los límites de ese píxel si la lor siguiera (que pueden ser dos,
  // límite en el borde x (fila) o borde y (col) del píxel. De los 4 segmentos del píxel, me quedan dos, porque
  // se en que sentido avanza la lor por eso miro la pendiente para el calculo.).
  // Luego con los dos valores de alpha_min y alpha_max, me quedo con el mayor y el menor respectivamente porque son
  // los puntos más cercanos al punto de intersección con el fov circular. De esta forma obtengo el punto de la cara
  // del píxel que se intersecta primero, y ese va a ser la entrada al fov considerando al píxel entero.
  // Antes recalculaba el alpha para i_min o (i_min+1), pero en realidad se que que cuando la pendiente es positiva
  // el próximo costado del píxel es alpha_ +- alpha_x_u
  if (LOR->x>0)
  {
    alpha_x_min = ( -d_RadioFov_mm + i_min * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
    alpha_x_max = ( -d_RadioFov_mm + (i_max+1) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  }
  else if (LOR->x<0)
  {
    alpha_x_min = ( -d_RadioFov_mm + (i_min+1) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	// Limit to the left
    alpha_x_max = ( -d_RadioFov_mm + i_max * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	// Limit to the left
  }
  if(LOR->y > 0)
  {
    alpha_y_min = ( -d_RadioFov_mm + j_min * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
    alpha_y_max = ( -d_RadioFov_mm + (j_max+1) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
  }
  else if (LOR->y < 0)
  {
    alpha_y_min = ( -d_RadioFov_mm + (j_min+1) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
    alpha_y_max = ( -d_RadioFov_mm + j_max * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
  }

  // Es poco probable que toque la tapa del cilindro pero como en realidad es el borde del píxel,
  // lo vuelvo a calcular
  if(LOR->z > 0)
  {
    alpha_z_min = ( offsetZ_mm + k_min * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
    alpha_z_max = ( offsetZ_mm + (k_max+1) * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
  }
  else if (LOR->z < 0)
  {
    alpha_z_min = ( offsetZ_mm + (k_min+1) * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
    alpha_z_max = ( offsetZ_mm + k_max * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
  }
  alpha_min = max(alpha_x_min, max(alpha_y_min, alpha_z_min));
  alpha_max = min(alpha_x_max, min(alpha_y_max, alpha_z_max));

  // Largo de la los dentro del fov. También podría ir calculándolo sumando todos los segmentos.
  // Si tengo en cuenta que para hacer este calculo tengo que hacer como 10 multiplicaciones
  // y una raiz cuadrada, y de la otra forma serán cierta cantidad de sumas dependiendo el tamaño 
  // de la imagen, pero en promedio pueden ser 100. No habría mucha diferencia entre hacerlo de una forma u otra.
  // Puntos exactos de entrada y salida basados en los límtes del píxel:
  x_1_mm = P0->x + alpha_min * LOR->x;
  y_1_mm = P0->y + alpha_min * LOR->y;
  z_1_mm = P0->z + alpha_min * LOR->z;
  
  x_2_mm = P0->x + alpha_max * LOR->x;
  y_2_mm = P0->y + alpha_max * LOR->y;
  z_2_mm = P0->z + alpha_max * LOR->z;

  rayLengthInFov_mm = sqrt((x_2_mm-x_1_mm) * (x_2_mm-x_1_mm) + (y_2_mm-y_1_mm) * (y_2_mm-y_1_mm) + (z_2_mm-z_1_mm) * (z_2_mm-z_1_mm));

  // En alpha_c voy guardando el valor de alpha con el que voy recorriendo los píxeles.
  alpha_c = alpha_min;
  // En alpha_x guardo los próximos límites en todas las coordenadas. En alpha_c está el punto actual, que es el mínimo.
  alpha_x = alpha_x_min + alpha_x_u;
  alpha_y = alpha_y_min + alpha_y_u;
  alpha_z = alpha_z_min + alpha_z_u;

  // Inicialización de i,j a sus valores de entrada.
  i = i_min;
  j = j_min;
  k = k_min;

  // Peso para cada píxel:
  float4 weight = make_float4(0,0,0,0);	// Weight for every pixel
  float nPixelsXY = (d_imageSize.nPixelsX * d_imageSize.nPixelsY);
  rayLength_mm = sqrt(((P0->x + LOR->x) - P0->x) * ((P0->x + LOR->x) - P0->x) 
    + ((P0->y + LOR->y) - P0->y) * ((P0->y + LOR->y) - P0->y)
    + ((P0->z + LOR->z) - P0->z) * ((P0->z + LOR->z) - P0->z));

  int indicePixel;
  // Recorro la lor y guardo los segmentos en la lista de salida.
  for(int m = 0; m < numIntersectedPixels; m++)
  {
    weight.x = i;
    // El índice en Y de las coordenadas de píxeles crece de forma inversa que las coordenadas geométricas,
    // por eso en vez de guardar el j, guardo nPixelsY -j - 1;
    weight.y = d_imageSize.nPixelsY - 1 - j;
    weight.z = k;
    if((alpha_x <= alpha_y) && (alpha_x <= alpha_z))
    {
      // Cruce por el plano x: avanzo en i.
      weight.w = (alpha_x - alpha_c) * rayLength_mm;
      i += i_incr;
      alpha_c = alpha_x;
      alpha_x += alpha_x_u;
    }
    else if((alpha_y <= alpha_x) && (alpha_y <= alpha_z))
    {
      weight.w = (alpha_y - alpha_c) * rayLength_mm;
      // Cruce por el plano y: avanzo en j.
      j += j_incr;
      alpha_c = alpha_y;
      alpha_y += alpha_y_u;
    }
    else
    {
      // Cruce por el plano y: avanzo en j.
      weight.w = (alpha_z - alpha_c) * rayLength_mm;
      k += k_incr;
      alpha_c = alpha_z;
      alpha_z += alpha_z_u;
    }
    if((weight.x<d_imageSize.nPixelsX)&&(weight.y<d_imageSize.nPixelsY)&&(weight.z<d_imageSize.nPixelsZ)&&(weight.x>=0)&&(weight.y>=0)&&(weight.z>=0)&&(weight.w>=0))
    { 
      indicePixel = (int)(weight.x + weight.y * d_imageSize.nPixelsX + weight.z * nPixelsXY);
      switch(Mode)
      {  
	case SENSIBILITY_IMAGE:
	  atomicAdd(Result+indicePixel, weight.w / d_imageSize.sizePixelX_mm);
	  break;
	case PROJECTION:
	  Result[indiceMichelogram] += weight.w / d_imageSize.sizePixelX_mm * Input[indicePixel];
	  break;
	case BACKPROJECTION:
	  atomicAdd(Result+indicePixel, weight.w / d_imageSize.sizePixelX_mm * Input[indiceMichelogram]);
	  break;
      }

    }

  }

}


#endif

