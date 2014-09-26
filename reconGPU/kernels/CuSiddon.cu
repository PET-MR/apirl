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
	float SegundoTermino = sqrt(4*(LOR->x*LOR->x*(d_RadioFov_mm*d_RadioFov_mm-P0->y*P0->y)
		+LOR->y*LOR->y*(d_RadioFov_mm*d_RadioFov_mm-P0->x*P0->x)) + 8*LOR->x*P0->x*LOR->y*P0->y);
	/// Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
	/// Como la deberïṡẄa cruzar en dos puntos hay dos soluciones.
	float alpha_xy_1 = (-2*(LOR->x*P0->x+LOR->y*P0->y) + SegundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));
	float alpha_xy_2 = (-2*(LOR->x*P0->x+LOR->y*P0->y) - SegundoTermino)/(2*(LOR->x*LOR->x+LOR->y*LOR->y));

	// Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
	  // se recorre desde P0 a P1.
	  float alpha_min = min(alpha_xy_1, alpha_xy_2);
	  float alpha_max = max(alpha_xy_1, alpha_xy_2);
	// Coordenadas dentro de la imagen de los dos puntos de entrada:
	/// Ahora calculo los dos puntos (X,Y)
	// float X_Circ_1 = LOR.P0.X + alpha_xy_1*LOR.Vx;
	// float Y_Circ_1 = LOR.P0.Y + alpha_xy_1*LOR.Vy;
	// float Z_Circ_1 = LOR.P0.Z + alpha_xy_1*LOR.Vz;
	float3 puntoEntrada = make_float3(P0->x + alpha_min*LOR->x,P0->y + alpha_min*LOR->y, P0->z + alpha_min*LOR->z);	
	// float X_Circ_2 = LOR.P0.X + alpha_xy_2*LOR.Vx;	
	// float Y_Circ_2 = LOR.P0.Y + alpha_xy_2*LOR.Vy;
	// float Z_Circ_2 = LOR.P0.Z + alpha_xy_2*LOR.Vz;
	float3 puntoSalida = make_float3(P0->x + alpha_max*LOR->x,P0->y + alpha_max*LOR->y, P0->z + alpha_max*LOR->z);	

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
	/*if((sqrt(MinValue.x*MinValue.x+MinValue.y+MinValue.y)>(d_RadioFov_mm))||(sqrt(MaxValue.x*MaxValue.x+MaxValue.y+MaxValue.y)>(d_RadioFov_mm))
		||(MinValue.z<0)||(MaxValue.z<0)||(MinValue.z>(d_AxialFov_mm))||(MaxValue.z>(d_AxialFov_mm)))
		return;	// Salgo porque no es una lor vïṡẄlida
*/
	indexes_min.x = floorf((puntoEntrada.x + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
	indexes_min.y = floorf((puntoEntrada.y + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); 
	indexes_min.z = floorf((puntoEntrada.z - 0)/d_imageSize.sizePixelZ_mm); //

	// Calculus of end pixel
	float3 indexes_max = make_float3(0,0,0);
	indexes_max.x = floorf((puntoSalida.x + d_RadioFov_mm)/d_imageSize.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
	indexes_max.y = floorf((puntoSalida.y + d_RadioFov_mm)/d_imageSize.sizePixelY_mm); // 
	indexes_max.z = floorf((puntoSalida.z - 0)/d_imageSize.sizePixelZ_mm);	// indexes_max.z = floorf((puntoEntrada.z - OffsetZ)/d_imageSize.sizePixelZ_mm);

	/// EstïṡẄ dentro del FOV? Para eso verifico que el rango de valores de i, de j y de k estïṡẄ al menos parcialmente dentro de la imagen.
	/*if(((indexes_min.x<0)&&(indexes_max.x<0))||((indexes_min.y<0)&&(indexes_max.y<0))||((indexes_min.z<0)&&(indexes_max.z<0))||((indexes_min.x>=d_imageSize.nPixelsX)&&(indexes_max.x>=d_imageSize.nPixelsX))
		||((indexes_min.y>=d_imageSize.nPixelsY)&&(indexes_max.y>=d_imageSize.nPixelsY))||((indexes_min.z>=d_imageSize.nPixelsZ)&&(indexes_max.z>=d_imageSize.nPixelsZ)))
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
	float3 alpha_u = make_float3(fabsf(d_imageSize.sizePixelX_mm / (LOR->x)),fabsf(d_imageSize.sizePixelY_mm / (LOR->y)),fabsf(d_imageSize.sizePixelZ_mm / (LOR->z))); //alpha_u.x = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
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
		alpha.x = ( -d_RadioFov_mm + (indexes_min.x + incr.x) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	//The formula is (indexes_min.x+incr.x) because que want the limit to the next change of pixel
	else if (LOR->x<0)
		alpha.x = ( -d_RadioFov_mm + (indexes_min.x) * d_imageSize.sizePixelX_mm - P0->x ) / LOR->x;	// Limit to the left
	else
		alpha.x = 1000000;
	if	(alpha.x <0)		// If its outside the FOV o set to a big value so it doesn't bother
		alpha.x = 1000000;

	if(LOR->y > 0)
		alpha.y = ( -d_RadioFov_mm + (indexes_min.y + incr.y) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
	else if (LOR->y < 0)
		alpha.y = ( -d_RadioFov_mm + (indexes_min.y) * d_imageSize.sizePixelY_mm - P0->y ) / LOR->y;
	else
		alpha.y = 1000000;
	if	(alpha.y <0)
		alpha.y = 1000000;
	/*
	if(LOR->z > 0)
		alpha.z = ( OffsetZ + (indexes_min.z + incr.z) * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
	else if (LOR->z < 0)
		alpha.z = ( OffsetZ + (indexes_min.z) * d_imageSize.sizePixelZ_mm - P0->z ) / LOR->z;
	else	// Vz = 0 -> The line is paralles to z axis, I do alpha.z the fmaxf value
		alpha.z = 1000000;
	if	(alpha.z <0)
		alpha.z = 1000000; */

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
	  if((Weight.x<d_imageSize.nPixelsX)&&(Weight.y<d_imageSize.nPixelsY)&&(Weight.z<d_imageSize.nPixelsZ))
	  {
	    
	    switch(Mode)
	    {  
	      case SENSIBILITY_IMAGE:
		Result[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))] 
		  += Weight.w;
		break;
	      case PROJECTION:
		Result[indiceMichelogram] += Weight.w * Input[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))];
		 /*if(Result[indiceMichelogram] < 0)
		  */printf("Proy. Bin negativo: %f. Pixel Value: %f Pixel coord: %f %f %f Weight: %f Lor: %d \n", Result[indiceMichelogram], Input[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))], Weight.x, Weight.y, Weight.z, Weight.w,indiceMichelogram);
		 break;
	      case BACKPROJECTION:
		Result[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))] 
		  += Weight.w * Input[indiceMichelogram];
		  /*if(Result[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))] < 0)
		  */printf("Backproy Pixel negativo: %f. Pixel: %f %f %f Weight: %f Lor: %d \n", Result[(int)(Weight.x + Weight.y * d_imageSize.nPixelsX + Weight.z * (d_imageSize.nPixelsX * d_imageSize.nPixelsY))], Weight.x, Weight.y, Weight.z, Weight.w, indiceMichelogram);
		
		break;
	    }
	  }
	}
}
#endif

