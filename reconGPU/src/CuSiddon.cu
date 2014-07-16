
#include <CUDA_Siddon.h>


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


