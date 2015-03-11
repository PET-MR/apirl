
#include <Siddon.h>

/// ZFOV del sino de GE, despu�s hay que cambiar esto. Ver cuerpo de la funci�n.
#define SCANNER_ZFOV	260.0f
//#define SCANNER_ZFOV	156.96f
// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR in
// a Line3D object which P0 is the P1 of the LOR, the values of the planes in X, Y, Z, and a double pointer
// where all the wieghts will be loaded. It's a double pointer, because is a dynamic array, so the adress
// of the array can change when reallocating memory. In order to not loose the reference in the calling
// function we use a double pointer. The last parameter factor, sets a factor for setting an scalar factor
// for the weights ( for a normal LOR, factor should be 1)
float Siddon (Line3D LOR, Image* image, SiddonSegment** weightsList, unsigned int* lengthList, float factor)
{
  // Coordenadas (x,y,z) de los dos puntos de intersección de la lor con el fov:
  float xCirc_1_mm, xCirc_2_mm, yCirc_1_mm, yCirc_2_mm, zCirc_1_mm, zCirc_2_mm;
  
  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_xy_1, alpha_xy_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_1, alpha_y_1, alpha_z_1, alpha_x_2, alpha_y_2, alpha_z_2; // Valores de alpha para el punto de salida y entrada al fov.
  float alpha_x_min, alpha_y_min, alpha_z_min, alpha_x_max, alpha_y_max, alpha_z_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.
  // Valores de alpha para recorrer la lor:
  float alpha_x, alpha_y, alpha_z;	// Valores de alpha si avanzo en x o en y, siempre debo ir siguiendo al más cercano.
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
  
  // Tamaño de la imagen.
  SizeImage sizeImage = image->getSize();
  // Radio del FOV. 
  float rFov_mm = image->getFovRadio();
  // Largo del FOV.
  float zFov_mm = image->getFovHeight();
	/*if(zFov != (sizeImage.nPixelsZ*sizeImage.sizePixelZ_mm))
	{
	  /// Deben coincidir, si no  signfica que la imagen tiene mal sus par�metros.
	  printf("Siddon: Imagen no v�lida.");
	  return;
	}*/
  lengthList[0] = 0;
  
//   // Cálculo de intersección de la lor con un fov cilíndrico.
//   // Las lors siempre interesectan las caras curvas del cilindro y no las tapas. Ya
//   // que el fov de los scanner está limitado por eso.
//   
//   // Lo calculo como la intersección entre la recta y una circunferencia de radio rFov_mm. La ecuación a resolver es:
//   // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=rFov_mm.^2
//   // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
//   //float c = LOR.P0.X*LOR.P0.X + LOR.P0.Y*LOR.P0.Y - rFov_mm*rFov_mm;
//   float segundoTermino = sqrt(4*(LOR.Vx*LOR.Vx*(rFov_mm*rFov_mm-LOR.P0.Y*LOR.P0.Y)
// 	  +LOR.Vy*LOR.Vy*(rFov_mm*rFov_mm-LOR.P0.X*LOR.P0.X)) + 8*LOR.Vx*LOR.P0.X*LOR.Vy*LOR.P0.Y);
//   // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
//   // Como la debería cruzar en dos puntos hay dos soluciones.
//   alpha_xy_1 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) + segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
//   alpha_xy_2 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) - segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
//   
//   // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
//   // se recorre desde P0 a P1.
//   alpha_min = min(alpha_xy_1, alpha_xy_2);
//   alpha_max = max(alpha_xy_1, alpha_xy_2);

  
  // Para FOV cuadrado:
  // Obtengo la intersección de la lor con las rectas x=-rFov_mm x=rFov_mm y=-rFov_mm y =rFov_mm
  // Para dichos valores verifico que la otra coordenada este dentro de los valores, y obtengo
  // los puntos de entrada y salida de la lor. No me fijo z, porque no debería ingresar por las
  // tapas del cilindro, al menos que haya algún error entre el sinograma y la imagen de entrada.
  float minValueX_mm = -rFov_mm;
  float minValueY_mm = -rFov_mm;
  float maxValueX_mm = rFov_mm;
  float maxValueY_mm = rFov_mm;
  
  
  // Calculates alpha values for the inferior planes (entry planes) of the FOV
  if(LOR.Vx == 0) // Parallel to x axis
  {
    alpha_y_1 = (minValueY_mm - LOR.P0.Y) / LOR.Vy; 
    alpha_y_2 = (maxValueY_mm - LOR.P0.Y) / LOR.Vy;
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
  else if(LOR.Vy == 0) // Parallel to y axis.
  {
    alpha_x_1 = (minValueX_mm - LOR.P0.X) / LOR.Vx;
    alpha_x_2 = (maxValueX_mm - LOR.P0.X) / LOR.Vx;
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
    alpha_x_1 = (minValueX_mm - LOR.P0.X) / LOR.Vx;
    alpha_y_1 = (minValueY_mm - LOR.P0.Y) / LOR.Vy;  
    // Calculates alpha values for superior planes ( going out planes) of the fov
    alpha_x_2 = (maxValueX_mm - LOR.P0.X) / LOR.Vx;	// ValuesX has one more element than pixels in X, thats we can use InputVolume->SizeX as index for the las element
    alpha_y_2 = (maxValueY_mm - LOR.P0.Y) / LOR.Vy;
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
  
  //   // Coordenadas dentro de la imagen de los dos puntos de entrada:
  x_1_mm = LOR.P0.X + LOR.Vx * alpha_min;
  y_1_mm = LOR.P0.Y + LOR.Vy * alpha_min;
  z_1_mm = LOR.P0.Z + LOR.Vz * alpha_min;
  
  x_2_mm = LOR.P0.X + LOR.Vx * alpha_max;
  y_2_mm = LOR.P0.Y + LOR.Vy * alpha_max;
  z_2_mm = LOR.P0.Z + LOR.Vz * alpha_max;
  /// Esto después hay que cambiarlo! Tiene que ir en la clase Michelogram!!!!!!!!!!!!!
  /// Necesito tener el dato del zfov del michelograma, que no lo tengo accesible ahora. Lo pongo a mano, pero
  /// cambiarlo de maner aurgente lo antes posible.!!!
  /// Ente el offsetZ lo calculaba en base al FOV del sinograma, ahora que fov es el de la imagen adquirida. Debo
  /// centrar dicho FOV en el FOV del sinograma y calcular el offsetZ relativo. Esto sería el valor mínimo de Z de la
  /// imagen a reconstruir. Lo puedo obtener del zFOV de la imagen o del sizePixelZ_mm.
  float offsetZ_mm = 0;//(SCANNER_ZFOV - zFov_mm)/2;

  if((z_1_mm < offsetZ_mm)||(z_1_mm > (SCANNER_ZFOV-offsetZ_mm)) || (z_2_mm < offsetZ_mm)||(z_2_mm > (SCANNER_ZFOV-offsetZ_mm)))
  {
	// La lor entra por las tapas del clindro del fov:
	printf("Warning: Lor que entra por las tapas del cilindro del FoV. alpha_min:%f alpha_max:%f x1:%f y1:%f z1: %f x2:%f y2:%f z2:%f offsetZ:%f rFov:%f.", 
	       alpha_min, alpha_max,x_1_mm,y_1_mm, z_1_mm, x_2_mm, y_2_mm, z_2_mm, offsetZ_mm, rFov_mm);
	printf("Puntos LOR: %f %f %f Pendiente:%f %f %f.\n", 
	       LOR.P0.X, LOR.P0.Y,LOR.P0.Z,LOR.Vx, LOR.Vy, LOR.Vz);
  }
  
  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  i_min = floor((x_1_mm + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = floor((y_1_mm + rFov_mm)/sizeImage.sizePixelY_mm); 
  k_min = floor((z_1_mm - offsetZ_mm)/sizeImage.sizePixelZ_mm); 
  i_max = floor((x_2_mm + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = floor((y_2_mm + rFov_mm)/sizeImage.sizePixelY_mm); // 
  k_max = floor((z_2_mm - offsetZ_mm)/sizeImage.sizePixelZ_mm);
  
  // En algunos casos puede quedarme el punto justo en el límite y que el píxel sea el contiguo a la imagen. En ese caso lo reduzco en 1:
//   if(i_min<0)
//     i_min = 0;
//   if(j_min<0)
//     j_min = 0;
//   if(k_min<0)
//     k_min = 0;
//   if(i_max<0)
//     i_max = 0;
//   if(j_max<0)
//     j_max = 0;
//   if(k_max<0)
//     k_max = 0;
//   
//   if(i_max>=sizeImage.nPixelsX)
//     i_max = sizeImage.nPixelsX;
//   if(j_max>=sizeImage.nPixelsY)
//     j_max = sizeImage.nPixelsY;
//   if(k_max>=sizeImage.nPixelsZ)
//     k_max = sizeImage.nPixelsZ;
//   if(i_min>=sizeImage.nPixelsX)
//     i_min = sizeImage.nPixelsX;
//   if(j_min>=sizeImage.nPixelsY)
//     j_min = sizeImage.nPixelsY;
//   if(k_min>=sizeImage.nPixelsZ)
//     k_min = sizeImage.nPixelsZ;
//   
//   // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
//   if(((i_min<0)||(i_max<0))||((j_min<0)||(j_max<0))||((k_min<0)||(k_max<0))||((i_min>=sizeImage.nPixelsX)||(i_max>=sizeImage.nPixelsX))||
// 	((j_min>=sizeImage.nPixelsY)||(j_max>=sizeImage.nPixelsY))||((k_min>=sizeImage.nPixelsZ)||(k_max>=sizeImage.nPixelsZ)))
//   {
// 	lengthList = 0;
// 	return 0;
//   }
  
  // Cantidad de píxeles intersectados:
  numIntersectedPixels = fabs(i_max - i_min) + fabs(j_max - j_min) + fabs(k_max - k_min) + 2; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2

  weightsList[0] = (SiddonSegment*) malloc((size_t)(sizeof(SiddonSegment)* numIntersectedPixels));
  
  // Pixels increments
  i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR.Vx > 0)
	  i_incr = 1;
  else if(LOR.Vx < 0)
	  i_incr = -1;
  if(LOR.Vy > 0)
	  j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR.Vy < 0)
	  j_incr = -1;
  if(LOR.Vz > 0)
	  k_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR.Vz < 0)
	  k_incr = -1;
    
  // Largo de la los dentro del fov. También podría ir calculándolo sumando todos los segmentos.
  // Si tengo en cuenta que para hacer este calculo tengo que hacer como 10 multiplicaciones
  // y una raiz cuadrada, y de la otra forma serán cierta cantidad de sumas dependiendo el tamaño 
  // de la imagen, pero en promedio pueden ser 100. No habría mucha diferencia entre hacerlo de una forma u otra.
  // Puntos exactos de entrada y salida basados en los límtes del píxel:
  x_1_mm = LOR.P0.X + alpha_min * LOR.Vx;
  y_1_mm = LOR.P0.Y + alpha_min * LOR.Vy;
  z_1_mm = LOR.P0.Z + alpha_min * LOR.Vz;
  x_2_mm = LOR.P0.X + alpha_max * LOR.Vx;
  y_2_mm = LOR.P0.Y + alpha_max * LOR.Vy;
  z_2_mm = LOR.P0.Z + alpha_max * LOR.Vz;
  rayLengthInFov_mm = sqrt((x_2_mm-x_1_mm) * (x_2_mm-x_1_mm) + (y_2_mm-y_1_mm) * (y_2_mm-y_1_mm) + (z_2_mm-z_1_mm) * (z_2_mm-z_1_mm));

  // Distancia total de la LOR. Es la distancia entre los puntos P0 y P1, habitualmente, esos son
  // los puntos de la lor sobre el detector.
  rayLength_mm = sqrt(((LOR.P0.X + LOR.Vx) - LOR.P0.X) * ((LOR.P0.X + LOR.Vx) - LOR.P0.X) 
	  + ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y) * ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y)
	  + ((LOR.P0.Z + LOR.Vz) - LOR.P0.Z) * ((LOR.P0.Z + LOR.Vz) - LOR.P0.Z));

  // Incremento en los valores de alpha, según se avanza un píxel en x o en y.
  alpha_x_u = fabs(sizeImage.sizePixelX_mm / (LOR.Vx)); //alpha_x_u = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
  alpha_y_u = fabs(sizeImage.sizePixelY_mm / (LOR.Vy));
  alpha_z_u = fabs(sizeImage.sizePixelZ_mm / (LOR.Vz));
  
  // A partir del (i_min,j_min) voy recorriendo la lor, para determinar el índice del próximo píxel, o sea
  // saber si avanzo en i o en j, debo ver si el cambio se da en x o en y. Para esto en cada avance se calcula
  // el valor de alpha si avanzar un píxel en i (o x) y el valor de alpha en j (o y). De estos dos valores: alpha_x
  // y alpha_y, el que sea menor indicará en que sentido tengo que avanzar con el píxel.
  if (LOR.Vx>0)
    alpha_x = ( -rFov_mm + (i_min + i_incr) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
    alpha_x = ( -rFov_mm + i_min * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  else
    alpha_x = numeric_limits<float>::max();;
  if (alpha_x <0)		// If its outside the FOV que get to the maximum value so it doesn't bother
    alpha_x = numeric_limits<float>::max();
  
  if(LOR.Vy > 0)
    alpha_y = ( -rFov_mm + (j_min + j_incr) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
    alpha_y = ( -rFov_mm + j_min * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else
    alpha_y = numeric_limits<float>::max();
  if (alpha_y <0)
    alpha_y = numeric_limits<float>::max();
  
  if(LOR.Vz > 0)
    alpha_z = ( offsetZ_mm + (k_min + k_incr) * sizeImage.sizePixelZ_mm - LOR.P0.Z ) / LOR.Vz;
  else if (LOR.Vz < 0)
    alpha_z = ( offsetZ_mm + k_min * sizeImage.sizePixelZ_mm - LOR.P0.Z ) / LOR.Vz;
  else
    alpha_z = numeric_limits<float>::max();
  if (alpha_z <0)
    alpha_z = numeric_limits<float>::max();

  // En alpha_c voy guardando el valor de alpha con el que voy recorriendo los píxeles.
  alpha_c = alpha_min;
  
  // Inicialización de i,j a sus valores de entrada.
  i = i_min;
  j = j_min;
  k = k_min;
  // Recorro la lor y guardo los segmentos en la lista de salida.
  for(int m = 0; m < numIntersectedPixels; m++)
  {
      // Cruce por el plano x: avanzo en i.
    weightsList[0][m].IndexX = i;
    // En y avanza también (antes se hacía al revés que la rpimera fila era la de arriba).
    weightsList[0][m].IndexY = j;
    weightsList[0][m].IndexZ = k;
    if((alpha_x <= alpha_y) && (alpha_x <= alpha_z))
    {
      weightsList[0][m].Segment = (alpha_x - alpha_c) * rayLength_mm;
      i += i_incr;
      alpha_c = alpha_x;
      alpha_x += alpha_x_u;
    }
    else if((alpha_y <= alpha_x) && (alpha_y <= alpha_z))
    {
      weightsList[0][m].Segment = (alpha_y - alpha_c) * rayLength_mm;
      // Cruce por el plano y: avanzo en j.
      j += j_incr;
      alpha_c = alpha_y;
      alpha_y += alpha_y_u;
    }
    else
    {
      // Cruce por el plano y: avanzo en j.
      weightsList[0][m].Segment = (alpha_z - alpha_c) * rayLength_mm;
      k += k_incr;
      alpha_c = alpha_z;
      alpha_z += alpha_z_u;
    }
  }
  lengthList[0] = numIntersectedPixels;
  
  return rayLengthInFov_mm;
}

// Siddon algorithm for a plane. To be used in 2D reconstruction.
// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR in
// a Line2D object which P0 is the P1 of the LOR, the size of the image, and a double pointer
// where all the wieghts will be loaded. It's a double pointer, because is a dynamic array, so the adress
// of the array can change when reallocating memory. In order to not loose the reference in the calling
// function we use a double pointer. The last parameter factor, sets a factor for setting an scalar factor
// for the weights ( for a normal LOR, factor should be 1)
// ACLARACIÓN IMPORTANTE: El sistema de coordenadas geométrico crece de izquierda a derecha y abajo hacia arriba,
// pero los índices de los píxeles crece en y de arriba hacia abajo (como una matriz). Por una cuestión de practicidad
// todos los cálculos los hago como si los índices de píxeles crecieran en el mismo sentido que en las coordenadas geométricas,
// pero a la hora de guardarlas coordenadas del índice en Y hago la conversión correspondiente del sistema de coordenadas:
// j_final = NpixelsY - 1 - j;
float Siddon (Line2D LOR, Image* image, SiddonSegment** weightsList, unsigned int* lengthList, float factor)
{
  // Coordenadas (x,y) de los dos puntos de intersección de la lor con el fov:
  float xCirc_1_mm, xCirc_2_mm, yCirc_1_mm, yCirc_2_mm;
  // Valores de entrada y salida para fov cuadrado. Por ahora solo dejamos la parte de fov circular.
  //float minValueX_mm, minValueY_mm, maxValueX_mm, maxValueY_mm;
  
  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_xy_1, alpha_xy_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_1, alpha_y_1, alpha_x_2, alpha_y_2; // Valores de alpha para el punto de salida y entrada al fov.
  float alpha_x_min, alpha_y_min, alpha_x_max, alpha_y_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.
  // Valores de alpha para recorrer la lor:
  float alpha_x, alpha_y;	// Valores de alpha si avanzo en x o en y, siempre debo ir siguiendo al más cercano.
  float alpha_x_u, alpha_y_u;	// Valor de incremento de alpha según avance un píxel en x o en y.
  float alpha_c;	// Valor de alhpa actual, cuando se recorre la lor.
  
  // Variables relacionadas con los índices de píxeles recorridos:
  int i_min = 0, j_min = 0;	// Índices (i,j) del píxel de entrada al fov.
  int i_max = 0, j_max = 0;	// Índices (i,j) del píxel de salida al fov.
  int i, j;	// Índices con que recorro los píxeles de la lor.
  // Incrementos en píxeles. Puede ser +-1 según la dirección de la lor.
  int i_incr = 0, j_incr = 0, k_incr = 0;
  
  // Cantidad de píxeles intersectados:
  float numIntersectedPixels;
  // Punto de entrada y salida al fov trasladado al borde del píxel, ya que en esta versión el píxel
  // de entrada se considera entero, y no desde el punto exacto de itnersección con el círculo:
  float x_1_mm, x_2_mm, y_1_mm, y_2_mm;
  // Largo de la lor teniendo en cuenta P0 y P1, y largo de la lor dentro del fov:
  float rayLength_mm, rayLengthInFov_mm;
  
  // Tamaño de la imagen.
  SizeImage sizeImage = image->getSize();
  // Radio del fov.
  float rFov_mm = image->getFovRadio();
  // se podria usar la funcion: IntersectionLinePlane(LOR, PlaneX, Point3D* IntersectionPoint);
  // para calcular los distintos puntos de interseccion, pero para hcerlo mas eficiente, lo vamos
  // a calcular como Siddon
  lengthList[0] = 0;
  
  // Cálculo de intersección de la lor con un fov circular.
  // Lo calculo como la intersección entre la recta y una circunferencia de radio rFov_mm. La ecuación a resolver es:
  // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=rFov_mm.^2
  // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
  //float c = LOR.P0.X*LOR.P0.X + LOR.P0.Y*LOR.P0.Y - rFov_mm*rFov_mm;
  float segundoTermino = sqrt(4*(LOR.Vx*LOR.Vx*(rFov_mm*rFov_mm-LOR.P0.Y*LOR.P0.Y)
	  +LOR.Vy*LOR.Vy*(rFov_mm*rFov_mm-LOR.P0.X*LOR.P0.X)) + 8*LOR.Vx*LOR.P0.X*LOR.Vy*LOR.P0.Y);
  // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
  // Como la debería cruzar en dos puntos hay dos soluciones.
  alpha_xy_1 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) + segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
  alpha_xy_2 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) - segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
  
  // Ahora calculo los dos puntos (X,Y)
  // Para esta implementación no lo necesito, porque me interesa el índice de píxel de entrada y de salida
  // y no el punto exacto,ya que se considera el píxel entero en el brode del fov.
  /*xCirc_1_mm = LOR.P0.X + alpha_xy_1*LOR.Vx;
  xCirc_2_mm = LOR.P0.X + alpha_xy_2*LOR.Vx;
  yCirc_1_mm = LOR.P0.Y + alpha_xy_1*LOR.Vy;
  yCirc_2_mm = LOR.P0.Y + alpha_xy_2*LOR.Vy;*/
  
  // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
  // se recorre desde P0 a P1.
  alpha_min = min(alpha_xy_1, alpha_xy_2);
  alpha_max = max(alpha_xy_1, alpha_xy_2);
  
  
  /*// Para FOV cuadrado:
  // Obtengo la intersección de la lor con las rectas x=-rFov_mm x=rFov_mm y=-rFov_mm y =rFov_mm
  // Para dichos valores verifico que la otra coordenada este dentro de los valores, y obtengo
  // los puntos de entrada y salida de la lor.	
  float minValueX_mm = -rFov_mm;
  float minValueY_mm = -rFov_mm;
  float maxValueX_mm = rFov_mm;
  float maxValueY_mm = rFov_mm;
  
  // Calculates alpha values for the inferior planes (entry planes) of the FOV
  alpha_x_1 = (minValueX_mm - LOR.P0.X) / LOR.Vx;
  alpha_y_1 = (minValueY_mm - LOR.P0.Y) / LOR.Vy;
  // Calculates alpha values for superior planes ( going out planes) of the fov
  alpha_x_2 = (maxValueX_mm - LOR.P0.X) / LOR.Vx;	// ValuesX has one more element than pixels in X, thats we can use InputVolume->SizeX as index for the las element
  alpha_y_2 = (maxValueY_mm - LOR.P0.Y) / LOR.Vy;

  //alpha min
  alpha_x_min = min(alpha_x_1, alpha_x_2);
  //alpha_x_min = max((float)0, alpha_x_min);		// If alpha_min is negative, we forced it to zero
  alpha_y_min = min(alpha_y_1, alpha_y_2);
  //alpha_y_min = max((float)0, alpha_y_min);
  alpha_min = max(alpha_x_min, alpha_y_min); // alpha_min is the maximum values
						  // bewtween the two alpha values. Because this means that we our inside the FOV
  
  //alpha max
  alpha_x_max = max(alpha_x_1, alpha_x_2);
  alpha_y_max = max(alpha_y_1, alpha_y_2);
  alpha_max = min(alpha_x_max, alpha_y_max);
  // Fin para Fov Cuadrado.*/
  

  // Coordenadas dentro de la imagen de los dos puntos de entrada:
  x_1_mm = LOR.P0.X + LOR.Vx * alpha_min;
  y_1_mm = LOR.P0.Y + LOR.Vy * alpha_min;
  x_2_mm = LOR.P0.X + LOR.Vx * alpha_max;
  y_2_mm = LOR.P0.Y + LOR.Vy * alpha_max;
  // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
  /*if(((x_1_mm*x_1_mm+y_1_mm*y_1_mm)>(rFov_mm*rFov_mm))||((x_2_mm*x_2_mm+y_2_mm*y_2_mm)>(rFov_mm*rFov_mm)))
  {
	lengthList = 0;
	return 0;
  }*/
  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  i_min = floor((x_1_mm + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = floor((y_1_mm + rFov_mm)/sizeImage.sizePixelY_mm); 
  i_max = floor((x_2_mm + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = floor((y_2_mm + rFov_mm)/sizeImage.sizePixelY_mm); // 
  
  // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
  if(((i_min<0)&&(i_max<0))||((j_min<0)&&(j_max<0))||((i_min>=sizeImage.nPixelsX)&&(i_max>=sizeImage.nPixelsX))|| ((j_min>=sizeImage.nPixelsY)&&(j_max>=sizeImage.nPixelsY)))
  {
	lengthList = 0;
	return 0;
  }
  // Cantidad de píxeles intersectados:
  numIntersectedPixels = fabs(i_max - i_min) + fabs(j_max - j_min) + 1; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2
  
  // Memoria para el vector de segmentos de siddon:
  weightsList[0] = (SiddonSegment*) malloc((size_t)(sizeof(SiddonSegment)* numIntersectedPixels));
  
  // Pixels increments
  i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR.Vx > 0)
	  i_incr = 1;
  else if(LOR.Vx < 0)
	  i_incr = -1;
  if(LOR.Vy > 0)
	  j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR.Vy < 0)
	  j_incr = -1;

  // Los alpha min y alpha max los tengo calcular para la entrada
  // y salida al primer y último píxel en vez del punto de intersección del fov circular.
  // Entonces a partir del i_min, i_max, j_min, j_max, y las direcciones de las lors, determino
  // cuales serían los valores alpha para los límites de ese píxel si la lor siguiera (que pueden ser dos,
  // límite en el borde x (fila) o borde y (col) del píxel. De los 4 segmentos del píxel, me quedan dos, porque
  // se en que sentido avanza la lor por eso miro la pendiente para el calculo.).
  // Luego con los dos valores de alpha_min y alpha_max, me quedo con el mayor y el menor respectivamente porque son
  // los puntos más cercanos al punto de intersección con el fov circular. De esta forma obtengo el punto de la cara
  // del píxel que se intersecta primero, y ese va a ser la entrada al fov considerando al píxel entero.
  if (LOR.Vx>0)
	alpha_x_min = ( -rFov_mm + i_min * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	alpha_x_min = ( -rFov_mm + (i_min+1) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  if(LOR.Vy > 0)
	alpha_y_min = ( -rFov_mm + j_min * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	alpha_y_min = ( -rFov_mm + (j_min+1) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  alpha_min = max(alpha_x_min, alpha_y_min);
  if (LOR.Vx>0)
	alpha_x_max = ( -rFov_mm + (i_max+1) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	alpha_x_max = ( -rFov_mm + i_max * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  if(LOR.Vy > 0)
	alpha_y_max = ( -rFov_mm + (j_max+1) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	alpha_y_max = ( -rFov_mm + j_max * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  alpha_max = min(alpha_x_max, alpha_y_max);
  
  // Largo de la los dentro del fov. También podría ir calculándolo sumando todos los segmentos.
  // Si tengo en cuenta que para hacer este calculo tengo que hacer como 10 multiplicaciones
  // y una raiz cuadrada, y de la otra forma serán cierta cantidad de sumas dependiendo el tamaño 
  // de la imagen, pero en promedio pueden ser 100. No habría mucha diferencia entre hacerlo de una forma u otra.
  // Puntos exactos de entrada y salida basados en los límtes del píxel:
  x_1_mm = LOR.P0.X + alpha_min * LOR.Vx;
  y_1_mm = LOR.P0.Y + alpha_min * LOR.Vy;
  x_2_mm = LOR.P0.X + alpha_max * LOR.Vx;
  y_2_mm = LOR.P0.Y + alpha_max * LOR.Vy;
  rayLengthInFov_mm = sqrt((x_2_mm-x_1_mm) * (x_2_mm-x_1_mm) + (y_2_mm-y_1_mm) * (y_2_mm-y_1_mm));

  
  // Distancia total de la LOR. Es la distancia entre los puntos P0 y P1, habitualmente, esos son
  // los puntos de la lor sobre el detector.
  rayLength_mm = sqrt(((LOR.P0.X + LOR.Vx) - LOR.P0.X) * ((LOR.P0.X + LOR.Vx) - LOR.P0.X) 
	  + ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y) * ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y));
	  
  // Incremento en los valores de alpha, según se avanza un píxel en x o en y.
  alpha_x_u = fabs(sizeImage.sizePixelX_mm / (LOR.Vx)); //alpha_x_u = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
  alpha_y_u = fabs(sizeImage.sizePixelY_mm / (LOR.Vy));
  
  // A partir del (i_min,j_min) voy recorriendo la lor, para determinar el índice del próximo píxel, o sea
  // saber si avanzo en i o en j, debo ver si el cambio se da en x o en y. Para esto en cada avance se calcula
  // el valor de alpha si avanzar un píxel en i (o x) y el valor de alpha en j (o y). De estos dos valores: alpha_x
  // y alpha_y, el que sea menor indicará en que sentido tengo que avanzar con el píxel.
  if (LOR.Vx>0)
	  alpha_x = ( -rFov_mm + (i_min + i_incr) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	  alpha_x = ( -rFov_mm + i_min * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  else
	  alpha_x = numeric_limits<float>::max();;
  if	(alpha_x <0)		// If its outside the FOV que get to the maximum value so it doesn't bother
	  alpha_x = numeric_limits<float>::max();
  
  if(LOR.Vy > 0)
	  alpha_y = ( -rFov_mm + (j_min + j_incr) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	  alpha_y = ( -rFov_mm + j_min * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else
	  alpha_y = numeric_limits<float>::max();
  if(alpha_y <0)
	  alpha_y = numeric_limits<float>::max();

  // En alpha_c voy guardando el valor de alpha con el que voy recorriendo los píxeles.
  alpha_c = alpha_min;	
  
  // Inicialización de i,j a sus valores de entrada.
  i = i_min;
  j = j_min;
  // Recorro la lor y guardo los segmentos en la lista de salida.
  for(int m = 0; m < numIntersectedPixels; m++)
  {
	if((alpha_x <= alpha_y))
	{
	  // Cruce por el plano x: avanzo en i.
	  weightsList[0][m].IndexX = i;
	  // El índice en Y crece igual que el x. La primera fila es la de abajo (antes era al revés).
	  weightsList[0][m].IndexY = j;
	  weightsList[0][m].Segment = (alpha_x - alpha_c) * rayLength_mm * factor;
	  i += i_incr;
	  alpha_c = alpha_x;
	  alpha_x += alpha_x_u;
	}
	else
	{
	  // Cruce por el plano y: avanzo en j.
	  weightsList[0][m].IndexX = i;
	  // El índice en Y crece igual que el x. La primera fila es la de abajo (antes era al revés).
	  weightsList[0][m].IndexY = j;
	  weightsList[0][m].Segment = (alpha_y - alpha_c) * rayLength_mm * factor;
	  j += j_incr;
	  alpha_c = alpha_y;
	  alpha_y += alpha_y_u;
	}
  }
  lengthList[0] = numIntersectedPixels;
  
  return rayLengthInFov_mm;
}



// Esta versión de siddon, arranca en P1 hacia P2, siendo P1 y P2 los puntos de entrada y salida al fov.
void Siddon (Point2D point1, Point2D point2, Image* image, SiddonSegment** WeightsList, unsigned int* LengthList, float factor)
{
  // Tamaño de la imagen.
  SizeImage sizeImage = image->getSize();
  float rFov = image->getFovRadio();
  // Lor:
  Line2D LOR;
  LOR.P0 = point1;
  LOR.Vx = point2.X - point1.X;
  LOR.Vy = point2.Y - point1.Y;
  LengthList[0] = 0;
  
  // Al tener ya la entrada y salida del fov para alpha=0 y alpha=1, ya tengo los valroes de alpha_min y alpha_max.
  float alpha_min = 0; 
  float alpha_max = 1;
  
  //Voxel size
  const float dx = sizeImage.sizePixelX_mm;
  const float dy = sizeImage.sizePixelX_mm;

  // Calculus of coordinates of the first pixel (getting in pixel)
  // For x indexes de value in x increases from left to righ in Coordinate System,
  // and also in Pixel indexes. So the reference (offset) is ValueX[0].
  // On the other hand, Y and Z coordinates increase from down to up, and from bottom to top.
  // But the pixel indexes do it in the oposite way, so now the reference ( offset)
  // is ValuesY[InputVolume->SizeY] and ValuesZ[InputVolume->SizeZ] respectively.
  float i_min = 0, j_min = 0, k_min = 0;
  i_min = (LOR.P0.X + LOR.Vx * alpha_min + rFov)/dx; // In X increase of System Coordinate = Increase Pixels.
  j_min = (LOR.P0.Y + LOR.Vy * alpha_min + rFov)/dy; 
  
  //First touch in an x limit of the fov
  if(LOR.Vx>0)
	  i_min = floor(i_min + 0.5);	// To avoid error of caclulations, because of the precision
  else
	  i_min = floor(i_min - 0.5); // If gets inside the FOV by the greatest value we have to go down

  //First touch in an y limit of the fov
  if(LOR.Vy>0)
	  j_min = floor(j_min + 0.5);
  else
	  j_min = floor(j_min - 0.5);
  i_min = floor(i_min);

  // Calculus of end pixel
  float i_max = 0, j_max = 0;
  i_max = (LOR.P0.X + LOR.Vx * alpha_max + rFov)/dx; // In X increase of System Coordinate = Increase Pixels.
  j_max = (LOR.P0.Y + LOR.Vy * alpha_max + rFov)/dy; // 
  
  //First touch in an x limit of the fov
  if(LOR.Vx>0)
	  i_max = floor(i_max - 0.5);
  else
	  i_max = floor(i_max + 0.5);
  j_max = floor(j_max);

  //First touch in an y limit of the fov
  if(LOR.Vy > 0)
	  j_max = floor(j_max - 0.5);
  else
	  j_max = floor(j_max + 0.5);
  i_max = floor(i_max);
  
  // Pixels increments
  int i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR.Vx > 0)
	  i_incr = 1;
  else if(LOR.Vx < 0)
	  i_incr = -1;
  if(LOR.Vy > 0)
	  j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR.Vy < 0)
	  j_incr = -1;

  // Amount of pixels intersected
  float Np = fabs(i_max - i_min) + fabs(j_max - j_min) + 1; // +0 in each dimension(for getting the amount of itnersections) -1 toget pixels> 3x1-1 = +2
  // Allocates memory for the segments
  WeightsList[0] = (SiddonSegment*) malloc((size_t)(sizeof(SiddonSegment)* Np));

  //Distance between thw two points of the LOR, the LOR has to be set in such way that
  // P0 is P1 of the LOR and the point represented by a=1, is P2 of the LOR
  float RayLength = sqrt(((LOR.P0.X + LOR.Vx) - LOR.P0.X) * ((LOR.P0.X + LOR.Vx) - LOR.P0.X) 
	  + ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y) * ((LOR.P0.Y + LOR.Vy) - LOR.P0.Y));
  //Alpha increment per each increment in one plane
  float alpha_x_u = fabs(dx / (LOR.Vx)); //alpha_x_u = DistanciaPixelX / TotalDelRayo - Remember that Vx must be loaded in order to be the diference in X between the two points of the lor
  float alpha_y_u = fabs(dy / (LOR.Vy));
  //Now we go through by every pixel crossed by the LOR
  //We get the alpha values for the starting pixel
  float alpha_x, alpha_y;

  if (LOR.Vx>0)
	  alpha_x = ( -rFov + (i_min + i_incr) * dx - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	  alpha_x = ( -rFov + (i_min) * dx - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  else
	  alpha_x = numeric_limits<float>::max();;
  if	(alpha_x <0)		// If its outside the FOV que get to the maximum value so it doesn't bother
	  alpha_x = numeric_limits<float>::max();

  if(LOR.Vy > 0)
	  alpha_y = ( -rFov + (j_min + j_incr) * dy - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	  alpha_y = ( -rFov + (j_min) * dy - LOR.P0.Y ) / LOR.Vy;
  else
	  alpha_y = numeric_limits<float>::max();
  if	(alpha_y <0)
	  alpha_y = numeric_limits<float>::max();


  float alpha_c = alpha_min;	// Auxiliar alpha value for save the latest alpha vlaue calculated
  //Initialization of first alpha value and update
  //Initialization of i,j,k values with alpha_min
  int i = (unsigned int)i_min;
  int j = (unsigned int)j_min;
  //We start going through the ray following the line directon
  for(int m = 0; m < Np; m++)
  {
	  if((alpha_x <= alpha_y))
	  {
		  // Crossing an x plane
		  WeightsList[0][m].IndexX = i;
		  WeightsList[0][m].IndexY = j;
		  WeightsList[0][m].Segment = (alpha_x - alpha_c) * RayLength * factor;
		  i += i_incr;
		  alpha_c = alpha_x;
		  alpha_x += alpha_x_u;
	  }
	  else
	  {
		  // Crossing y plane
		  WeightsList[0][m].IndexX = i;
		  WeightsList[0][m].IndexY = j;
		  WeightsList[0][m].Segment = (alpha_y - alpha_c) * RayLength * factor;
		  j += j_incr;
		  alpha_c = alpha_y;
		  alpha_y += alpha_y_u;
	  }
	  
  }
  LengthList[0] = Np;
}

float getRayLengthInFov(Line2D LOR, Image* image)
{
  // Variables relacionadas con el parámetro alpha de la recta de la lor.
  float alpha_xy_1, alpha_xy_2;	// Valores de alpha para la intersección de la recta con el círculo del fov.
  float alpha_x_1, alpha_y_1, alpha_x_2, alpha_y_2; // Valores de alpha para el punto de salida y entrada al fov.
  float alpha_x_min, alpha_y_min, alpha_x_max, alpha_y_max;	// Valores de alpha de ambos puntos por coordenada, pero ahora separados por menor y mayor.
  float alpha_min, alpha_max;	// Valores de alpha mínimo y máximo finales, o sea de entrada y salida al fov de la lor.
  // Variables relacionadas con los índices de píxeles recorridos:
  int i_min = 0, j_min = 0;	// Índices (i,j) del píxel de entrada al fov.
  int i_max = 0, j_max = 0;	// Índices (i,j) del píxel de salida al fov.
  int i, j;	// Índices con que recorro los píxeles de la lor.
  // Incrementos en píxeles. Puede ser +-1 según la dirección de la lor.
  int i_incr = 0, j_incr = 0, k_incr = 0;
  
  // Cantidad de píxeles intersectados:
  float numIntersectedPixels;

  // Punto de entrada y salida al fov trasladado al borde del píxel, ya que en esta versión el píxel
  // de entrada se considera entero, y no desde el punto exacto de itnersección con el círculo:
  float x_1_mm, x_2_mm, y_1_mm, y_2_mm;
  // Largo de la lor teniendo en cuenta P0 y P1, y largo de la lor dentro del fov:
  float rayLength_mm, rayLengthInFov_mm;
  
  // Tamaño de la imagen.
  SizeImage sizeImage = image->getSize();
  // Radio del fov.
  float rFov_mm = image->getFovRadio();
  
  // Cálculo de intersección de la lor con un fov circular.
  // Lo calculo como la intersección entre la recta y una circunferencia de radio rFov_mm. La ecuación a resolver es:
  // (X0+alpha*Vx).^2+(Y0+alpha*Vy).^2=rFov_mm.^2
  // alpha = (-2*(Vx+Vy)+sqrt(4*Vx^2*(1-c)+4*Vy^2*(1-c) + 8(Vx+Vy)))/(2*(Vx^2+Vy^2))
  //float c = LOR.P0.X*LOR.P0.X + LOR.P0.Y*LOR.P0.Y - rFov_mm*rFov_mm;
  float segundoTermino = sqrt(4*(LOR.Vx*LOR.Vx*(rFov_mm*rFov_mm-LOR.P0.Y*LOR.P0.Y)
	  +LOR.Vy*LOR.Vy*(rFov_mm*rFov_mm-LOR.P0.X*LOR.P0.X)) + 8*LOR.Vx*LOR.P0.X*LOR.Vy*LOR.P0.Y);
  // Si no corta:
  if(segundoTermino != segundoTermino)
  {
	// NaN
	return 0;
  }
  // Obtengo los valores de alpha donde se intersecciona la recta con la circunferencia.
  // Como la debería cruzar en dos puntos hay dos soluciones.
  alpha_xy_1 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) + segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
  alpha_xy_2 = (-2*(LOR.Vx*LOR.P0.X+LOR.Vy*LOR.P0.Y) - segundoTermino)/(2*(LOR.Vx*LOR.Vx+LOR.Vy*LOR.Vy));
  // Valores de alpha de entrada y de salida. El de entrada es el menor, porque la lor
  // se recorre desde P0 a P1.
  alpha_min = min(alpha_xy_1, alpha_xy_2);
  alpha_max = max(alpha_xy_1, alpha_xy_2);
  
  // Con el alhpa_min y el alpha_max tengo los puntos de entrada y salida al fov. De los cuales obtengo
  // los índices de los píxeles de entrada y salida del fov.
  // En este caso me interesa el píxel de entrada, para luego considerarlo entero,
  // por más que la entrada al fov sea en un punto intermedio:
  i_min = floor((LOR.P0.X + LOR.Vx * alpha_min + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_min = floor((LOR.P0.Y + LOR.Vy * alpha_min + rFov_mm)/sizeImage.sizePixelY_mm); 
  i_max = floor((LOR.P0.X + LOR.Vx * alpha_max + rFov_mm)/sizeImage.sizePixelX_mm); // In X increase of System Coordinate = Increase Pixels.
  j_max = floor((LOR.P0.Y + LOR.Vy * alpha_max + rFov_mm)/sizeImage.sizePixelY_mm); // 
  
  // Verifico que los índices de i y j dieron dentro de la imagen, sino es que que estoy fuera del fov.
  if(((i_min<0)&&(i_max<0))||((j_min<0)&&(j_max<0))||((i_min>=sizeImage.nPixelsX)&&(i_max>=sizeImage.nPixelsX))|| ((j_min>=sizeImage.nPixelsY)&&(j_max>=sizeImage.nPixelsY)))
  {
	  return 0;
  }
  
  // Pixels increments
  i_incr = 0, j_incr = 0, k_incr = 0;	//The increments are zero (perpendicular liine) if Vx = 0 for i, and so on
  if(LOR.Vx > 0)
	  i_incr = 1;
  else if(LOR.Vx < 0)
	  i_incr = -1;
  if(LOR.Vy > 0)
	  j_incr = 1;	// Remeber than in Y and Z the increase in the SystemCoordinate means a decreas in the pixel index
  else if(LOR.Vy < 0)
	  j_incr = -1;

  // Los alpha min y alpha max los tengo calcular para la entrada
  // y salida al primer y último píxel en vez del punto de intersección del fov circular.
  // Entonces a partir del i_min, i_max, j_min, j_max, y las direcciones de las lors, determino
  // cuales serían los valores alpha para los límites de ese píxel si la lor siguiera (que pueden ser dos,
  // límite en el borde x (fila) o borde y (col) del píxel. De los 4 segmentos del píxel, me quedan dos, porque
  // se en que sentido avanza la lor por eso miro la pendiente para el calculo.).
  // Luego con los dos valores de alpha_min y alpha_max, me quedo con el mayor y el menor respectivamente porque son
  // los puntos más cercanos al punto de intersección con el fov circular. De esta forma obtengo el punto de la cara
  // del píxel que se intersecta primero, y ese va a ser la entrada al fov considerando al píxel entero.
  if (LOR.Vx>0)
	alpha_x_min = ( -rFov_mm + i_min * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	alpha_x_min = ( -rFov_mm + (i_min+1) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  if(LOR.Vy > 0)
	alpha_y_min = ( -rFov_mm + j_min * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	alpha_y_min = ( -rFov_mm + (j_min+1) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  alpha_min = max(alpha_x_min, alpha_y_min);
  if (LOR.Vx>0)
	alpha_x_max = ( -rFov_mm + (i_max+1) * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	//The formula is (i_min+i_incr) because que want the limit to the next change of pixel
  else if (LOR.Vx<0)
	alpha_x_max = ( -rFov_mm + i_max * sizeImage.sizePixelX_mm - LOR.P0.X ) / LOR.Vx;	// Limit to the left
  if(LOR.Vy > 0)
	alpha_y_max = ( -rFov_mm + (j_max+1) * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  else if (LOR.Vy < 0)
	alpha_y_max = ( -rFov_mm + j_max * sizeImage.sizePixelY_mm - LOR.P0.Y ) / LOR.Vy;
  alpha_max = min(alpha_x_max, alpha_y_max);
  
  // Largo de la los dentro del fov. También podría ir calculándolo sumando todos los segmentos.
  // Si tengo en cuenta que para hacer este calculo tengo que hacer como 10 multiplicaciones
  // y una raiz cuadrada, y de la otra forma serán cierta cantidad de sumas dependiendo el tamaño 
  // de la imagen, pero en promedio pueden ser 100. No habría mucha diferencia entre hacerlo de una forma u otra.
  // Puntos exactos de entrada y salida basados en los límtes del píxel:
  x_1_mm = LOR.P0.X + alpha_min * LOR.Vx;
  y_1_mm = LOR.P0.Y + alpha_min * LOR.Vy;
  x_2_mm = LOR.P0.X + alpha_max * LOR.Vx;
  y_2_mm = LOR.P0.Y + alpha_max * LOR.Vy;
  rayLengthInFov_mm = sqrt((x_2_mm-x_1_mm) * (x_2_mm-x_1_mm) + (y_2_mm-y_1_mm) * (y_2_mm-y_1_mm));
  
  return rayLengthInFov_mm;
}
