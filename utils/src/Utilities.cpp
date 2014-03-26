#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>

// Function that searchs the index of the element of an array which
// value is nearer to the parameter Value.
// The index of the array goes from 0 to LENGTH-1
int SearchBin (float* Array, unsigned int LENGTH, float Value)
{
	int i = (unsigned int)floor((float)LENGTH/2);
	int Low = 0;
	int High = LENGTH - 1;
	int Mid;
	float Step = Array[1] - Array[0];	// Step between each bin, it's supposed that every
										// bin is equally spaced
	while (Low <= High) 
	{
       Mid = Low + (unsigned int)floor((float)(High - Low) / 2);  // Note: not (low + high) / 2 !!
       if (((Array[Mid]-(Step/2))<Value)&&(Value<(Array[Mid]+(Step/2))))
            return Mid; // found
       else if (Array[Mid] < Value)
           Low = Mid + 1;
       else
          High = Mid - 1;
   }
   return -1; // not found
}

/// Guardo los datos direccionados por un puntero a disco. Se pasan como par�metro
/// el puntero, la cantidad de de bytes por elemento (dado por el tipo de dato),
/// la cantidad de elementos y el path de destino.
/// El total de bytes escritos va a ser igual a: BytesElement*N
bool SaveRawFile(void* array, unsigned int BytesElement, unsigned int N, char* path)
{
	FILE* fid= fopen(path,"wb");
	if(fid!=NULL)
	{
		fwrite(array, BytesElement, N, fid);
		fclose(fid);
		return true;
	}
	else
		return false;
}
