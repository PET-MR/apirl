#include <stdio.h>
#include <stdlib.h>
/// Class that handles SparseMatrix. It has an NZ_R array with the amount of non zero elemnts per Row.
/// It has a Columns dynamic array with all the columns indexes of the non zeros elements, sorted by row, so
/// the size of this array is the total number of non zero elements in the matrix.
/// Finally it has a dynamic array of the size of the Columns array, in which all tehe values of the
/// non zero elemnts are saved.
/// The index of the element matriz goes from 0 to Rows-1, and from 0 to Columns-1
class SparseMatrix
{
public:
	unsigned int Rows;
	unsigned int Columns;
	
	unsigned int* NZ_R;		// Non Zero elements by Row
	unsigned int* NZ_C;		// Columns of the Non Zero elements, sorted by Row
	float* Values;			// Values of the Non Zero elements Sorted by Row and Columns

	SparseMatrix (unsigned int Rows, unsigned int Columns)
	{
		SparseMatrix::Rows = Rows;
		SparseMatrix::Columns = Columns;
		SparseMatrix::NZ_R = (unsigned int *) malloc(Rows*sizeof(unsigned int));
		SparseMatrix::NZ_C = (unsigned int *) malloc(sizeof(unsigned int));
		SparseMatrix::Values = (float *) malloc(sizeof(float));
	}

	~SparseMatrix ()
	{
		free(NZ_R);
		free(NZ_C);
		free(Values);
	}

	int AddValue (unsigned int Row, unsigned int Column)
	{
		if ((Row >= SparseMatrix::Rows)||(Column >= SparseMatrix::Columns))
			return false;
		if(isFill(Row,Column))
		{
		
		}
		else
		{

		}
	}

	bool isFill (unsigned int Row, unsigned int Column)
	{
		unsigned int internalRowIndex;
		if ((Row >= SparseMatrix::Rows)||(Column >= SparseMatrix::Columns))
			return false;
		// Busco el indice de los elementos para la columna que voy a analizar
		for(unsigned int i = 0; i < Row; i++)
			internalRowIndex += NZ_R[i];
		for(unsigned int internalColumnIndex = internalRowIndex; internalColumnIndex < internalRowIndex + NZ_R[Row] ; internalColumnIndex++)
		{
			if(SparseMatrix::NZ_C[internalColumnIndex] == Column)
				return true;	//This Element is inside the matrix
			else if(SparseMatrix::NZ_C[internalColumnIndex] > Column)
				return false;	//This element is not in the list
		}
	}
private:
	int Replace (unsigned int Row, unsigned int Column, float Value)
	{
		unsigned int internalRowIndex;
		// Busco el indice de los elementos para la columna que voy a analizar
		for(unsigned int i = 0; i < Row; i++)
			internalRowIndex += NZ_R[i];
		for(unsigned int internalColumnIndex = internalRowIndex; internalColumnIndex < internalRowIndex + NZ_R[Row] ; internalColumnIndex++)
		{
			if(SparseMatrix::NZ_C[internalColumnIndex] == Column)
			{
				Values[internalColumnIndex] = Value;
				return true;	//This Element is inside the matrix			
			}
			else if(SparseMatrix::NZ_C[internalColumnIndex] > Column)
				return false;	//This element is not in the list
		}
	}

	int Insert (unsigned int Row, unsigned int Column, float Value)
	{
		unsigned int internalRowIndex;
		// Busco el indice de los elementos para la columna que voy a analizar
		for(unsigned int i = 0; i < Row; i++)
			internalRowIndex += NZ_R[i];
		for(unsigned int internalColumnIndex = internalRowIndex; internalColumnIndex < internalRowIndex + NZ_R[Row] ; internalColumnIndex++)
		{
			if(SparseMatrix::NZ_C[internalColumnIndex-1] == Column)
			{
				Values[internalColumnIndex] = Value;
				return true;	//This Element is inside the matrix			
			}
			else if(SparseMatrix::NZ_C[internalColumnIndex] > Column)
				return false;	//This element is not in the list
		}
	}
};