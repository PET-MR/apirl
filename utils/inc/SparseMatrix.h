#pragma once

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
		SparseMatrix::Values = (float *) malloc(sizeof(unsigned int));
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
		if(NZ_R[Row] == 0)
		{
			// If there is no non zero element in that row, I have to add the first one
			SparseMatrix::NZ_C[internalRowIndex-1];
		}
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
