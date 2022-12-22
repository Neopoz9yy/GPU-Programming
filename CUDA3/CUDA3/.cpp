#include "UtilsMat.h"


int summ(float* A, int size) {
	float s = 0;
	for (int i = 0; i < size; i++)
	{
		s += A[i];
	}
	printf("summ = %f \n", s);
	return s;
}