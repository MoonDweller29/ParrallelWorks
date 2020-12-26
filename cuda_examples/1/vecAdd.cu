#include <cuda.h>
#include "cuda_macro.h"
#include "vecAdd.h"

__constant__ int mult;


// gridDim.x - grid size x
// index = blockIdx.x * blockDim.x + threadIdx.x;

// Функция сложения двух векторов
__global__ void addVector(float* left, float* right, float* result)
{
  //Получаем id текущей нити.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  //Расчитываем результат.
  result[idx] = left[idx] + mult*right[idx];
}

void addVector(float* left, float* right, float* result, int SIZE, const Stream& stream) {
	int curr_mult = 2;
    SAFE_CALL( cudaMemcpyToSymbol(mult, &curr_mult, sizeof(int)) )

    dim3 gridSize = dim3(SIZE/512, 1, 1);    //Размер используемого грида
    dim3 blockSize = dim3(512, 1, 1); //Размер используемого блока

    addVector<<<gridSize, blockSize, 0, *stream>>>(left, right, result);
}
