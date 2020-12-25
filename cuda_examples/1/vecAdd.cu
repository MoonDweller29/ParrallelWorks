#include <cuda.h>
#include "vecAdd.h"

// gridDim.x - grid size x
// index = blockIdx.x * blockDim.x + threadIdx.x;

// Функция сложения двух векторов
__global__ void addVector(float* left, float* right, float* result)
{
  //Получаем id текущей нити.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  //Расчитываем результат.
  result[idx] = left[idx] + right[idx];
}