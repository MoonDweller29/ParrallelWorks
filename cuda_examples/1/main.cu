#include <iostream>
#include <vector>
#include <cuda.h>
#include "cuda_macro.h"

// gridDim.x - grid size x
// index = blockIdx.x * blockDim.x + threadIdx.x;

// Функция сложения двух векторов
__global__ void addVector(float* left, float* right, float* result)
{
  //Получаем id текущей нити.
  int idx = threadIdx.x;
  
  //Расчитываем результат.
  result[idx] = left[idx] + right[idx];
}


int main(int argc, char const *argv[])
{
    //Выделяем память под вектора
    const int SIZE = 512;
    float* vec1 = new float[SIZE];
    float* vec2 = new float[SIZE];
    float* vec3 = new float[SIZE];

    //Инициализируем значения векторов
    for (int i = 0; i < SIZE; i++)
    {
        vec1[i] = i;
        vec2[i] = i;
    }

    //Указатели на память видеокарты
    float* devVec1;
    float* devVec2;
    float* devVec3;

    //Выделяем память для векторов на видеокарте
    SAFE_CALL( cudaMalloc((void**)&devVec1, sizeof(float) * SIZE) )
    SAFE_CALL( cudaMalloc((void**)&devVec2, sizeof(float) * SIZE) )
    SAFE_CALL( cudaMalloc((void**)&devVec3, sizeof(float) * SIZE) )

    //Копируем данные в память видеокарты
    SAFE_CALL( cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    SAFE_CALL( cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    
    dim3 gridSize = dim3(1, 1, 1);    //Размер используемого грида
    dim3 blockSize = dim3(SIZE, 1, 1); //Размер используемого блока

    //Выполняем вызов функции ядра
    addVector<<<gridSize, blockSize>>>(devVec1, devVec2, devVec3);
    checkErr();
    // addVector<<<1, SIZE>>>(devVec1, devVec2, devVec3);

    //Хендл event'а
    cudaEvent_t syncEvent;

    cudaEventCreate(&syncEvent);    //Создаем event
    cudaEventRecord(syncEvent, 0);  //Записываем event
    cudaEventSynchronize(syncEvent);  //Синхронизируем event

    //Только теперь получаем результат расчета
    SAFE_CALL( cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost) )

    for (int i = 0; i < SIZE; i++)
    {
        std::cout<< i <<" : "<< vec3[i] << std::endl;
    }

    cudaEventDestroy(syncEvent);

    cudaFree(devVec1);
    cudaFree(devVec2);
    cudaFree(devVec3);

    delete[] vec1;
    delete[] vec2;
    delete[] vec3;

    return 0;
}