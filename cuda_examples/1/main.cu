#include <iostream>
#include <vector>
#include <cuda.h>
#include "cuda_macro.h"
#include "Event.h"
#include "Stream.h"
#include "CudaVec.h"

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


int main(int argc, char const *argv[])
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    Event startEvent[2];
    Event syncEvent[2];
    Stream stream1, stream2;

    //Выделяем память под вектора
    const int SIZE = 512*512*100;
    float* vec1 = new float[SIZE];
    float* vec2 = new float[SIZE];
    float* vec3 = new float[SIZE];

    //Инициализируем значения векторов
    for (int i = 0; i < SIZE; i++)
    {
        vec1[i] = 1;
        vec2[i] = 1;
    }

    //Указатели на память видеокарты
    DeviceVec devVec1;
    DeviceVec devVec2;
    DeviceVec devVec3;
    DeviceVec devVec4;

    //Выделяем память для векторов на видеокарте
    SAFE_CALL( devVec1.malloc(sizeof(float) * SIZE) )
    SAFE_CALL( devVec2.malloc(sizeof(float) * SIZE) )
    SAFE_CALL( devVec3.malloc(sizeof(float) * SIZE) )
    SAFE_CALL( devVec4.malloc(sizeof(float) * SIZE) )

    //Копируем данные в память видеокарты
    SAFE_CALL( cudaMemcpy(devVec1.data(), vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    SAFE_CALL( cudaMemcpy(devVec2.data(), vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    
    dim3 gridSize = dim3(SIZE/512, 1, 1);    //Размер используемого грида
    dim3 blockSize = dim3(512, 1, 1); //Размер используемого блока

    startEvent[0].record(*stream1);

    //Выполняем вызов функции ядра
    for (int i = 0; i < 500; ++i) {
        addVector<<<gridSize, blockSize, 0, *stream1>>>(devVec1.data(), devVec2.data(), devVec3.data());
    }
    
    checkErr();
    // addVector<<<1, SIZE>>>(devVec1, devVec2, devVec3);

    //Хендл event'а
    syncEvent[0].record(*stream1); //Записываем event
    // Event::wait(syncEvent[0]); //Синхронизируем event
    
    
    // stream2.wait(syncEvent[0]);
    for (int i = 0; i < 500; ++i)
    {
        addVector<<<gridSize, blockSize, 0, *stream2>>>(devVec1.data(), devVec2.data(), devVec4.data());
    }

    //Только теперь получаем результат расчета
    // SAFE_CALL( cudaMemcpyAsync(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost, *stream1) )
    cudaMemcpyAsync(vec3, devVec3.data(), sizeof(float) * SIZE, cudaMemcpyDeviceToHost, *stream1);

    float time = Event::elapsedTime(startEvent[0], syncEvent[0]);


    for (int i = SIZE-30; i < SIZE; i++)
    {
        std::cout<< i <<" : "<< vec3[i] << std::endl;
    }

    std::cout << "Elapsed Time = " << time << std::endl;

    delete[] vec1;
    delete[] vec2;
    delete[] vec3;

    return 0;
}