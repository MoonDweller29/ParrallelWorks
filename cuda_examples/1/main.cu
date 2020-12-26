#include <iostream>
#include <vector>
#include <cuda.h>
#include "cuda_macro.h"
#include "vecAdd.h"
#include "Event.h"
#include "Stream.h"
#include "CudaVec.h"



int main(int argc, char const *argv[])
{
    cudaSetDeviceFlags(cudaDeviceMapHost);

    Event startEvent[2];
    Event syncEvent[2];
    Stream stream1, stream2;

    //Выделяем память под вектора
    const int SIZE = 512*512*200;
    HostVec vec1, vec2, vec3;
    vec1.malloc(SIZE, true);
    vec2.malloc(SIZE, true);
    vec3.malloc(SIZE, true);

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
    PinnedVec pinnedVec;

    //Выделяем память для векторов на видеокарте
    SAFE_CALL( devVec1.malloc(SIZE) )
    SAFE_CALL( devVec2.malloc(SIZE) )
    SAFE_CALL( devVec3.malloc(SIZE) )
    SAFE_CALL( devVec4.malloc(SIZE) )
    SAFE_CALL( pinnedVec.malloc(SIZE) )

    //Копируем данные в память видеокарты
    SAFE_CALL( cudaMemcpy(devVec1.data(), vec1.data(), sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    SAFE_CALL( cudaMemcpy(devVec2.data(), vec2.data(), sizeof(float) * SIZE, cudaMemcpyHostToDevice) )
    

    startEvent[0].record(*stream1);

    //Выполняем вызов функции ядра
    for (int i = 0; i < 1; ++i) {
        // addVector<<<gridSize, blockSize, 0, *stream1>>>(devVec1.data(), devVec2.data(), devVec3.data());
        addVector(devVec1.data(), devVec2.data(), devVec3.data(), SIZE, stream1);
    }
    
    checkErr();
    // addVector<<<1, SIZE>>>(devVec1, devVec2, devVec3);

    //Хендл event'а
    syncEvent[0].record(*stream1); //Записываем event
    // Event::wait(syncEvent[0]); //Синхронизируем event
    
    //Только теперь получаем результат расчета
    // SAFE_CALL( cudaMemcpyAsync(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost, *stream1) )
    cudaMemcpyAsync(vec3.data(), devVec3.data(), sizeof(float) * SIZE, cudaMemcpyDeviceToHost, *stream1);
    
    stream2.wait(syncEvent[0]);
    for (int i = 0; i < 1; ++i)
    {
        // addVector<<<gridSize, blockSize, 0, *stream2>>>(devVec1.data(), devVec2.data(), devVec4.data());
        addVector(devVec1.data(), devVec2.data(), devVec4.data(), SIZE, stream2);
    }
    syncEvent[1].record(*stream2);

    float time = Event::elapsedTime(startEvent[0], syncEvent[0]);
    Event::wait(syncEvent[1]);

    stream1.synchronize();
    stream2.synchronize();
    // cudaDeviceSynchronize();


    for (int i = 0; i < 1; ++i) {
        // addVector<<<gridSize, blockSize, 0, *stream1>>>(devVec1.data(), devVec2.data(), pinnedVec.device());
        addVector(devVec1.data(), devVec2.data(), pinnedVec.device(), SIZE, stream1);
    }
    for (int i = 0; i < 1; ++i)
    {
        // addVector<<<gridSize, blockSize, 0, *stream2>>>(devVec1.data(), devVec2.data(), devVec4.data());
        addVector(devVec1.data(), devVec2.data(), devVec4.data(), SIZE, stream2);
    }

    stream1.synchronize();

    for (int i = SIZE-30; i < SIZE; i++)
    {
        std::cout<< i <<" : "<< vec3[i] <<" __ "<< pinnedVec[i] << std::endl;
    }

    std::cout << "Elapsed Time = " << time << std::endl;

    return 0;
}