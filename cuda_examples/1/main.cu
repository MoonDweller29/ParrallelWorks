#include <iostream>
#include <vector>
#include <cuda.h>
#include "cuda_macro.h"
#include "vecAdd.h"
#include "Event.h"
#include "Stream.h"
#include "CudaVec.h"

#include "reduce.h"
#include <cstdlib>
#include <ctime>

template <typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

void copy_vec(HostVec &dest, HostVec &src) {
    for (int i = 0; i < src.size(); ++i) {
        float value = src[i];
        dest[i] = value;
    }
}

int get_rand_num(int min, int max) {
    return min + (std::rand() % static_cast<int>(max - min + 1));
}

void find_max(HostVec &arr) {
    float _max = -1;

    for (int i = 0; i < arr.size(); ++i) {
        _max = max(_max, arr[i]);
    }

    std::cout << "CPU MAX = " << _max << std::endl;
}

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

    std::srand(std::time(nullptr));
    int random_variable = std::rand();
    std::cout << "Random value on [0 " << RAND_MAX << "]: " 
              << random_variable << '\n';

    int test_size = 131;
    HostVec orig_host(test_size);
    HostVec copy_host(test_size);
    for (int i = 0; i < orig_host.size(); ++i) {
        orig_host[i] = get_rand_num(0, 50);
    }

    float real_max = get_rand_num(51, 150);
    std::cout << "real_max = " << real_max << std::endl;
    copy_vec(copy_host, orig_host);
    int rand_ind = get_rand_num(0, test_size);
    std::cout << "rand_ind = " << rand_ind << std::endl;

    copy_host[rand_ind] = real_max;
    find_max(copy_host);
    float gpu_max = max_reduce(copy_host, stream1);
    std::cout << "GPU MAX = " << gpu_max << std::endl;

    std::cout << "=========================\n";
    copy_vec(copy_host, orig_host);
    rand_ind = 0;
    std::cout << "rand_ind = " << rand_ind << std::endl;
    copy_host[rand_ind] = real_max;
    find_max(copy_host);
    gpu_max = max_reduce(copy_host, stream1);
    std::cout << "GPU MAX = " << gpu_max << std::endl;

    std::cout << "=========================\n";
    copy_vec(copy_host, orig_host);
    rand_ind = test_size - 1;
    std::cout << "rand_ind = " << rand_ind << std::endl;
    copy_host[rand_ind] = real_max;
    find_max(copy_host);
    gpu_max = max_reduce(copy_host, stream1);
    std::cout << "GPU MAX = " << gpu_max << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << copy_host[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}