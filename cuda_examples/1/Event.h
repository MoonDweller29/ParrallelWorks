#pragma once
#include <cuda.h>


class Event {
public:
    Event();
    ~Event();

    cudaEvent_t operator*() const { return event; }
    cudaError_t record(cudaStream_t stream = 0);
    static cudaError_t wait(const Event &event);
private:
    cudaEvent_t event;
};