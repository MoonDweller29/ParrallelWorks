#pragma once
#include <cuda.h>


class Event {
public:
    Event();
    ~Event();

    cudaEvent_t operator*() const { return event; }
    cudaError_t record(cudaStream_t stream = 0);
    static cudaError_t wait(const Event &event);
    static float elapsedTime(const Event &event_1, const Event &event_2); //implicit sync
private:
    cudaEvent_t event;
};