#include "Event.h"

Event::Event() {
    cudaEventCreate(&event);
}

cudaError_t Event::record(cudaStream_t stream) {
    return cudaEventRecord(event, stream);
}

cudaError_t Event::wait(const Event &event) {
    return cudaEventSynchronize(*event);
}

float Event::elapsedTime(const Event &event_1, const Event &event_2) {
    float time;
    wait(event_1);
    wait(event_2);
    cudaEventElapsedTime(&time, *event_1, *event_2);

    return time;
}


Event::~Event() {
    cudaEventDestroy(event);
}