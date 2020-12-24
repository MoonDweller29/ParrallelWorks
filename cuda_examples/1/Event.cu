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

Event::~Event() {
    cudaEventDestroy(event);
}