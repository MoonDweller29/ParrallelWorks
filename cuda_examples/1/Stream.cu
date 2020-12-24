#include "Stream.h"
#include "Event.h"

Stream::Stream() {
	cudaStreamCreate(&stream);
}

Stream::~Stream() {
	cudaStreamDestroy(stream);
}

cudaError_t Stream::wait(const Event& event) const {
	return cudaStreamWaitEvent(stream, *event, 0);
} 
