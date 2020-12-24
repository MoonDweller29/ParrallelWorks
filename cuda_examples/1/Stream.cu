#include "Stream.h"

Stream::Stream() {
	cudaStreamCreate(&stream);
}

Stream::~Stream() {
	cudaStreamDestroy(stream);
}