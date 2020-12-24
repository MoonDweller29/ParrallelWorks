#pragma once
#include <cuda.h>

class Stream {
public:
	Stream();
	~Stream();

	cudaStream_t operator*() { return stream; }
private:
	cudaStream_t stream;
};