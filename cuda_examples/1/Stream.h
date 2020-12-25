#pragma once
#include <cuda.h>

class Event;

class Stream {
public:
	Stream();
	~Stream();

	cudaStream_t operator*() const { return stream; }
	cudaError_t wait(const Event& event) const; 
	cudaError_t synchronize() const;
private:
	cudaStream_t stream;
};