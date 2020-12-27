#pragma once
#include <cuda.h>
#include "Stream.h"
#include "CudaVec.h"

float max_reduce(HostVec& arr, const Stream& stream);