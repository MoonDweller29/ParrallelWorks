#pragma once
#include <cuda.h>
#include "Stream.h"

void addVector(float* left, float* right, float* result, int SIZE, const Stream& stream);