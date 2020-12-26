#pragma once
#include "Mat3D.h"

void fillU0(Mat3D &block, double L_host[], int Nmin_host[], double h_host[], double a_t_host, cudaStream_t stream);