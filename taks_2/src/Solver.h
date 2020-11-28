#pragma once
#include "Config.h"

class Solver {
public:
	Solver(const Config &config);
private:
	double L[3];
	double T;
};