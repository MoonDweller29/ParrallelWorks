#pragma once
#include "IFunction4D.h"

class U : public IFunction4D {
	double operator()(double x, double y, double z, double t) const;
}