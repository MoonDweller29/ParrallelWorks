#pragma once
#include "IFunction4D.h"

#define PI 3.1415926535

class U4D : public IFunction4D {
public:
	U4D(double Lx, double Ly, double Lz);

	virtual double operator()(double x, double y, double z, double t) const;
	virtual double getA_t() const { return a_t; }
private:
	double _Lx, _Ly, _Lz, a_t;
};