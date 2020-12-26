#pragma once

class IFunction3D {
public:
	virtual double operator()(double x, double y, double z) const = 0;
	virtual double getA_t() const = 0;
};