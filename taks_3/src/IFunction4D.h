#pragma once

class IFunction4D {
public:
	virtual double operator()(double x, double y, double z, double t) const = 0;
	virtual double getA_t() const = 0;
};