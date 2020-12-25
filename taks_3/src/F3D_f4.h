#pragma once
#include "IFunction3D.h"

class IFunction4D;

class F3D_f4 : public IFunction3D {
public:
	F3D_f4(const IFunction4D& f4);
	virtual double operator()(double x, double y, double z) const;
	void setT(double t);
private:
	const IFunction4D& _f4;
	double _t;
};