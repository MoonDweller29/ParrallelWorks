#include "F3D_f4.h"
#include "IFunction4D.h"

F3D_f4::F3D_f4(const IFunction4D& f4) :
	_f4(f4), _t(0)
{}

double F3D_f4::operator()(double x, double y, double z) const {
	return _f4(x, y, z, _t);
}

void F3D_f4::setT(double t) {
	_t = t;
}