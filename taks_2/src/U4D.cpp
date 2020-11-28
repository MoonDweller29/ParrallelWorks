#include "U4D.h"
#include <math.h>

U4D::U4D(double Lx, double Ly, double Lz) :
	_Lx(Lx), _Ly(Ly), _Lz(Lz),
	a_t(PI*sqrt( 9/(Lx*Lx) + 4/(Ly*Ly) + 4/(Lz*Lz) ))
{}

double U4D::operator()(double x, double y, double z, double t) const {
	return sin(x*3*PI/_Lx)*sin(y*2*PI/_Ly)*sin(z*2*PI/_Lz)*
		cos(a_t*t + 4*PI);
}