#pragma once

class IFunction4D {
	double operator()(double x, double y, double z, double t) const = 0;
}