#include "Solver.h"
#include "INIReader.h"
#include <iostream>


Solver::Solver(const Config &config) :
	T(config.T)
{
}