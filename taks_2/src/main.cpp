#include <iostream>
#include "Mat3D.h"
#include "Solver.h"
#include <vector>
#include <stdlib.h>


int main(int argc, char **argv)
{
	if (argc < 2){
		return -1;
	}

	Config conf(argv[1]);
	Solver solver(conf, argc, argv);
	solver.run(conf.K);

	return 0;
}