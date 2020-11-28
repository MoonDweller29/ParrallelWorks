#include <iostream>
#include "Mat3D.h"
#include "Solver.h"
#include <vector>
#include <stdlib.h>


// #define OUT(arg) #arg" = " << arg

int main(int argc, char **argv)
{
	if (argc < 2){
		return -1;
	}

	Config conf(argv[1]);
	Solver solver(conf, argc, argv);
	solver.run(conf.K);

	// std::vector<int> divs = prime_divisors(atoi(argv[1]));
	// for (int i = 0; i < divs.size(); ++i)
	// {
	// 	std::cout << divs[i] << std::endl;
	// }

	// int x = 1;
	// for (int i = 0; i < divs.size(); ++i)
	// {
	// 	x*=divs[i];
	// }
	// std::cout << "ORIG_X " << x << std::endl;

	// Mat3D mat(2, 3, 3);
	// int counter = 0;	
	// for (int i = 0; i < mat.shape(0); ++i)
	// {
	// 	for (int j = 0; j < mat.shape(1); ++j)
	// 	{
	// 		for (int k = 0; k < mat.shape(2); ++k)
	// 		{
	// 			counter++;
	// 			mat(i,j,k) = counter;
	// 		}
	// 	}
	// }
	// mat.print();
	// mat.print(true);

	// std::vector<double> slice = mat.slice(0, 0);
	// mat.setSlice(1, 0, slice);
	// mat.setSlice(-1, 0, slice);


	// slice = mat.slice(0, 2);
	// mat.setSlice(1, 2, slice);

	// mat.setZeroSlice(0, 0);

	// std::cout << "changes" << std::endl;

	// mat.print();
	// mat.print(true);

	// mat.save("1.mat");

	return 0;
}