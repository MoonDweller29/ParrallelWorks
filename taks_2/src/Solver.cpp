#include "Solver.h"
#include "INIReader.h"
#include "mpi.h"
#include <iostream>
#include <vector>

static std::vector<int> prime_divisors(int x) {
	std::vector<int> divs;

	int curr_div = 2;
	int max_div = (x+1)/2;
	while(x > 1 && curr_div <= max_div) {
		if (x % curr_div == 0) {
			divs.push_back(curr_div);
			x = x / curr_div;
		} else {
			curr_div++;
		}
	}

	if (divs.empty()) {
		divs.push_back(x);
	}

	return divs;
}


Solver::Solver(const Config &config, int argc, char **argv) :
	_tag(0), _root(0)
{
	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &procCount);

  	procTime[0] = MPI_Wtime();

  	calcProcGrid();
	if (rank == _root) {
		config.print();
		std::cout << "MPI INFO\n" <<
			"num processes: " << procCount << std::endl <<
			"procShape: " <<procShape[0]<<","<<procShape[1]<<","<<procShape[2]<< std::endl;
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	// int coord[3];
	// idToCoord(rank, coord);
	// std::cout << rank <<": "<< 
	// 	coord[0]<<","<<coord[1]<<","<<coord[2] <<
	// 	" - "<<procId(coord) << std::endl;

	for (int i = 0; i < 3; ++i)
	{
		L[i] = config.L[i];
		N[i] = config.N[i];
		h[i] = config.h[i];
		periodic[i] = config.periodic[i];
	}
	K = config.K;
	tau = config.tau; 
}

void Solver::calcProcGrid() {
	std::vector<int> divs = prime_divisors(procCount);
	procShape[0] = 1;
	procShape[1] = 1;
	procShape[2] = 1;

	int divs_len = divs.size();
	int dim = 0;
	for (int i = 0; i < divs_len; ++i)
	{
		procShape[dim] *= divs[i];
		++dim;
	}
}

int Solver::procId(int coord[]) {
	return (coord[0]*procShape[1] + coord[1])*procShape[2] + coord[2];
}

void Solver::idToCoord(int id, int coord[]) {
	coord[2] = id % procShape[2];
	id /= procShape[2];
	coord[1] = id % procShape[1];
	coord[0] = id / procShape[1];
}


Solver::~Solver() {
	procTime[1] = MPI_Wtime();
	if (rank == _root) {
		std::cout << "elapsed time = " << procTime[1] - procTime[0] << std::endl;
	}

	MPI_Finalize();
}