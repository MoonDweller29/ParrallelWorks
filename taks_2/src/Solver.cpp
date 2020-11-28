#include "Solver.h"
#include "INIReader.h"
#include "mpi.h"
#include <iostream>
#include <vector>
#include <math.h>


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
	_root(0),
	u(config.L[0], config.L[1], config.L[2]), phi(u)
{
	MPI_Init(&argc, &argv);
  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &procCount);

  	procTime[0] = MPI_Wtime();

	for (int i = 0; i < 3; ++i)
	{
		L[i] = config.L[i];
		N[i] = config.N[i];
		h[i] = config.h[i];
		periodic[i] = config.periodic[i];
	}
	K = config.K;
	tau = config.tau;

  	calcProcGrid();
	idToCoord(rank, coord);
	for (int i = 0; i < 3; ++i) {
		Nsize[i] = N[i] / procShape[i];
		Nmin[i] = Nsize[i]*coord[i];
	}
	if (rank == _root) {
		config.print();
		std::cout << "MPI INFO\n" <<
			"num processes: " << procCount << std::endl <<
			"procShape: "<<procShape[0]<<","<<procShape[1]<<","<<procShape[2]<< std::endl<<
			"blockShape: "<<Nsize[0]<<","<<Nsize[1]<<","<<Nsize[2]<< std::endl;
	}

	double max = rank+0.5;
	if (rank == _root)
		max = 0;
	double out_max;

	MPI_Reduce(&max, &out_max, 1, MPI_DOUBLE, MPI_MAX, _root, MPI_COMM_WORLD);
	if (rank == _root){
		std::cout << "MAX = " << out_max << std::endl;
	}

	// MPI_Barrier(MPI_COMM_WORLD);
	// std::cout << rank <<": "<< 
	// 	coord[0]<<","<<coord[1]<<","<<coord[2] <<
	// 	" - "<<procId(coord) << std::endl;

	allocBlocks();
}


void Solver::fillU0(Mat3D &block, const IFunction3D &phi) {
	for (int i = -1; i <= Nsize[0]; ++i) {
		for (int j = -1; j <= Nsize[1]; ++j) {
			for (int k = -1; k <= Nsize[2]; ++k) {
				block(i,j,k) = phi(
					(i+Nmin[0])*h[0], (j+Nmin[1])*h[1], (k+Nmin[2])*h[2]
				);
			}
		}
	}
}


void Solver::printErr(Mat3D &block, const IFunction4D &u, double t) {
	double err_max = -1;
	for (int i = 0; i < Nsize[0]; ++i) {
		for (int j = 0; j < Nsize[1]; ++j) {
			for (int k = 0; k < Nsize[2]; ++k) {
				err_max = std::max(err_max, abs(
					u((i+Nmin[0])*h[0], (j+Nmin[1])*h[1], (k+Nmin[2])*h[2], t) - block(i,j,k)
				));
			}
		}
	}

	double out_max;
	MPI_Reduce(&err_max, &out_max, 1, MPI_DOUBLE, MPI_MAX, _root, MPI_COMM_WORLD);
	if (rank == _root) {
		std::cout <<"t = "<<t<<", max_err = "<<out_max<<std::endl;	
	}
}



void Solver::run(int K) {
	this->K = K;

	fillU0(*blocks[0], phi);
	printErr(*blocks[0], u, 0);
}


void Solver::allocBlocks() {
	for (int i = 0; i < 3; ++i) {
		blocks[i] = new Mat3D(Nsize[0], Nsize[1], Nsize[2]);
	}
}

void Solver::freeBlocks() {
	for (int i = 0; i < 3; ++i) {
		delete blocks[i];
	}
}

void Solver::rotateBlocks() {
	Mat3D* tmp = blocks[0];
	blocks[0] = blocks[1];
	blocks[1] = blocks[2];
	blocks[2] = tmp;
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
	freeBlocks();
	procTime[1] = MPI_Wtime();
	if (rank == _root) {
		std::cout << "elapsed time = " << procTime[1] - procTime[0] << std::endl;
	}

	MPI_Finalize();
}