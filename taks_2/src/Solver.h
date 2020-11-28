#pragma once
#include "Config.h"
#include "U4D.h"
#include "F3D_f4.h"
#include "Mat3D.h"

class Solver {
public:
	Solver(const Config &config, int argc, char **argv);
	
	void run(int K);

	~Solver();

private:
	double L[3];
	int N[3];
	double h[3];
	bool periodic[3];
	int K;
	double tau;

	int procShape[3]; //grid of processes
	int coord[3]; //coord of curr process in process grid
	int Nmin[3]; //start index in main grid for process
	int Nsize[3]; //size of block for process

	const int _root;
	int rank;
	int procCount;
	double procTime[2];

	U4D u;
	F3D_f4 phi;

	Mat3D *blocks[3];
	void allocBlocks();
	void freeBlocks();
	void rotateBlocks();

	void fillU0(Mat3D &block, const IFunction3D &phi);

	void calcProcGrid();
	int procId(int coord[]);
	void idToCoord(int id, int coord[]);
};