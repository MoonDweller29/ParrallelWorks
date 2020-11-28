#pragma once
#include "Config.h"

class Solver {
public:
	Solver(const Config &config, int argc, char **argv);
	~Solver();
private:
	double L[3];
	int N[3];
	double h[3];
	bool periodic[3];
	int K;
	double tau;
	int procShape[3];

	const int _tag;
	const int _root;
	int rank;
	int procCount;
	double procTime[2];

	void calcProcGrid();
	int procId(int coord[]);
	void idToCoord(int id, int coord[]);
};