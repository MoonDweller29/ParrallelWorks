#pragma once
#include "Config.h"
#include "U4D.h"
#include "F3D_f4.h"
#include "Mat3D.h"
#include "mpi.h"


class Solver {
public:
	Solver(const Config &config, int argc, char **argv);
	
	void run(int K);

	~Solver();

private:
	double L[3];
	int N[3]; //grid size
	double h[3];
	bool periodic[3];
	int K;
	double tau;

	int procShape[3]; //grid of processes
	int _coord[3]; //coord of curr process in process grid
	int Nmin[3]; //start index in main grid for process
	int BasicNsize[3]; //size of main part of blocks
	int Nsize[3]; //size of block for process

	int sender_tags[3][2]; //dims * both sides
	MPI_Request send_req[3][2]; //dims * both sides
	const int _root;
	int rank;
	int procCount;
	double procTime[2];

	U4D u;
	F3D_f4 phi;

	Mat3D *blocks[3];
	std::vector<double> slices[3][2];
	void calcBlockSize();
	void allocBlocks();
	void freeBlocks();
	void rotateBlocks();

	//send-recv
	void clearRequests();
	void initTags();
	void waitSend();
	void updateBorders(Mat3D& block);
	void sendBorders(Mat3D& block);
	void recvBorders(Mat3D& block);

	void fillU0(Mat3D &block, const IFunction3D &phi);
	void printErr(Mat3D &block, const IFunction4D &u, double t);
	double laplacian(const Mat3D &block, int i, int j, int k) const;
	void fillU1(const Mat3D &block0, Mat3D &block1);
	void step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2);


	void calcProcGrid();
	int procId(int coord[]);
	void idToCoord(int id, int coord[]);
	void normalizeCoord(int coord[]);
};