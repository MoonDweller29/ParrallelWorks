#pragma once
#include "Config.h"
#include "U4D.h"
#include "F3D_f4.h"
#include "Mat3D.h"
#include "cuda_utils/CudaVec.h"
#include "cuda_utils/Stream.h"
#include "cuda_utils/Event.h"
#include "CudaSolver.h"
#include "mpi.h"


class Solver {
public:
	Solver(const Config &config, int argc, char **argv);
	
	void run(int K);

	~Solver();

private:
	CudaSolver cudaSolver;
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

	Stream stream1, stream2;
	Event ready_for_reduce;


	Mat3D *blocks[3];
	HostVec out_slices[3][2];
	HostVec in_slices[3][2];
	DeviceVec gpu_slices[3][2];
	Event slice_is_on_host[3][2];

	std::vector<double> errors;
	void calcBlockSize();
	void allocBlocks();
	void allocSlices();
	void initCudaSolver();
	void freeBlocks();
	void rotateBlocks();

	//send-recv
	void clearRequests();
	void initTags();
	void waitSend();
	void sliceToCPU(int dim, int i);
	void sliceToGPU(int dim, int i, HostVec &slice);

	void copySlicesToCPU(Mat3D &block);
	void copySlicesToBlock(Mat3D &block);
	void updateBorders(Mat3D& block);
	void setZeroSlices(Mat3D &block);
	void sendBorders(Mat3D& block);
	void recvBorders(Mat3D& block);

	void fillU0(Mat3D &block, const IFunction3D &phi);
	void printErr(double t);
	double laplacian(const Mat3D &block, int i, int j, int k) const;
	void fillU1(const Mat3D &block0, Mat3D &block1);
	void step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2);


	void calcProcGrid();
	int procId(int coord[]);
	void idToCoord(int id, int coord[]);
	void normalizeCoord(int coord[]);
};