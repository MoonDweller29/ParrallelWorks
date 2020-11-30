#include "omp.h"
#include "Solver.h"
#include "INIReader.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>


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

static int mod(int x, int div) {
    return ((x%div)+div)%div;
}
static int up_div(int x, int div) {
    return (x+div-1)/div;
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
    idToCoord(rank, _coord);
    calcBlockSize();

    
    if (rank == _root) {
        config.print();
        std::cout << "MPI INFO\n" <<
            "num processes: " << procCount << std::endl <<
            "procShape: "<<procShape[0]<<","<<procShape[1]<<","<<procShape[2]<< std::endl<<
            // "BasicNsize: "<<BasicNsize[0]<<","<<BasicNsize[1]<<","<<BasicNsize[2]<< std::endl<<
            "blockShape: "<<Nsize[0]<<","<<Nsize[1]<<","<<Nsize[2]<< std::endl;
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << "ID:" << rank <<": "
    //     "blockShape: "<<Nsize[0]<<","<<Nsize[1]<<","<<Nsize[2]<< std::endl;

    clearRequests();

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
    //  coord[0]<<","<<coord[1]<<","<<coord[2] <<
    //  " - "<<procId(coord) << std::endl;

    initTags();
    allocBlocks();
}


void Solver::calcBlockSize() {
    for (int i = 0; i < 3; ++i) {
        BasicNsize[i] = up_div(N[i], procShape[i]);
        Nmin[i] = BasicNsize[i]*_coord[i];
        Nsize[i] = BasicNsize[i];
        if (_coord[i] == procShape[i] -1) {
            Nsize[i] = N[i]%BasicNsize[i];
            if (Nsize[i] == 0) {
                Nsize[i] = BasicNsize[i];
            }
        }
    }
}

void Solver::clearRequests() {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            send_req[i][j] = MPI_REQUEST_NULL;
        }
    }
}


void Solver::initTags() {
    int counter = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            sender_tags[i][j] = counter;
            ++counter;
        }
    }
}


void Solver::waitSend() {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            MPI_Wait(&(send_req[i][j]), MPI_STATUS_IGNORE);
        }
    }
}


void Solver::sendBorders(Mat3D& block) {
    waitSend();

    for (int axis = 0; axis < 3; ++axis) {
        if (periodic[axis] && (procShape[axis] == 1)) {
            slices[axis][0] = block.slice(0, axis);
            slices[axis][1] = block.slice(Nsize[axis]-1, axis);
            block.setSlice(-1, axis, slices[axis][1]);
            block.setSlice(Nsize[axis], axis, slices[axis][0]);
            continue;
        }
        
        if (_coord[axis] == 0 && !periodic[axis]) {
            block.setZeroSlice(-1, axis);
        } else {
            slices[axis][0] = block.slice(0, axis);
            int recv_coord[3] = {_coord[0], _coord[1], _coord[2]};
            recv_coord[axis] -= 1;
            int recv_rank = procId(recv_coord);
            MPI_Isend(
                slices[axis][0].data(), slices[axis][0].size(), MPI_DOUBLE,
                recv_rank, sender_tags[axis][0], MPI_COMM_WORLD, &(send_req[axis][0])
            );
        }

        if (_coord[axis] == (procShape[axis] -1) && !periodic[axis]) {
            block.setZeroSlice(Nsize[axis], axis);
        }
        else {
            slices[axis][1] = block.slice(Nsize[axis]-1, axis);
            int recv_coord[3] = {_coord[0], _coord[1], _coord[2]};
            recv_coord[axis] += 1;
            int recv_rank = procId(recv_coord);
            MPI_Isend(
                slices[axis][1].data(), slices[axis][1].size(), MPI_DOUBLE,
                recv_rank, sender_tags[axis][1], MPI_COMM_WORLD, &(send_req[axis][1])
            );
        }
    }
}
void Solver::recvBorders(Mat3D& block) {
    for (int axis = 0; axis < 3; ++axis) {
        if ( procShape[axis] == 1 ) {
            continue;
        }
        
        int slice_len = (Nsize[0]*Nsize[1]*Nsize[2])/Nsize[axis];

        if (!(_coord[axis] == (procShape[axis] -1) && !periodic[axis])) {
            int send_coord[3] = {_coord[0], _coord[1], _coord[2]};
            send_coord[axis] += 1;
            int sender_rank = procId(send_coord);

            std::vector<double> in_slice(slice_len);

            MPI_Recv(in_slice.data(), slice_len, MPI_DOUBLE, 
                sender_rank, sender_tags[axis][0],
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            block.setSlice(Nsize[axis], axis, in_slice);
        }

        
        if (!(_coord[axis] == 0 && !periodic[axis])) {
            int send_coord[3] = {_coord[0], _coord[1], _coord[2]};
            send_coord[axis] -= 1; 
            int sender_rank = procId(send_coord);
            
            std::vector<double> in_slice(slice_len);

            MPI_Recv(in_slice.data(), slice_len, MPI_DOUBLE, 
                sender_rank, sender_tags[axis][1],
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            block.setSlice(-1, axis, in_slice);
        }
    }
}

void Solver::updateBorders(Mat3D& block) {
    sendBorders(block);
    recvBorders(block);
}


void Solver::fillU0(Mat3D &block, const IFunction3D &phi) {
    #pragma omp parallel for
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
    #pragma omp parallel for reduction(max : err_max)
    for (int i = 0; i < Nsize[0]; ++i) {
        for (int j = 0; j < Nsize[1]; ++j) {
            for (int k = 0; k < Nsize[2]; ++k) {
                double curr_err = std::abs(
                    u((i+Nmin[0])*h[0], (j+Nmin[1])*h[1], (k+Nmin[2])*h[2], t) - block(i,j,k)
                );
                if (curr_err > err_max) {
                    err_max = curr_err;
                }
            }
        }
    }

    double out_max;
    MPI_Reduce(&err_max, &out_max, 1, MPI_DOUBLE, MPI_MAX, _root, MPI_COMM_WORLD);
    if (rank == _root) {
        std::cout <<"t = "<<t<<", max_err = "<<out_max<<std::endl;  
    }
}


double Solver::laplacian(const Mat3D &block, int i, int j, int k) const {
    double res = 0;
    res += (block(i-1,j,k) - 2*block(i,j,k) + block(i+1,j,k)) / (h[0]*h[0]);
    res += (block(i,j-1,k) - 2*block(i,j,k) + block(i,j+1,k)) / (h[1]*h[1]);
    res += (block(i,j,k-1) - 2*block(i,j,k) + block(i,j,k+1)) / (h[2]*h[2]);

    return res;
}


void Solver::fillU1(const Mat3D &block0, Mat3D &block1) {
    #pragma omp parallel for
    for (int i = 0; i < Nsize[0]; ++i) {
        for (int j = 0; j < Nsize[1]; ++j) {
            for (int k = 0; k < Nsize[2]; ++k) {
                block1(i,j,k) = block0(i,j,k) + tau*tau*0.5*laplacian(block0, i,j,k);
            }
        }
    }
}


void Solver::step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2) {
    #pragma omp parallel for
    for (int i = 0; i < Nsize[0]; ++i) {
        for (int j = 0; j < Nsize[1]; ++j) {
            for (int k = 0; k < Nsize[2]; ++k) {
                block2(i,j,k) = 2*block1(i,j,k) - block0(i,j,k) + tau*tau*laplacian(block1, i,j,k);
            }
        }
    }
}



void Solver::run(int K) {
    this->K = K;

    fillU0(*(blocks[0]), phi);
    printErr(*(blocks[0]), u, 0);
    fillU1(*(blocks[0]), *(blocks[1]));
    updateBorders(*(blocks[1]));
    printErr(*(blocks[1]), u, tau);

    for (int n = 2; n <= K; ++n) {
        step(*(blocks[0]), *(blocks[1]), *(blocks[2]));
        updateBorders(*(blocks[2]));
        printErr(*(blocks[2]), u, tau*n);
        rotateBlocks();
    }

    // std::stringstream s;
    // s << "res/"<<_coord[0]<<"_"<<_coord[1]<<"_"<<_coord[2]<<".mat";
    // blocks[1]->save(s.str().c_str());
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
        dim = (dim+1)%3;
    }
}

void Solver::normalizeCoord(int coord[]) {
    for (int i = 0; i < 3; ++i) {
        coord[i] = mod(coord[i], procShape[i]);
    }
}


int Solver::procId(int coord[]) {
    normalizeCoord(coord);
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