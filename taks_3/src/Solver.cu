#include "omp.h"
#include "Solver.h"
#include "INIReader.h"
#include "cuda_utils/cuda_macro.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>


static void dump_vector(const std::vector<double> &v, const char* filename) {
    std::ofstream out_file;
    out_file.open(filename, std::ios::out | std::ios::binary);
    if (!out_file) {
        std::cerr << "can't open file " << filename << std::endl;
        return;
    }
    out_file.write((char *) v.data(), v.size()*sizeof(double));
}

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

    // double max = rank+0.5;
    // if (rank == _root)
    //     max = 0;
    // double out_max;

    // MPI_Reduce(&max, &out_max, 1, MPI_DOUBLE, MPI_MAX, _root, MPI_COMM_WORLD);
    // if (rank == _root){
    //     std::cout << "MAX = " << out_max << std::endl;
    // }

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::cout << rank <<": "<< 
    //  coord[0]<<","<<coord[1]<<","<<coord[2] <<
    //  " - "<<procId(coord) << std::endl;

    initTags();
    allocBlocks();
    allocSlices();
    initCudaSolver();
}

void Solver::initCudaSolver() {
    cudaSolver.setL(L);
    cudaSolver.seth(h);
    cudaSolver.setTau(tau);
    cudaSolver.setNmin(Nmin);
    cudaSolver.seta_t(phi.getA_t());
    cudaSolver.setBlockSize(Nsize[0], Nsize[1], Nsize[2]);
    cudaSolver.mallocResources(stream1, rank);
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

void Solver::setZeroSlices(Mat3D &block) {
    for (int axis = 0; axis < 3; ++axis) {
        if(!periodic[axis]) {
            if (_coord[axis] == 0) {
                cudaSolver.setZeroSlice(block, -1, axis, *stream1);
                cudaSolver.setZeroSlice(block, 0, axis, *stream1);
            }
            if (_coord[axis] == (procShape[axis] - 1)) {
                cudaSolver.setZeroSlice(block, Nsize[axis], axis, *stream1);
            }
        }
    }
}


void Solver::sendBorders(Mat3D& block) {
    waitSend();

    for (int axis = 0; axis < 3; ++axis) {
        if (periodic[axis] && (procShape[axis] == 1)) {
            Event::wait(slice_is_on_host[axis][1]);
            sliceToGPU(axis, 0, out_slices[axis][1]);
            Event::wait(slice_is_on_host[axis][0]);
            sliceToGPU(axis, 1, out_slices[axis][0]);
            continue;
        }
        
        if ( !(_coord[axis] == 0 && !periodic[axis]) ) {
            Event::wait(slice_is_on_host[axis][0]);

            int recv_coord[3] = {_coord[0], _coord[1], _coord[2]};
            recv_coord[axis] -= 1;
            int recv_rank = procId(recv_coord);
            MPI_Isend(
                out_slices[axis][0].data(), out_slices[axis][0].size(), MPI_DOUBLE,
                recv_rank, sender_tags[axis][0], MPI_COMM_WORLD, &(send_req[axis][0])
            );
        }

        if ( !(_coord[axis] == (procShape[axis] -1) && !periodic[axis]) ) {
            Event::wait(slice_is_on_host[axis][1]);
            
            int recv_coord[3] = {_coord[0], _coord[1], _coord[2]};
            recv_coord[axis] += 1;
            int recv_rank = procId(recv_coord);
            MPI_Isend(
                out_slices[axis][1].data(), out_slices[axis][1].size(), MPI_DOUBLE,
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


            MPI_Recv(in_slices[axis][1].data(), slice_len, MPI_DOUBLE, 
                sender_rank, sender_tags[axis][0],
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sliceToGPU(axis, 1, in_slices[axis][1]);
        }

        
        if (!(_coord[axis] == 0 && !periodic[axis])) {
            int send_coord[3] = {_coord[0], _coord[1], _coord[2]};
            send_coord[axis] -= 1; 
            int sender_rank = procId(send_coord);
            
            MPI_Recv(in_slices[axis][0].data(), slice_len, MPI_DOUBLE, 
                sender_rank, sender_tags[axis][1],
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            sliceToGPU(axis, 0, in_slices[axis][0]);
        }
    }
}

void Solver::updateBorders(Mat3D& block) {
    sendBorders(block);
    recvBorders(block);
    copySlicesToBlock(block);
}


void Solver::fillU0(Mat3D &block, const IFunction3D &phi) {
    cudaSolver.fillU0(block, *stream1);
    setZeroSlices(block);
    copySlicesToCPU(block);
}


void Solver::printErr(double t) {
    stream2.synchronize();
    double err_max = cudaSolver.getErr();

    double out_max;
    MPI_Reduce(&err_max, &out_max, 1, MPI_DOUBLE, MPI_MAX, _root, MPI_COMM_WORLD);
    if (rank == _root) {
        std::cout <<"t = "<<t<<", max_err = "<<out_max<<std::endl;
        errors.push_back(out_max);  
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
    cudaSolver.fillU1(block0, block1, *stream1);
    setZeroSlices(block1);
    copySlicesToCPU(block1);
}


void Solver::step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2) {
    cudaSolver.step(block0, block1, block2, *stream1);
    setZeroSlices(block2);
    copySlicesToCPU(block2);
}

void Solver::sliceToCPU(int dim, int i) {
    SAFE_CALL(cudaMemcpyAsync(
                out_slices[dim][i].data(),
                gpu_slices[dim][i].data(),
                sizeof(double)*gpu_slices[dim][i].size(),
                cudaMemcpyDeviceToHost, *stream1))
    
    slice_is_on_host[dim][i].record(*stream1);
}

void Solver::sliceToGPU(int dim, int i, HostVec &slice) {
    SAFE_CALL(cudaMemcpyAsync(
                gpu_slices[dim][i].data(),
                slice.data(),
                sizeof(double)*gpu_slices[dim][i].size(),
                cudaMemcpyHostToDevice, *stream1))
}


void Solver::copySlicesToCPU(Mat3D &block) {
    //extract slices frim block
    for (int dim = 0; dim < 3; ++dim) {
        if (!periodic[dim]) {
            if (_coord[dim] != 0) {
                cudaSolver.getSlice(block, gpu_slices[dim][0], 0,            dim, *stream1);
            }
            if (_coord[dim] != (procShape[dim] - 1)) {
                cudaSolver.getSlice(block, gpu_slices[dim][1], Nsize[dim]-1, dim, *stream1);
            }
        } else {
            cudaSolver.getSlice(block, gpu_slices[dim][0], 0,            dim, *stream1);
            cudaSolver.getSlice(block, gpu_slices[dim][1], Nsize[dim]-1, dim, *stream1);
        }
    }

    ready_for_reduce.record(*stream1);

    //copy slices to cpu and record events
    for (int dim = 0; dim < 3; ++dim) {
        if (!periodic[dim]) {
            if (_coord[dim] != 0) {
                sliceToCPU(dim, 0);
            }
            if (_coord[dim] != (procShape[dim] - 1)) {
                sliceToCPU(dim, 1);
            }
        } else {
            sliceToCPU(dim, 0);
            sliceToCPU(dim, 1);
        }
    }
}


void Solver::copySlicesToBlock(Mat3D &block) {
    for (int dim = 0; dim < 3; ++dim) {
        if (!periodic[dim]) {
            if (_coord[dim] != 0) {
                cudaSolver.setSlice(block, gpu_slices[dim][0], -1,         dim, *stream1);
            }
            if (_coord[dim] != (procShape[dim] - 1)) {
                cudaSolver.setSlice(block, gpu_slices[dim][1], Nsize[dim], dim, *stream1);
            }
        } else {
            cudaSolver.setSlice(block, gpu_slices[dim][0], -1,         dim, *stream1);
            cudaSolver.setSlice(block, gpu_slices[dim][1], Nsize[dim], dim, *stream1);
        }
    }
}



void Solver::run(int K) {
    this->K = K;

    fillU0(*(blocks[0]), phi);
    cudaSolver.reduceErr(*(blocks[0]), 0, stream2, ready_for_reduce);
    updateBorders(*(blocks[0]));
    printErr(0);

    fillU1(*(blocks[0]), *(blocks[1]));
    cudaSolver.reduceErr(*(blocks[1]), tau, stream2, ready_for_reduce);
    updateBorders(*(blocks[1]));
    printErr(tau);


    for (int n = 2; n <= K; ++n) {
        step(*(blocks[0]), *(blocks[1]), *(blocks[2]));
        updateBorders(*(blocks[2]));
        cudaSolver.reduceErr(*(blocks[2]), tau*n, stream2, ready_for_reduce);
        printErr(tau*n);
        rotateBlocks();
    }

    // std::stringstream s;
    // s << "res/"<<_coord[0]<<"_"<<_coord[1]<<"_"<<_coord[2]<<".mat";
    // blocks[1]->save(s.str().c_str());
    // if (rank == _root) {
    //     std::stringstream s_e;
    //     s_e << N[0] << ".vec";
    //     dump_vector(errors, s_e.str().c_str());
    // }
}


void Solver::allocBlocks() {
    for (int i = 0; i < 3; ++i) {
        blocks[i] = new Mat3D(Nsize[0], Nsize[1], Nsize[2]);
    }
}

void Solver::allocSlices() {
    for (int dim = 0; dim < 3; ++dim) {
        if (!periodic[dim]) {
            if (_coord[dim] != 0) {
                out_slices[dim][0].malloc(blocks[0]->sliceLen(dim), true);
                in_slices[dim][0].malloc(blocks[0]->sliceLen(dim), true);
                gpu_slices[dim][0].malloc(blocks[0]->sliceLen(dim));
            }
            if (_coord[dim] != (procShape[dim] - 1)) {
                out_slices[dim][1].malloc(blocks[0]->sliceLen(dim), true);
                in_slices[dim][1].malloc(blocks[0]->sliceLen(dim), true);
                gpu_slices[dim][1].malloc(blocks[0]->sliceLen(dim));
            }
        } else {
            out_slices[dim][0].malloc(blocks[0]->sliceLen(dim), true);
            out_slices[dim][1].malloc(blocks[0]->sliceLen(dim), true);
            in_slices[dim][0].malloc(blocks[0]->sliceLen(dim), true);
            in_slices[dim][1].malloc(blocks[0]->sliceLen(dim), true);
            gpu_slices[dim][0].malloc(blocks[0]->sliceLen(dim));
            gpu_slices[dim][1].malloc(blocks[0]->sliceLen(dim));
        }
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