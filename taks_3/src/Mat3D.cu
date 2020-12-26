#include "Mat3D.h"
#include <iostream>
#include <fstream>

Mat3D::Mat3D(int I, int J, int K) :
	_pad(1),
	_I(I), _J(J), _K(K),
	_rI(I+2), _rJ(J+2), _rK(K+2),
	host_data((I+2)*(J+2)*(K+2)),
	device_data((I+2)*(J+2)*(K+2))
{
	_shape[0] = I;
	_shape[1] = J;
	_shape[2] = K;
}

void Mat3D::fill(double value, cudaStream_t stream) {
	host_data.fill(value);
	device_data.fill(value, stream);
}

void Mat3D::toCPU() {
    cudaMemcpy(host_data.data(), device_data.data(), sizeof(double)*_rI*_rJ*_rK, cudaMemcpyDeviceToHost);
}
void Mat3D::toGPU() {
    cudaMemcpy(device_data.data(), host_data.data(), sizeof(double)*_rI*_rJ*_rK, cudaMemcpyHostToDevice);
}

double &Mat3D::operator()(int i, int j, int k) {
	return host_data.at(((i+_pad)*_rJ + (j+_pad))*_rK + (k+_pad));
}


const double &Mat3D::operator()(int i, int j, int k) const {
	return const_cast<Mat3D &>(*this)(i, j, k);
}


const int &Mat3D::shape(int i) const {
	return _shape[i];
}

void Mat3D::slice(int ind, int axis, HostVec &out_slice) const {
	int i_min[3] = {0, 0, 0};
	int i_max[3] = {_I, _J, _K};
	if(axis < 0 || axis > 2) {
		std::cerr << "wrong axis " << axis << std::endl;
	}
	if (ind < -1 || ind > i_max[axis]) {
		std::cerr << "wrong ind " << ind << std::endl;
	}

	i_min[axis] = ind;
	i_max[axis] = ind+1;

	int counter = 0;
	for (int i = i_min[0]; i < i_max[0]; ++i) {
		for (int j = i_min[1]; j < i_max[1]; ++j) {
			for (int k = i_min[2]; k < i_max[2]; ++k) {
				out_slice.at(counter) = (*this)(i, j, k);
				counter++;
			}
		}
	}
}

int Mat3D::sliceLen(int axis) const {
	if(axis < 0 || axis > 2) {
		std::cerr << "wrong axis " << axis << std::endl;
	}

	int dim_size[3] = {_I, _J, _K};
	dim_size[axis] = 1;

	return dim_size[0]*dim_size[1]*dim_size[2];
}


void Mat3D::setSlice(int ind, int axis, HostVec &other_slice) {
	int i_min[3] = {0, 0, 0};
	int i_max[3] = {_I, _J, _K};
	if(axis < 0 || axis > 2) {
		std::cerr << "wrong axis " << axis << std::endl;
	}
	if (ind < -1 || ind > i_max[axis]) {
		std::cerr << "wrong ind " << ind << std::endl;
	}

	i_min[axis] = ind;
	i_max[axis] = ind+1;
	int dim_size[3] = { i_max[0]-i_min[0], i_max[1]-i_min[1], i_max[2]-i_min[2] };
	if (dim_size[0]*dim_size[1]*dim_size[2] != static_cast<int>(other_slice.size())) {
		std::cerr << "shape (" << dim_size[0] << ", " <<
					dim_size[1] << ", " << dim_size[2] <<
					"can't be filled by vector" << other_slice.size() << std::endl;
	}

	int slice_ind = 0;
	for (int i = i_min[0]; i < i_max[0]; ++i) {
		for (int j = i_min[1]; j < i_max[1]; ++j) {
			for (int k = i_min[2]; k < i_max[2]; ++k) {
				(*this)(i, j, k) = other_slice.at(slice_ind);
				slice_ind++;
			}
		}
	}
}


void Mat3D::setZeroSlice(int ind, int axis) {
	int i_min[3] = {0, 0, 0};
	int i_max[3] = {_I, _J, _K};
	if(axis < 0 || axis > 2) {
		std::cerr << "wrong axis " << axis << std::endl;
	}
	if (ind < -1 || ind > i_max[axis]) {
		std::cerr << "wrong ind " << ind << std::endl;
	}

	i_min[axis] = ind;
	i_max[axis] = ind+1;

	for (int i = i_min[0]; i < i_max[0]; ++i) {
		for (int j = i_min[1]; j < i_max[1]; ++j) {
			for (int k = i_min[2]; k < i_max[2]; ++k) {
				(*this)(i, j, k) = 0.0;
			}
		}
	}
}


void Mat3D::print(bool padded) const {
	int pad = 0;
	int I = _I;
	int J = _J;
	int K = _K;
	if (padded) {
		pad = -1;
		I = _rI-1;
		J = _rJ-1;
		K = _rK-1;
	}

	std::cout << "[\n"; 
	for (int i = pad; i < I; ++i) {
		std::cout << "  [\n"; 
		for (int j = pad; j < J; ++j) {
			std::cout << "    [";
			for (int k = pad; k < K; ++k)	{
				std::cout << (*this)(i,j,k);
				if (k < K-1) {
					std::cout << ", ";
				}
			}
			std::cout << "]";
			std::cout << std::endl;
		}
		std::cout << "  ]" << std::endl;
	}
	std::cout << "]" << std::endl;

}

void Mat3D::saveToCSV(const char* filename) const {
	std::ofstream out_file;
	out_file.open(filename);
	if (!out_file) {
		std::cerr << "can't open file " << filename << std::endl;
		return;
	}
	out_file << _I <<","<< _J <<","<< _K << std::endl; 
	for (int i = 0; i < _I; ++i) {
		for (int j = 0; j < _J; ++j) {
			for (int k = 0; k < _K; ++k) {
				out_file << (*this)(i,j,k);
				if (k < _K-1) {
					out_file << ",";
				}
			}
			out_file << std::endl;
		}
	}

}

void Mat3D::save(const char* filename) const {
	std::ofstream out_file;
	out_file.open(filename, std::ios::out | std::ios::binary);
	if (!out_file) {
		std::cerr << "can't open file " << filename << std::endl;
		return;
	}
	out_file.write((char *) host_data.data(), host_data.size()*sizeof(double));
}