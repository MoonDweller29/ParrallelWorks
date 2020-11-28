#pragma once
#include <vector>

//padded with border width 1 by default
class Mat3D
{
public:
	Mat3D(int I, int J, int K);
	~Mat3D(){};

	double &operator()(int i, int j, int k);
	const double &operator()(int i, int j, int k) const;
	const int &shape(int i) const;
	std::vector<double> slice(int ind, int axis) const;
	void setSlice(int ind, int axis, const std::vector<double> &other_slice);
	void setZeroSlice(int ind, int axis);

	void print(bool padded = false) const;
	void saveToCSV(const char* filename) const;
	void save(const char* filename) const;

private:
	const int _pad;
	int _I, _J, _K;
	int _rI, _rJ, _rK; //real data size
	int _shape[3];
	std::vector<double> _data;
};