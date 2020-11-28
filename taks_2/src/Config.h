#pragma once

struct Config {
public:
	double L[3];
	int N[3];
	double h[3];
	bool periodic[3];
	int K;
	double tau; 

	Config(const char* config_path);
	void print() const;

private:
	void readConfig(const char *config_path);
};