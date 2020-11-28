#pragma once

struct Config {
public:
	Config(const char* config_path);
	double L[3];
	double T;

private:
	void readConfig(const char *config_path);
};