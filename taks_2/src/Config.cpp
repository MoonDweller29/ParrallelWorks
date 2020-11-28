#include "Config.h"
#include "INIReader.h"
#include <iostream>

Config::Config(const char* config_path) {
	readConfig(config_path);
}

void Config::readConfig(const char *config_path) {
	INIReader reader(config_path);

    if (reader.ParseError() != 0) {
        std::cerr << "Can't load "<< config_path << std::endl;
        return;
    }


    periodic[0] = reader.GetBoolean("Grid", "PeriodicX", false);
    periodic[1] = reader.GetBoolean("Grid", "PeriodicY", false);
    periodic[2] = reader.GetBoolean("Grid", "PeriodicZ", false);
    L[0] = reader.GetReal("Grid", "Lx", -1);
    L[1] = reader.GetReal("Grid", "Ly", -1);
    L[2] = reader.GetReal("Grid", "Lz", -1);
    N[0] = reader.GetInteger("Grid", "Nx",  -1);
    N[1] = reader.GetInteger("Grid", "Ny",  -1);
    N[2] = reader.GetInteger("Grid", "Nz",  -1);
    K = reader.GetInteger("Grid", "K",  -1);

    for (int i = 0; i < 3; ++i) {
    	h[i] = L[i]/N[i];
    }

    tau = h[0]*h[0]; //heuristics
}

void Config::print() const {
	std::cout << "CONFIG:\n" <<
		"Periodic: " << periodic[0] <<","<< periodic[1] <<","<< periodic[2] << std::endl<<
		"Lx  = "<< L[0] << std::endl<<
		"Ly  = "<< L[1] << std::endl<<
		"Lz  = "<< L[2] << std::endl<<
		"Nx  = "<< N[0] << std::endl<<
		"Ny  = "<< N[1] << std::endl<<
		"Nz  = "<< N[2] << std::endl<<
		"K   = "<< K    << std::endl<<
		"tau = "<< tau  << std::endl;
}
