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

    L[0] = reader.GetReal("Grid", "Lx", -1);
    L[1] = reader.GetReal("Grid", "Ly", -1);
    L[2] = reader.GetReal("Grid", "Lz", -1);
    T = reader.GetReal("Grid", "T",  -1);

    std::cout << "CONFIG:\n" <<
		"Lx = "<< L[0] << std::endl<<
		"Ly = "<< L[1] << std::endl<<
		"Lz = "<< L[2] << std::endl<<
		"T  = "<< T    << std::endl;
}