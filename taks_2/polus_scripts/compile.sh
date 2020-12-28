module load SpectrumMPI
module load OpenMPI

# mpi
mpixlC -Wall -std=c++98 -O3 \
	src/main.cpp \
	src/Mat3D.cpp \
	src/Config.cpp \
	src/Solver.cpp \
	src/U4D.cpp \
	src/F3D_f4.cpp \
	 -o main.out

# opemMP
mpixlC -Wall -std=c++98 -qsmp=omp -O3 \
	src/main.cpp \
	src/Mat3D.cpp \
	src/Config.cpp \
	src/Solver.cpp \
	src/U4D.cpp \
	src/F3D_f4.cpp \
	 -o main_openMP.out