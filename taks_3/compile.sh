nvcc -ccbin mpic++ -O0 \
	src/main.cu \
	src/Mat3D.cu \
	src/Config.cpp \
	src/Solver.cu \
	src/U4D.cpp \
	src/F3D_f4.cpp \
	src/cuda_utils/Stream.cu \
	src/cuda_utils/Event.cu \
	src/cuda_utils/CudaVec.cu \
	src/CudaSolver.cu \
	 -o main.out