module load SpectrumMPI
# module load OpenMPI

for N in 1 2 3 4 ; do
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_128.txt  mpirun -n ${N} ./main.out configs/config_128.ini
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_128p.txt mpirun -n ${N} ./main.out configs/config_128p.ini
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_256.txt  mpirun -n ${N} ./main.out configs/config_256.ini
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_256p.txt mpirun -n ${N} ./main.out configs/config_256p.ini
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_512.txt  mpirun -n ${N} ./main.out configs/config_512.ini
	bsub -q normal -W 00:30 -x -R "span[ptile=1]" -n $N -oo mpi_out/MPI_out_${N}_512p.txt mpirun -n ${N} ./main.out configs/config_512p.ini


done
