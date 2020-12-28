module load SpectrumMPI
# module load OpenMPI

for N in 1 2 3 4 ; do
	for THR in 8; do
		export OMP_NUM_THREADS=${THR}
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_128.txt  mpirun -n ${N} ./main_openMP.out configs/config_128.ini
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_128p.txt mpirun -n ${N} ./main_openMP.out configs/config_128p.ini
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_256.txt  mpirun -n ${N} ./main_openMP.out configs/config_256.ini
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_256p.txt mpirun -n ${N} ./main_openMP.out configs/config_256p.ini
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_512.txt  mpirun -n ${N} ./main_openMP.out configs/config_512.ini
		bsub -q normal -W 00:15 -x -R "span[ptile=1]" -n $N -oo omp_out/openMP_out_${N}_${THR}_512p.txt mpirun -n ${N} ./main_openMP.out configs/config_512p.ini


		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_128.ini
		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_128p.ini
		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_256.ini
		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_256p.ini
		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_512.ini
		# mpisubmit.pl -p $procNum -w 00:15 main.out -- configs/config_512p.ini
	done
done
