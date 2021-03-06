#!/bin/bash -x

for procNum in 64 128 256
do 
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_128.ini
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_128p.ini
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_256.ini
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_256p.ini
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_512.ini
	mpisubmit.bg -n $procNum -w 00:10:00 -m smp mainMPI -- configs/config_512p.ini
done