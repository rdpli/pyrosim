#!/bin/bash
if [ -z "$1" ]
then
	echo "Number of runs not specified"
	exit 1
fi
echo "Using number of TEST runs: " $1

for x in `seq 1 $1`
do
	seed=${x}
	qsub -vARG_SEED=${seed} ~/scratch/rigid_bodies/pyrosim/evodevo/single_runner_devo.pbs
	echo "TEST run $x started with seed $seed"
done
