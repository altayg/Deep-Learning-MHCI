#!/bin/bash
#PBS -N snkmk_DLMHC
#PBS -l walltime=196:00:00
#PBS -q default
#PBS -o out
#PBS -e err
#PBS -l mem=2GB
#PBS -m abe
#PBS -M YourEmail@lji.org
#PBS -l nodes=1:ppn=1:avx

cd $PBS_O_WORKDIR

WORKDIR=./tmp/$PBS_JOBNAME
mkdir -p $WORKDIR/logs

snakemake --jobs 74 --latency-wait 90 --cluster-config cluster.json --rerun-incomplete \
--cluster "qsub -l {cluster.walltime} -l {cluster.cores} -l {cluster.memory} -m n -q peters -e $WORKDIR/logs/ -o $WORKDIR/logs/" \
--config tmp=$WORKDIR/
