#!/bin/bash

while getopts d:e:n:m:a: flag
do
    case "${flag}" in
        d) directory=${OPTARG};;
        e) experiment_num=${OPTARG};;
        n) experiment_name=${OPTARG};;
        m) em_methods=${OPTARG};;
        a) ablation_methods=${OPTARG};;
    esac
done
: "${experiment_num:=0}"
: "${experiment_name:="default"}"
: "${em_methods:="true"}"
: "${ablation_methods:="true"}"


experiment_types=("standard" "anisotropy" "outlier" "offset")
experiment=${experiment_types[${experiment_num}]}
# 0 omnidirectional; 1 pinhole
camera_nums=(0 1)
# 0 isotropic homogeneous; 1 isotropic inhomogeneous; 2 anisotropic homogenous; 3 anisotropic inhomogeneous
noise_nums=(0 1 2 3)
translation=("true" "false")

if [ $experiment_num == "0" ]
then
    linspace_start=0.125
    linspace_end=4.0
    linspace_num=32
fi
if [ $experiment_num == "1" ]
then
    linspace_start=0.5
    linspace_end=0.975
    linspace_num=20
fi
if [ $experiment_num == "2" ]
then
    linspace_start=0.0
    linspace_end=10.0
    linspace_num=21
fi
if [ $experiment_num == "3" ]
then
    linspace_start=0.0
    linspace_end=0.5
    linspace_num=21
fi

for camera_num in "${camera_nums[@]}"
do
    for t in "${translation[@]}"
    do
        if [ $experiment_num != "1" ]
        then
            for noise_num in "${noise_nums[@]}"
            do
                ./build/run_simulation ${directory} ${experiment_num} ${camera_num} ${t} ${noise_num} ${experiment_name} -e=${em_methods} -a=${ablation_methods} -sc=1.0 -ls=${linspace_start} -le=${linspace_end} -ln=${linspace_num} &
            done
        else
            ./build/run_simulation ${directory} ${experiment_num} ${camera_num} ${t} ${noise_num} ${experiment_name} -e=${em_methods} -a=${ablation_methods} -sc=1.0 -ls=${linspace_start} -le=${linspace_end} -ln=${linspace_num} &
        fi
        wait
    done
done
