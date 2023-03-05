#!/bin/bash
#SBATCH -J ALL_TEST
#SBATCH -e %j.e
#SBATCH -p wzhdnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=90G
#SBATCH --exclusive

#module list

module unload compiler/dtk/21.10
module load apps/anaconda3/5.2.0
module load compiler/dtk/22.04.2

module list

# cd /work/home/xd_fm_011/demo-spgemm-20220802-spgemm/
make clean && make gpu=1 

echo "--------- TEST 0.0001 --------------"
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.0001 1
echo "--------- TEST 0.001  --------------"
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.001 1
echo "--------- TEST 0.002  --------------"
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.002 1


echo "--------- af23560 0.001 --------------"
./Csrsparse_rocsparse 1 data/af23560.csr 0.001 1
echo "--------- af23560 0.01  --------------"
./Csrsparse_rocsparse 1 data/af23560.csr 0.01 1
echo "--------- af23560 0.02  --------------"
./Csrsparse_rocsparse 1 data/af23560.csr 0.02 1
echo "--------- bayer10 0.001 --------------"
./Csrsparse_rocsparse 1 data/bayer10.csr 0.001 1
echo "--------- bayer10 0.01  --------------"
./Csrsparse_rocsparse 1 data/bayer10.csr 0.01 1
echo "--------- bayer10 0.02  --------------"
./Csrsparse_rocsparse 1 data/bayer10.csr 0.02 1


echo "--------- bcsstk18 0.001 --------------"
./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.001 1
echo "--------- bcsstk18 0.01  --------------"
./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.01 1
echo "--------- bcsstk18 0.02  --------------"
./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.02 1

echo "--------- dw4096 0.001 --------------"
./Csrsparse_rocsparse 1 data/dw4096.csr 0.001 1
echo "--------- dw4096 0.01  --------------"
./Csrsparse_rocsparse 1 data/dw4096.csr 0.01 1
echo "--------- dw4096 0.02  --------------"
./Csrsparse_rocsparse 1 data/dw4096.csr 0.02 1
echo "-----------------------------------------"
