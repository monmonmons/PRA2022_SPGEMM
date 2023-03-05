#### 切换编译环境

```bash
# bash
module unload compiler/dtk/21.10
module load apps/anaconda3/5.2.0
module load compiler/dtk/22.04.2
```



#### 编译代码命令

```bash
cd $HOME$/demo-spgemm-20220802/
make clean && make gpu=1
```



#### 运行可执行文件命令（决赛赛题--15个算例）

```bash
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.0001 1
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.001 1
./Csrsparse_rocsparse 0 50000 50000 0.00007 0.002 1

./Csrsparse_rocsparse 1 data/af23560.csr 0.001 1
./Csrsparse_rocsparse 1 data/af23560.csr 0.01 1
./Csrsparse_rocsparse 1 data/af23560.csr 0.02 1

./Csrsparse_rocsparse 1 data/bayer10.csr 0.001 1
./Csrsparse_rocsparse 1 data/bayer10.csr 0.01 1
./Csrsparse_rocsparse 1 data/bayer10.csr 0.02 1

./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.001 1
./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.01 1
./Csrsparse_rocsparse 1 data/bcsstk18.csr 0.02 1

./Csrsparse_rocsparse 1 data/dw4096.csr 0.001 1
./Csrsparse_rocsparse 1 data/dw4096.csr 0.01 1
./Csrsparse_rocsparse 1 data/dw4096.csr 0.02 1
```



