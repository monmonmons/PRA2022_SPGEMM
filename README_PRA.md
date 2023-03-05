# 测试用例说明
测试算例使用DTK-22.04环境，请先卸载原有的rocm环境，切换环境 dtk-22.04
命令如下:
module unload compiler/dtk/21.10
module load apps/anaconda3/5.2.0
module load compiler/dtk/22.04

## 编译方式    
进入demo测试目录后，运行make命令进行编译
编译后的可执行程序为Csrsparse

### 程序测试与验证
主机端验证调用下面的函数
```c++
host_spgemm(...);
```
与参赛者实现的接口函数进行结果比对，验证程序正确性

### 运行方式
```shell
./Csrsparse 0  rows_A  cols_A  sparsity_A[0,1]  sparsity_B[0,1]  random_seed[1,-]

参数说明：
矩阵A随机生成时，需要提供8个运行参数：
第一个参数为矩阵A生成方式，0表示随机生成；
矩阵A的行数；
矩阵A的列数；
矩阵A的稀疏度，范围0~1；
矩阵B的稀疏度，范围0~1；
生成矩阵A、B的随机数种子，范围>=1。
```

### 数据文件测试
```shell
./Csrsparse 1  "sparse matrix A filename"  sparsity_B[0,1]  random_seed[1,-]

参数说明：
矩阵A文件读入时，需要提供6个运行参数：
第一个参数为矩阵A生成方式，1表示文件读入；
矩阵A所需数据文件存放路径；
矩阵B的稀疏度，范围0~1；
生成矩阵B的随机数种子，范围>=1。
```

### rocsparse实现的demo
demo代码中提供了一个使用rocsparse库接口实现的示例，用户可以使用命令：
make gpu=1
编译并测试，编译后的可执行程序为Csrsparse_rocsparse

注意：
1）rocsparse实现的示例仅供参考，参赛选手不可以调用rocsparse以及其他稀疏矩阵库函数接口。
2）参赛选手需要手动实现所有Kernel代码，并保持如下调用接口不变:
call_device_spgemm(transA, transB, alpha, m, n, k, 
		A_nnz, dptr_offset_A, dptr_colindex_A, dptr_value_A,
		B_nnz, dptr_offset_B, dptr_colindex_B, dptr_value_B,
		&C_nnz, dptr_offset_C, &pdptr_colindex_C, &pdptr_value_C);
