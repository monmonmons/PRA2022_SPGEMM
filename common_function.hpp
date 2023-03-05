#include <vector>

using namespace std;

#ifdef gpu
#include <hipsparse.h>
#include <rocsparse.h>
#endif

#define HIP_CHECK(stat)                                                                        \
{                                                                                              \
    if(stat != hipSuccess)                                                                     \
    {                                                                                          \
        std::cerr << "Error: hip error " << stat <<" in line " << __LINE__ << std::endl;       \
        exit(-1);                                                                              \
    }                                                                                          \
}

#define ROCSPARSE_CHECK(stat)                                                                  \
{                                                                                              \
    if(stat != rocsparse_status_success)                                                       \
    {                                                                                          \
        std::cerr << "Error: rocsparse error " << stat <<" in line " << __LINE__ << std::endl; \
        exit(-1);                                                                              \
    }                                                                                          \
}

//=============================================================================

#define dtype double                    //Set calculation Accuracy to double
#define csrIdxType int                  //Set CSR index data type

size_t random_seed = 1;                 //Random seed to generate matrix B or both A and B

enum sparse_operation {operation_none=0, operation_transpose=1};
enum sparse_operation opA = operation_none;
enum sparse_operation opB = operation_none;

size_t A_num_rows = 0;                  //Rows of Matrix A
size_t A_num_cols = 0;                  //Columns of Matrix A
size_t B_num_rows = 0;                  //Rows of Matrix B
size_t B_num_cols = 0;                  //Columns of Matrix B

double sparsityA = 0;                   //sparsity for generating matrix A
double sparsityB = 0;                   //sparsity for generating matrix B

size_t A_nnz = 0;                       //length of csr value A
size_t B_nnz = 0;                       //length of csr value B

dtype alpha = 1.0;                      //Scalar Coefficient for Sparse Matrix AXB
dtype beta  = 0.0;                      //Scalar Coefficient for Sparse Matrix D

csrIdxType* host_offsetA    = NULL;     //csr offset host ptr of A
csrIdxType* host_colindexA  = NULL;     //csr column index host ptr of A
dtype*  host_valueA         = NULL;     //csr value host ptr of A

csrIdxType* host_offsetB    = NULL;     //csr offset host ptr of B
csrIdxType* host_colindexB  = NULL;     //csr column index host ptr of B
dtype*  host_valueB         = NULL;     //csr value host ptr of B

csrIdxType* device_offsetA;             //csr offset device ptr of A
csrIdxType* device_colindexA;           //csr column index device ptr of A
dtype*  device_valueA;                  //csr value device ptr of A

csrIdxType* device_offsetB;             //csr offset device ptr of B
csrIdxType* device_colindexB;           //csr column index device ptr of B
dtype*  device_valueB;                  //csr value device ptr of B

csrIdxType* host_offsetC = NULL;           //csr offset host ptr of C
csrIdxType* host_colindexC = NULL;         //csr column index host ptr of C
dtype*  host_valueC = NULL;                //csr value host ptr of C
csrIdxType* host_compare_offsetC = NULL;   //csr offset host ptr of C
csrIdxType* host_compare_colindexC = NULL; //csr column index host ptr of C
dtype*  host_compare_valueC = NULL;        //csr value host compare ptr of C
dtype*  host_compare_dnmatC = NULL;        //Value host ptr of  Result Dense Matrix C Calculated by rocsparse or cpu function

//-----------------------------------------------------------------------------

vector<dtype>  csr_data;
vector<csrIdxType> csr_indices;
vector<csrIdxType> csr_indptr;
vector<dtype>  dense_vector;

dtype* matA = NULL;
dtype* matB = NULL;
//=============================================================================
//Release Host and Device Memory
void ReleaseHostDeviceMemory()
{
    HIP_CHECK( hipFree(device_offsetA) )
    HIP_CHECK( hipFree(device_colindexA) )
    HIP_CHECK( hipFree(device_valueA) )

    HIP_CHECK( hipFree(device_offsetB) )
    HIP_CHECK( hipFree(device_colindexB) )
    HIP_CHECK( hipFree(device_valueB) )

    delete [] host_offsetA;
    delete [] host_colindexA;
    delete [] host_valueA;
    
    delete [] host_offsetB;
    delete [] host_colindexB;    
    delete [] host_valueB;

    if(host_compare_offsetC!=NULL)
    {
        delete [] host_compare_offsetC;
    }

    if(host_compare_colindexC!=NULL)
    {
        delete [] host_compare_colindexC;
    }

    if(host_compare_valueC!=NULL)
    {
        delete [] host_compare_valueC;
    }

    if(host_compare_dnmatC!=NULL)
    {
        delete [] host_compare_dnmatC;
    }
}

//=============================================================================

struct my_timer
{
    struct timeval start_time, end_time;
    double time_use;

    void start()
    {
        gettimeofday(&start_time, NULL);
    }

    void stop()
    {
        gettimeofday(&end_time, NULL);
        time_use = (end_time.tv_sec-start_time.tv_sec)*1.0e6 + end_time.tv_usec-start_time.tv_usec;
    }
};

//=============================================================================

template<typename T>
void init_csr_dense_vector(char *buf, vector<T> &vec)
{
    char* p;
    p = strtok(buf, " ");
    while (p != NULL)
    {   
        bool isInt = std::is_same<T, csrIdxType>::value;
        if (isInt==true)
        {
            vec.push_back(atoi(p));
        }
        else
        {
            vec.push_back(atof(p));
        }
        p = strtok(NULL, " ");
    }
}

//----------------------------------------------------------------------------

void read_file(char *path)
{
    FILE *fp = fopen(path, "r");
    char buf[1024*1024*1024*5];
    int line_num=0;

    while( fgets(buf, sizeof(buf), fp) != NULL )
    {
        if(line_num==1)
        {
            init_csr_dense_vector<dtype>(buf,csr_data);
        }
        if(line_num==2)
        {
            init_csr_dense_vector<csrIdxType>(buf,csr_indices);
        }
        if(line_num==3)
        {
            init_csr_dense_vector<csrIdxType>(buf,csr_indptr);
        }
        if(line_num==4)
        {
            init_csr_dense_vector<dtype>(buf,dense_vector);
        }
        line_num++; 
    }
    fclose(fp);
}

//-----------------------------------------------------------------------------

void write_file(char *path, dtype* mat, csrIdxType len)
{
    FILE *fp = fopen(path, "w");

    for(int i=0; i<len; i++)
    {
        fprintf(fp,"%lf\n", mat[i]);
    }

    fclose(fp);
}

//-----------------------------------------------------------------------------

void load_csr_vector_to_matrix( dtype*&mat, size_t rows, size_t cols)
{
    size_t j = 0;
    for(size_t i=0; i<rows; i++)
    {
        for(size_t off=csr_indptr[i]; off<csr_indptr[i+1]; off++)
        {
            j = csr_indices[off];
            mat[i*cols+j] = csr_data[off];
        }
    }
}

//=============================================================================

void showmatrix(size_t rows, size_t cols, dtype* mat, bool matbyrow)
{
    if(matbyrow)
    {
        for(size_t i=0; i<rows; i++)
        {
            for(size_t j=0; j<cols; j++)
            {
                cout << mat[i*cols+j] <<"  ";
            }
            cout << endl;
        }
    }
    else
    {
        for(size_t i=0; i<rows; i++)
        {
            for(size_t j=0; j<cols; j++)
            {
                cout << mat[j*rows+i] << "  ";
            }
            cout << endl;
        }
    }
}

//=============================================================================

void generate_sparse_matrix(dtype*& spmat, size_t rows, size_t cols, double sparsity)
{
    for(size_t i=0; i<rows; i++)
    {
        for(size_t j=0; j<cols; j++)
        {
            size_t x = rand() % 1000000;
            if( x < 1000000.0 * sparsity )
            {
                spmat[i*cols+j] = x/1000000.0 + 1.0;
            }
        }
    }
}

//-----------------------------------------------------------------------------

size_t matrix_to_csr(string  matName, int trans, size_t rows, size_t cols, dtype* spmat, csrIdxType*& offset, csrIdxType*& colindex, dtype*& value)
{
    size_t nonzeros = 0;
 
    for(size_t i=0; i<rows; i++)
    {
        for(size_t j=0; j<cols; j++)
        {
            if(spmat[i*cols+j]!=0)
            {
                nonzeros++;
            }
        }
    }

    cout << "non zero num of matrix "<< matName << " : " << nonzeros << endl;

    if(trans==0)
    {
        offset   = new csrIdxType[rows+1];
        colindex = new csrIdxType[nonzeros];
        value    = new dtype[nonzeros];

        size_t k = 0;
        size_t l = 0;

        for(size_t i=0; i<rows; i++)
        {
            for(size_t j=0; j<cols; j++)
            {
                if(j==0)
                {
                    offset[l++] = k;
                }
                if(spmat[i*cols+j]!=0)
                {
                    colindex[k] = j;
                    value[k] = spmat[i*cols+j];
                    k++;
                }
            }
        }
        offset[l] = nonzeros;
    }
    else
    {
        offset   = new csrIdxType[cols+1];
        colindex = new csrIdxType[nonzeros];
        value    = new dtype[nonzeros];
    
        size_t k = 0;
        size_t l = 0;
    
        for(size_t j=0; j<cols; j++)
        {
            for(size_t i=0; i<rows; i++)
            {
                if(i==0)
                {
                    offset[l++] = k;
                }
                if(spmat[i*cols+j]!=0)
                {
                    colindex[k] = i;
                    value[k] = spmat[i*cols+j];
                    k++;
                }
            }
        }
        offset[l] = nonzeros;
    }
    return nonzeros;
}

//=============================================================================

void create_host_data(int FromFile)
{
    dtype* matA = NULL;
    dtype* matB = NULL;

    matA = new dtype[A_num_rows*A_num_cols];
    matB = new dtype[B_num_rows*B_num_cols];

    memset(matA, 0, A_num_rows*A_num_cols*sizeof(dtype));
    memset(matB, 0, B_num_rows*B_num_cols*sizeof(dtype));

    //Generate Sparse Matrix
    if(FromFile==0)
    {
        generate_sparse_matrix(matA, A_num_rows, A_num_cols, sparsityA);
    }
    else
    {
        load_csr_vector_to_matrix(matA, A_num_rows, A_num_cols);
    }

    generate_sparse_matrix(matB, B_num_rows, B_num_cols, sparsityB);
    cout << "generate_sparse_matrix finished!" << endl; 

    //showmatrix(A_num_rows, A_num_cols, matA, true);
    //showmatrix(B_num_rows, B_num_cols, matB, true);

    //Transe Sparse Matrix to csr format
    A_nnz = matrix_to_csr( "A", 0, A_num_rows, A_num_cols, matA, host_offsetA, host_colindexA, host_valueA );
    B_nnz = matrix_to_csr( "B", 0, B_num_rows, B_num_cols, matB, host_offsetB, host_colindexB, host_valueB );
    cout << "matrix_to_csr finished!" << endl;

    delete [] matA;
    delete [] matB;

    host_compare_dnmatC = new dtype[A_num_rows*B_num_cols];
    memset(host_compare_dnmatC, 0, A_num_rows*B_num_cols*sizeof(dtype));
}

//-----------------------------------------------------------------------------

void create_deivce_data()
{
    if(opA==0)
    {    
        HIP_CHECK( hipMalloc((void**) &device_offsetA, (A_num_rows + 1) * sizeof(csrIdxType)) )
    }
    else
    {
        HIP_CHECK( hipMalloc((void**) &device_offsetA, (A_num_cols + 1) * sizeof(csrIdxType)) )
    }

    HIP_CHECK( hipMalloc((void**) &device_colindexA, A_nnz * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) &device_valueA, A_nnz * sizeof(dtype)) )
    
    if(opB==0)
    { 
        HIP_CHECK( hipMalloc((void**) &device_offsetB, (B_num_rows + 1) * sizeof(csrIdxType)) )
    }
    else
    {
        HIP_CHECK( hipMalloc((void**) &device_offsetB, (B_num_cols + 1) * sizeof(csrIdxType)) )
    }

    HIP_CHECK( hipMalloc((void**) &device_colindexB, B_nnz * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) &device_valueB, B_nnz * sizeof(dtype)) )

    cout << "device memory malloc finished!" << endl;

    if(opA==0)
    {
        HIP_CHECK( hipMemcpy(device_offsetA, host_offsetA, (A_num_rows + 1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    }
    else
    {
        HIP_CHECK( hipMemcpy(device_offsetA, host_offsetA, (A_num_cols + 1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    }

    HIP_CHECK( hipMemcpy(device_colindexA, host_colindexA, A_nnz * sizeof(csrIdxType),  hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(device_valueA, host_valueA, A_nnz * sizeof(dtype), hipMemcpyHostToDevice) )

    if(opB==0)
    {
        HIP_CHECK( hipMemcpy(device_offsetB, host_offsetB, (B_num_rows + 1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    }
    else
    {
        HIP_CHECK( hipMemcpy(device_offsetB, host_offsetB, (B_num_cols + 1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    }

    HIP_CHECK( hipMemcpy(device_colindexB, host_colindexB, B_nnz * sizeof(csrIdxType),  hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(device_valueB, host_valueB, B_nnz * sizeof(dtype), hipMemcpyHostToDevice) )

    cout << "device data memcpy finished!" << endl;
}

//=============================================================================

void verify(dtype* mat_kernel, dtype* mat_compare, size_t rows, size_t cols)
{
    size_t total_validation=0;
    bool bEqual = true;

    for(size_t i=0; i<rows*cols; i++)
    {
        double fab_diff = fabs( mat_kernel[i] - mat_compare[i] );
        double fab_mat = fabs(mat_compare[i]);  

        if((fab_mat>1.0)&&(fab_diff>=fab_mat*1e-9))
        {
            bEqual = false;
        }
        else if((fab_mat<=1.0)&&(fab_diff>=1e-9))
        {
            bEqual = false;
        }

        if(!bEqual)
        {
            #ifdef gpu
                cout<<"["<<i/cols<<","<<i%cols<<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<mat_kernel[i]<<" rocsparse_value "<<mat_compare[i]<<endl;
            #else
                cout<<"["<<i/cols<<","<<i%cols<<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<mat_kernel[i]<<" host_value "<<mat_compare[i]<<endl;
            #endif

            cout<<"Failed verification,please check your code!"<<endl;
            return ;
        }
        total_validation = i;
    } 
 
    cout<<"Congratulation, pass "<<total_validation+1<< " validation!"<<endl;
}

//-----------------------------------------------------------------------------

void verify(size_t nnz_kernel, size_t nnz_compare)
{
    bool bEqual = true;
    
    for(size_t row=0; row<A_num_rows; row++)
    {
        size_t off1_l = host_offsetC[row];
        size_t off1_r = host_offsetC[row+1];
        size_t off2_l = host_compare_offsetC[row];
        size_t off2_r = host_compare_offsetC[row+1];

        size_t len1 = off1_r-off1_l;
        size_t len2 = off2_r-off2_l;
        size_t num_max = max(len1,len2);

        size_t num  = 0;
        size_t num1 = 0;
        size_t num2 = 0;

        size_t col  = 0;
        size_t col1 = 0;
        size_t col2 = 0;

        double value1 = 0.0;
        double value2 = 0.0;

        double fab_diff = 0.0;
        double fab_mat = 0.0;
    
        if((off1_r==off1_l) && (off2_r==off2_l))
        {
            continue;
        }
        else if((off1_r==off1_l) && (off2_r>off2_l))
        {
            while( num2<len2 )
            {
                col2 = host_compare_colindexC[off2_l+num2];
                value2 = host_compare_valueC[off2_l+num2];

                fab_diff = fabs( value2 );

                if(fab_diff>=1e-9)
                {
                    bEqual = false;
                }

                if(!bEqual)
                {
                    col = col2;
                    #ifdef gpu
                        cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<0.0<<" rocsparse_value "<<value2<<endl;
                    #else
                        cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<0.0<<" host_value "<<value2<<endl;
                    #endif

                    cout<<"off "<<off1_l<<" "<<off1_r<<" "<<off2_l<<" "<<off2_r<<endl;
                    cout<<"col "<<"empty"<<" "<<col2<<" index "<<"empty"<<" "<<num2<<endl;

                    cout<<"Failed verification, please check your code!"<<endl;
                    return;
                }
 
                num2++;
            }           
        }
        else if((off1_r>off1_l) && (off2_r==off2_l))
        {
            while( num1<len1 )
            {
                col1 = host_colindexC[off1_l+num1];
                value1 = host_valueC[off1_l+num1];

                fab_diff = fabs( value1 );

                if(fab_diff>=1e-9)
                {
                    bEqual = false;
                }

                if(!bEqual)
                {
                    col = col1;
                    #ifdef gpu
                        cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" rocsparse_value "<<0.0<<endl;
                    #else
                        cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" host_value "<<0.0<<endl;
                    #endif

                    cout<<"off "<<off1_l<<" "<<off1_r<<" "<<off2_l<<" "<<off2_r<<endl;
                    cout<<"col "<<col1<<" "<<"empty"<<" index "<<num1<<" "<<"empty"<<endl;

                    cout<<"Failed verification, please check your code!"<<endl;
                    return;
                }
  
                num1++;
            }
        }
        else
        {
            while( num<=num_max )
            {
                col1 = num1<len1 ? host_colindexC[off1_l+num1] : host_colindexC[off1_r-1];
                value1 = num1<len1 ? host_valueC[off1_l+num1] : 0.0;
                    
                col2 = num2<len2 ? host_compare_colindexC[off2_l+num2] : host_compare_colindexC[off2_r-1];
                value2 = num2<len2 ? host_compare_valueC[off2_l+num2] : 0.0;

                if(col1==col2)
                {
                    fab_diff = fabs( value1 - value2);
                    fab_mat  = fabs( value2 );

                    if((fab_mat>1.0)&&(fab_diff>=fab_mat*1e-9))
                    {
                        bEqual = false;
                    }
                    else if((fab_mat<=1.0)&&(fab_diff>=1e-9))
                    {
                        bEqual = false;
                    }

                    if(!bEqual)
                    {
                        col = col1;
                        #ifdef gpu
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" rocsparse_value "<<value2<<endl;
                        #else
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" host_value "<<value2<<endl;
                        #endif

                        cout<<"off "<<off1_l<<" "<<off1_r<<" "<<off2_l<<" "<<off2_r<<endl;
                        cout<<"col "<<col1<<" "<<col2<<" index "<<num1<<" "<<num2<<endl;
 
                        cout<<"Failed verification, please check your code!"<<endl;
                        return;
                    }

                    num1++;
                    num2++;
                }
                else if (col1<col2)
                {
                    fab_diff = num1<len1 ? fabs( value1 ) : fabs( value2 );

                    if(fab_diff>=1e-9)
                    {
                        bEqual = false;
                    }

                    if(!bEqual)
                    {
                        col = num1<len1 ? col1 : col2;
                        #ifdef gpu
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" rocsparse_value "<<value2<<endl;
                        #else
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" host_value "<<value2<<endl;
                        #endif
   
                        cout<<"off "<<off1_l<<" "<<off1_r<<" "<<off2_l<<" "<<off2_r<<endl;
                        cout<<"col "<<col1<<" "<<col2<<" index "<<num1<<" "<<num2<<endl;
  
                        cout<<"Failed verification, please check your code!"<<endl;
                        return;
                    }

                    if(num1<len1)
                    {
                        num1++;
                    }
                    else
                    {
                        num2++;
                    }
                }
                else if (col1>col2)
                {
                    fab_diff = num2<len2 ? fabs( value2 ) : fabs( value1 );
   
                    if(fab_diff>=1e-9)
                    {
                        bEqual = false;
                    }

                    if(!bEqual)
                    {
                        col = num2<len2 ? col2 : col1;
                        #ifdef gpu
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" rocsparse_value "<<value2<<endl;
                        #else
                            cout<<"["<< row <<","<< col <<"] : "<<"fab_diff "<<fab_diff<<" device_value "<<value1<<" host_value "<<value2<<endl;
                        #endif

                        cout<<"off "<<off1_l<<" "<<off1_r<<" "<<off2_l<<" "<<off2_r<<endl;
                        cout<<"col "<<col1<<" "<<col2<<" index "<<num1<<" "<<num2<<endl;

                        cout<<"Failed verification, please check your code!"<<endl;
                        return;
                    }

                    if(num2<len2)
                    {
                        num2++;
                    }
                    else
                    {
                        num1++;
                    }
                }
                num++;
            }
        }
    }

    cout<<"Congratulations, verification pass! "<<endl;
}


size_t host_spgemm()
{
    size_t rowA = 0;
    size_t colA = 0;
    size_t rowB = 0;
    size_t colB = 0;

    if(opA==0 && opB==0)
    {
        for(size_t rowA=0; rowA<A_num_rows; rowA++)
        {
            for(size_t offA=host_offsetA[rowA]; offA<host_offsetA[rowA+1]; offA++)
            {
                colA = host_colindexA[offA];
                for(size_t offB=host_offsetB[colA]; offB<host_offsetB[colA+1]; offB++)
                {
                    colB = host_colindexB[offB];
                    host_compare_dnmatC[rowA*B_num_cols+colB] += alpha * host_valueA[offA] * host_valueB[offB];
                }
            }
        }
    }
    else if(opA==0 && opB==1)
    {
         for(size_t rowA=0; rowA<A_num_rows; rowA++)
         {
             for(size_t offA=host_offsetA[rowA]; offA<host_offsetA[rowA+1]; offA++)
             {
                 colA = host_colindexA[offA];
                 for(size_t colB=0; colB<B_num_cols; colB++)
                 {
                     for(size_t offB=host_offsetB[colB]; offB<host_offsetB[colB+1]; offB++)
                     {
                         rowB = host_colindexB[offB];
                         if(rowB==colA)
                         {
                             host_compare_dnmatC[rowA*B_num_cols+colB] += alpha * host_valueA[offA] * host_valueB[offB];
                         }
                     }
                 }
             }
         }
     }
     else if(opA==1 && opB==0)
     {
         for(size_t colA=0; colA<A_num_cols; colA++)
         {
             for(size_t offA=host_offsetA[colA]; offA<host_offsetA[colA+1]; offA++)

             {
                 rowA = host_colindexA[offA];
                 for(size_t offB=host_offsetB[colA]; offB<host_offsetB[colA+1]; offB++)
                 {
                     colB = host_colindexB[offB];
                     host_compare_dnmatC[rowA*B_num_cols+colB] += alpha * host_valueA[offA] * host_valueB[offB];
                 }
             }
         }
    }
    else if(opA==1 && opB==1)
    {
        for(size_t colA=0; colA<A_num_cols; colA++)
        {
            for(size_t offA=host_offsetA[colA]; offA<host_offsetA[colA+1]; offA++)
            {
                rowA = host_colindexA[offA];

                for(size_t colB=0; colB<B_num_cols; colB++)
                {
                    for(size_t offB=host_offsetB[colB]; offB<host_offsetB[colB+1]; offB++)
                    {
                        rowB = host_colindexB[offB];
                        if(rowB==colA)
                        {
                            host_compare_dnmatC[rowA*B_num_cols+colB] += alpha * host_valueA[offA] * host_valueB[offB];
                        }
                    }
                }
            }
        }
    }

    size_t nonzero = matrix_to_csr( "C", 0, A_num_rows, A_num_cols, host_compare_dnmatC, host_compare_offsetC, host_compare_colindexC, host_compare_valueC );

    return nonzero;
}

//=============================================================================

#ifdef gpu
size_t rocsparse_spgemm()
{
    rocsparse_handle handle = NULL;
    ROCSPARSE_CHECK( rocsparse_create_handle(&handle) );
    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);

    rocsparse_operation operationA = rocsparse_operation_none;
    rocsparse_operation operationB = rocsparse_operation_none;

    rocsparse_mat_descr descr_A;
    rocsparse_mat_descr descr_B;
    rocsparse_mat_descr descr_C;
    rocsparse_mat_descr descr_D;

    rocsparse_create_mat_descr(&descr_A);
    rocsparse_create_mat_descr(&descr_B);
    rocsparse_create_mat_descr(&descr_C);
    rocsparse_create_mat_descr(&descr_D); 

    rocsparse_mat_info info_C;
    rocsparse_create_mat_info(&info_C);

    rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);

    csrIdxType D_nnz = 1;

    csrIdxType* host_offsetD = new csrIdxType[A_num_rows+1];
    memset(host_offsetD, 0, (A_num_rows+1) * sizeof(csrIdxType));
    host_offsetD[A_num_rows] = 1;
 
    csrIdxType host_colindexD = A_num_rows-1;
    dtype host_valueD = 0.0;

    csrIdxType* device_offsetD;
    csrIdxType* device_colindexD;
    dtype* device_valueD;

    HIP_CHECK( hipMalloc((void**) &device_offsetD, (A_num_rows+1) * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) &device_colindexD, D_nnz * sizeof(csrIdxType)) )
    HIP_CHECK( hipMalloc((void**) &device_valueD, D_nnz * sizeof(dtype)) )
    
    HIP_CHECK( hipMemcpy(device_offsetD, host_offsetD, (A_num_rows+1) * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(device_colindexD, &host_colindexD, D_nnz * sizeof(csrIdxType), hipMemcpyHostToDevice) )
    HIP_CHECK( hipMemcpy(device_valueD, &host_valueD, D_nnz * sizeof(dtype), hipMemcpyHostToDevice) )

    delete [] host_offsetD;

    #if dtype==double
    size_t buffer_size;
    ROCSPARSE_CHECK( rocsparse_dcsrgemm_buffer_size(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               &alpha, descr_A, A_nnz, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_offsetB, device_colindexB,
                               &beta, descr_D, D_nnz, device_offsetD, device_colindexD,
                               info_C, &buffer_size) )
    void* buffer;
    HIP_CHECK( hipMalloc(&buffer, buffer_size) )

    csrIdxType nonzero = 0;
    csrIdxType* csr_row_ptr_C;
    HIP_CHECK( hipMalloc((void**)&csr_row_ptr_C, sizeof(csrIdxType)*(A_num_rows+1)) )

    ROCSPARSE_CHECK( rocsparse_csrgemm_nnz(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               descr_A, A_nnz, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_offsetB, device_colindexB,
                               descr_D, D_nnz, device_offsetD, device_colindexD,
                               descr_C, csr_row_ptr_C, &nonzero, info_C, buffer) )

    csrIdxType* csr_col_ind_C;
    dtype* csr_val_C;
    HIP_CHECK( hipMalloc((void**)&csr_col_ind_C, sizeof(csrIdxType) * nonzero) )
    HIP_CHECK( hipMalloc((void**)&csr_val_C, sizeof(dtype) * nonzero) )

    ROCSPARSE_CHECK( rocsparse_dcsrgemm(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               &alpha, descr_A, A_nnz, device_valueA, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_valueB, device_offsetB, device_colindexB,
                               &beta, descr_D, D_nnz, device_valueD, device_offsetD, device_colindexD,
                               descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, buffer) )


    host_compare_offsetC = new csrIdxType[A_num_rows+1];
    memset(host_compare_offsetC, 0, (A_num_rows+1)*sizeof(csrIdxType));

    host_compare_colindexC = new csrIdxType[nonzero];
    memset(host_compare_colindexC, 0, nonzero*sizeof(csrIdxType));

    host_compare_valueC = new dtype[nonzero];
    memset(host_compare_valueC, 0, nonzero*sizeof(dtype));
    
    HIP_CHECK( hipMemcpy(host_compare_offsetC, csr_row_ptr_C, (A_num_rows+1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
    HIP_CHECK( hipMemcpy(host_compare_colindexC, csr_col_ind_C, nonzero * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
    HIP_CHECK( hipMemcpy(host_compare_valueC, csr_val_C, nonzero * sizeof(dtype), hipMemcpyDeviceToHost) )

    HIP_CHECK( hipFree(buffer) )
    HIP_CHECK( hipFree(csr_row_ptr_C) )
    HIP_CHECK( hipFree(csr_col_ind_C) )
    HIP_CHECK( hipFree(csr_val_C) )

    #elif dtype==float
    size_t buffer_size;
    ROCSPARSE_CHECK( rocsparse_scsrgemm_buffer_size(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               &alpha, descr_A, A_nnz, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_offsetB, device_colindexB,
                               &beta, descr_D, D_nnz, device_offsetD, device_colindexD,
                               info_C, &buffer_size) )
    void* buffer;
    HIP_CHECK( hipMalloc(&buffer, buffer_size) )

    csrIdxType nonzero = 0;
    csrIdxType* csr_row_ptr_C;
    HIP_CHECK( hipMalloc((void**)&csr_row_ptr_C, sizeof(csrIdxType)*(A_num_rows+1)) )

    ROCSPARSE_CHECK( rocsparse_csrgemm_nnz(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               descr_A, A_nnz, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_offsetB, device_colindexB,
                               descr_D, D_nnz, device_offsetD, device_colindexD,
                               descr_C, csr_row_ptr_C, &nonzero, info_C, buffer) )

    csrIdxType* csr_col_ind_C;
    dtype* csr_val_C;
    HIP_CHECK( hipMalloc((void**)&csr_col_ind_C, sizeof(csrIdxType) * nonzero) )
    HIP_CHECK( hipMalloc((void**)&csr_val_C, sizeof(dtype) * nonzero) )

    ROCSPARSE_CHECK( rocsparse_scsrgemm(handle,operationA, operationB, A_num_rows, B_num_cols, A_num_cols,
                               &alpha, descr_A, A_nnz, device_valueA, device_offsetA, device_colindexA,
                               descr_B, B_nnz, device_valueB, device_offsetB, device_colindexB,
                               &beta, descr_D, D_nnz, device_valueD, device_offsetD, device_colindexD,
                               descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, buffer) )


    host_compare_offsetC = new csrIdxType[A_num_rows+1];
    memset(host_compare_offsetC, 0, (A_num_rows+1)*sizeof(csrIdxType));

    host_compare_colindexC = new csrIdxType[nonzero];
    memset(host_compare_colindexC, 0, nonzero*sizeof(csrIdxType));

    host_compare_valueC = new dtype[nonzero];
    memset(host_compare_valueC, 0, nonzero*sizeof(dtype));

    HIP_CHECK( hipMemcpy(host_compare_offsetC, csr_row_ptr_C, (A_num_rows+1) * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
    HIP_CHECK( hipMemcpy(host_compare_colindexC, csr_col_ind_C, nonzero * sizeof(csrIdxType), hipMemcpyDeviceToHost) )
    HIP_CHECK( hipMemcpy(host_compare_valueC, csr_val_C, nonzero * sizeof(dtype), hipMemcpyDeviceToHost) )


    HIP_CHECK( hipFree(buffer) )
    HIP_CHECK( hipFree(csr_row_ptr_C) )
    HIP_CHECK( hipFree(csr_col_ind_C) )
    HIP_CHECK( hipFree(csr_val_C) )

    #endif

    HIP_CHECK( hipFree(device_valueD) )
    HIP_CHECK( hipFree(device_colindexD) )
    HIP_CHECK( hipFree(device_offsetD) )

    cout << "non zero num of matrix C : " << nonzero << endl;
    return nonzero;
}

#endif
//=============================================================================
