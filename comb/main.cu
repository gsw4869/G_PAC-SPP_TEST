//3090
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
//#include <conio.h>
#include <time.h>
using namespace std;

#define M 128
#define N 64
#define MaxThreadPerBlock 1024
#define Blocknum 4096//1time:2^22  18min:2^38  1day:2^44-2^45
int array1[64];
int results[64] = {0}; //保存已经找到的结果前缀数组
int results_end = 0; //结果前缀数组有效数据下标
int64_t res=0;
int64_t num=0;
int64_t simutime=0;
clock_t start;
clock_t endtime;
int64_t* d_A, * d_B,* h_B;
int* d_C, * d_resultnum;
int* resultnum,* d_weight,*weight;
/**
 * 组合
 * @param deep 递归深度
 * @param n 数组最大长度
 * @param m 要查找的组合的长度
 */
 FILE* fp_16;
 FILE* fp_18;
 FILE* fp_20;
 FILE* fp_22;
 FILE* fp_test;

 void checkrepeat(FILE *f,int name,int mode)
{
    uint64_t num;
    int k=0;
    int sum=0;
    uint64_t results[1000000];
    fseek(f,0,0);
    for(uint64_t i=0;!feof(f);i++)
    {
        k=0;
        fscanf(f, "%ld", &num);
        // for(uint64_t t=0;t<sum;t++)
        // {
        //     if(num==results[t])
        //     {
        //         k=1;
        //             //cout<<num<<endl;
        //         break;
        //     }
        // }
        if(k==0)
        {
            results[sum]=num;
            sum=sum+1;
            //cout<<sum<<endl;
        }
    }
    fclose(f);
    if(mode==1)
    {
        if(name==16)
        {
            f=fopen("16_1.txt","w+");
        }
        else if(name==18)
        {
            f=fopen("18_1.txt","w+");
        }
        else if(name==20)
        {
            f=fopen("20_1.txt","w+");
        }
        else if(name==22)
        {
            f=fopen("22_1.txt","w+");
        }
        for(uint64_t i=0;i<sum;i++)
        {
            fprintf(f,"%ld ",results[i]);
        }
        fprintf(f,"\n\n");
        fclose(f);
        printf("saved\n");
    }
    if(name==16)
    {
        f=fopen("16.txt","a+");
    }
    else if(name==18)
    {
        f=fopen("18.txt","a+");
    }
    else if(name==20)
    {
        f=fopen("20.txt","a+");
    }
    else if(name==22)
    {
        f=fopen("22.txt","a+");
    }
    // for(uint64_t i=0;i<sum;i++)
    // {
    //     fprintf(f,"%ld ",results[i]);
    // }
    // fprintf(f,"\n\n");
    //fclose(f);
    //return sum;
}

__device__ int64_t hamming(int64_t n) {
    n = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555);
    n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333);
    n = (n & 0x0f0f0f0f0f0f0f0f) + ((n >> 4) & 0x0f0f0f0f0f0f0f0f);
    n = (n & 0x00ff00ff00ff00ff) + ((n >> 8) & 0x00ff00ff00ff00ff);
    n = (n & 0x0000ffff0000ffff) + ((n >> 16) & 0x0000ffff0000ffff);
    n = (n & 0x00000000ffffffff) + ((n >> 32) & 0x00000000ffffffff);
    return n;
}

__global__ void simu(int64_t* a, int64_t* b, int* c, int t,int* weight)//���ж������
{
    int offset;
    offset = threadIdx.x + blockIdx.x * blockDim.x;  
    __shared__ int k;
    k= offset / 128 + t * MaxThreadPerBlock * Blocknum / 128;   
    //c[k * (M + 1) + offset%128] = hamming(a[offset%128] & b[k])%2;  
    atomicAdd(&weight[k], hamming(a[offset%128] & b[k])%2);    
}
__global__ void countweight(int64_t* b,int* c,int* result1,int64_t num)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    if(k<num)
    {
         if (c[k] < 23)
         {
            atomicAdd(&result1[c[k]],1);
         }
    }
}
__global__ void clearzero(int* c)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    c[k] = 0;
}
void comb(int deep, int n, int m,int64_t* b)
 {
     if (deep > n) // 越界递归结束
         return;
     if (results_end == m) // 找到结果，打印，递归结束
     {
         res=0;
         for (int i = 0; i < m; i++)
         {
             
             res=res|((int64_t)1<<results[i]);
             
         }
         b[num]=res;
         num++;
         if(num==MaxThreadPerBlock*Blocknum)
         {
            cudaMemcpy(d_B, b, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyHostToDevice);
            for (int t = 0; t < 128; t++)
            {
                simu << <Blocknum * MaxThreadPerBlock / 128, 128 >> > (d_A, d_B, d_C, t,d_weight);
            }
            countweight << < Blocknum, MaxThreadPerBlock >> > (d_B,d_weight,d_resultnum,num);
            num=0;
            simutime++;
            cudaMemcpy(weight, d_weight, Blocknum * MaxThreadPerBlock * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyDeviceToHost);
            for(int i=0;i < MaxThreadPerBlock * Blocknum;i++)
            {
                
                    if(weight[i]==16)
                    {
                        fprintf(fp_16,"%ld ",h_B[i]);
                    }
                    else if(weight[i]==18)
                    {
                        fprintf(fp_18,"%ld ",h_B[i]);
                    }
                    else if(weight[i]==20)
                    {
                        fprintf(fp_20,"%ld ",h_B[i]);
                    }
                    else if(weight[i]==22)
                    {
                        fprintf(fp_22,"%ld ",h_B[i]);
                    }
            }
            clearzero << < Blocknum, MaxThreadPerBlock >> > (d_weight);
            if(simutime%1024==0)
            {
                if(simutime%4096==0)
                {
                    checkrepeat(fp_16,16,1);
                    checkrepeat(fp_18,18,1);
                    checkrepeat(fp_20,20,1);
                    checkrepeat(fp_22,22,1);
                }
                endtime = clock();
                printf("\ntime=%f min\n", (double)(endtime - start) / CLOCKS_PER_SEC/60);
           
                cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
                fprintf(fp_test,"simutimes = %ld\n",simutime);
                for (int i = 0; i < 23; i++)
                {
                    cout << resultnum[i] << "  ";
                    fprintf(fp_test,"%d ",resultnum[i]);
                }
                cout << endl;
                cout<<"i = "<<m<<endl;
                cout << simutime << endl;
            }
         }
         return;
     }
     results[results_end++] = array1[deep];
     comb(deep+1, n, m,b); //向下一级递归
     results_end--;
     comb(deep+1, n, m,b); //向下一级递归
 }
int main()
{
    cudaError_t cudaStatus;
    cudaDeviceProp prop;
    int Num_Device;

    // ����ϵͳ�е�GPU����,��ָ��������һ��,ͬʱ�õ���GPU�����ܲ���
    cudaStatus = cudaGetDeviceCount(&Num_Device);
    if (cudaStatus != cudaSuccess)	// û��һ��������ڼ����GPU,���������в����޷�����
    {
        printf("There is no GPU beyond 1.0, exit!\n");
        exit(0);
    }
    else
    {
        cudaStatus = cudaGetDeviceProperties(&prop, Num_Device - 1);	// ѡ�����һ��GPU���ڼ���,ͬʱ����������ܲ���
        if (cudaStatus != cudaSuccess)	// û��һ��������ڼ����GPU,���������в����޷�����
        {
            printf("Cannot get device properties, exit!\n");
            exit(0);
        }
    }
    printf("Device Name : %s.\n", prop.name);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
    printf("maxThreadsPerMultiProcessor : %d.\n", prop.maxThreadsPerMultiProcessor);
    printf("Blocknum : %d.\n", Blocknum);
    int* h_AT = (int*)malloc(sizeof(int) * M * N);
    int64_t* h_A = (int64_t*)malloc(sizeof(int64_t) * M);
    h_B = (int64_t*)malloc(sizeof(int64_t) * Blocknum * MaxThreadPerBlock);
    int* h_C = (int*)malloc(sizeof(int) * Blocknum * MaxThreadPerBlock * M);
    weight = (int*)malloc(sizeof(int) * Blocknum * MaxThreadPerBlock);
    resultnum = (int*)malloc(sizeof(int) * 23);
    for (int i = 0; i < 23; i++)
    {
        resultnum[i] = 0;
    }

    fp_test = fopen("G_SPP.txt", "r");
    int h_num = 0;
    for (int i = 0; i < M * N; i++)
    {

        fscanf(fp_test, "%d", &h_num);//ÿM��Ԫ�أ�ֻ��¼һ����1��λ��        
        h_AT[i] = h_num;
    }
    fclose(fp_test);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i] = 2 * h_A[i] + h_AT[(63-j)*M + i];
        }
    }

    for(int k=0;k<64;k++)
    {
        array1[k]=k;
    }
    cudaMalloc((void**)&d_A, sizeof(int64_t) * M);
    cudaMalloc((void**)&d_B, sizeof(int64_t) * Blocknum * MaxThreadPerBlock);
    cudaMalloc((void**)&d_C, sizeof(int) * Blocknum * MaxThreadPerBlock * M);
    cudaMalloc((void**)&d_resultnum, sizeof(int) * 23);
    cudaMalloc((void**)&d_weight, sizeof(int) * Blocknum * MaxThreadPerBlock);

    cudaMemcpy(d_A, h_A, M * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultnum, resultnum, 23 * sizeof(int), cudaMemcpyHostToDevice);
    start = clock();
    fp_16=fopen("16.txt","w+");
    fp_18=fopen("18.txt","w+");
    fp_20=fopen("20.txt","w+");
    fp_22=fopen("22.txt","w+");
    fp_test=fopen("results.txt","w");
    fprintf(fp_16,"\n\n");
    fprintf(fp_18,"\n\n");
    fprintf(fp_20,"\n\n");
    fprintf(fp_22,"\n\n");
    for(int i=1; i<=13; i++){
        comb(0, 64, i, h_B);
        cout<<"i= "<<i<<endl<<endl;
    }    
    if(num!=0)
    {
        cudaMemcpy(d_B, h_B, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyHostToDevice);
        for (int t = 0; t < 128; t++)
        {
            simu << <Blocknum * MaxThreadPerBlock / 128, 128 >> > (d_A, d_B, d_C, t,d_weight);
        }
        countweight << < Blocknum, MaxThreadPerBlock >> > (d_B,d_weight,d_resultnum,num);
        cudaMemcpy(weight, d_weight, Blocknum * MaxThreadPerBlock * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyDeviceToHost);
        for(int i=0;i < MaxThreadPerBlock * Blocknum;i++)
        {
            
                if(weight[i]==16)
                {
                    fprintf(fp_16,"%ld ",h_B[i]);
                }
                else if(weight[i]==18)
                {
                    fprintf(fp_18,"%ld ",h_B[i]);
                }
                else if(weight[i]==20)
                {
                    fprintf(fp_20,"%ld ",h_B[i]);
                }
                else if(weight[i]==22)
                {
                    fprintf(fp_22,"%ld ",h_B[i]);
                }
        }
        clearzero << < Blocknum, MaxThreadPerBlock >> > (d_weight);

    }
    endtime = clock();
    printf("\ntime=%f min", (double)(endtime - start) / CLOCKS_PER_SEC/60);
    printf("\nsimutime=%ld times", simutime);
    printf("\nextranum=%ld times", num); 
    cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
    cout<<endl;
    fprintf(fp_test,"simutime = %ld\n",simutime);
    fprintf(fp_test,"extranum = %ld\n",num);
    for (int i = 0; i < 23; i++)
    {
        cout << resultnum[i] << "  ";
        fprintf(fp_test,"%d ",resultnum[i]);
    }
    fprintf(fp_test,"\n");
    fclose(fp_16);
    fclose(fp_18);
    fclose(fp_20);
    fclose(fp_22);
    fclose(fp_test);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_AT);
    free(h_B);
    free(h_C);
    return 0;
}


