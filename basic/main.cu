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

__device__ int64_t hamming(int64_t n) {
    n = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555);
    n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333);
    n = (n & 0x0f0f0f0f0f0f0f0f) + ((n >> 4) & 0x0f0f0f0f0f0f0f0f);
    n = (n & 0x00ff00ff00ff00ff) + ((n >> 8) & 0x00ff00ff00ff00ff);
    n = (n & 0x0000ffff0000ffff) + ((n >> 16) & 0x0000ffff0000ffff);
    n = (n & 0x00000000ffffffff) + ((n >> 32) & 0x00000000ffffffff);
    return n;
}

__global__ void simu(int64_t* a, int64_t* b, int* c, int t)//���ж������
{
    int offset;
    offset = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int k;
    k= offset / 128 + t * MaxThreadPerBlock * Blocknum / 128;
    c[k * (M + 1) + offset%128] = hamming(a[offset%128] & b[k])%2;  
    atomicAdd(&c[k * (M + 1) + M], c[k * (M + 1) + offset%128]); 
}
__global__ void countweight(int64_t* b,int* c,int* result)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    if (c[k * (M + 1) + M] < 23)
    {
        atomicAdd(&result[c[k * (M + 1) + M]],1);
    }
    b[k] += MaxThreadPerBlock * Blocknum;
    //c[k * (M + 1) + M] = 0;
}

int main()
{
    cudaError_t cudaStatus;
    cudaDeviceProp prop;
    FILE* fp_test;
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
    int64_t* h_B = (int64_t*)malloc(sizeof(int64_t) * Blocknum * MaxThreadPerBlock);
    int* h_C = (int*)malloc(sizeof(int) * Blocknum * MaxThreadPerBlock * (M + 1));
    int* resultnum = (int*)malloc(sizeof(int) * 23);
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
            h_A[i] = 2 * h_A[i] + h_AT[(63-j) * M + i];
        }
    }
    for (int i = 0; i < Blocknum * MaxThreadPerBlock; i++)
    {
        h_B[i] = -361976000113803264+i;   
    }

    int64_t* d_A, * d_B;
    int* d_C, * d_resultnum;
    cudaMalloc((void**)&d_A, sizeof(int64_t) * M);
    cudaMalloc((void**)&d_B, sizeof(int64_t) * Blocknum * MaxThreadPerBlock);
    cudaMalloc((void**)&d_C, sizeof(int) * Blocknum * MaxThreadPerBlock * (M + 1));
    cudaMalloc((void**)&d_resultnum, sizeof(int) * 23);

    cudaMemcpy(d_A, h_A, M * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultnum, resultnum, 23 * sizeof(int), cudaMemcpyHostToDevice);
    int64_t k = 0;
    clock_t start = clock();
    clock_t end;
    while (k < 1)//��
    {
        for (int i = 0; i < 128; i++)
        {
            simu << <Blocknum * MaxThreadPerBlock / 128, 128 >> > (d_A, d_B, d_C, i);
        }
        //cudaThreadSynchronize();
        countweight << < Blocknum, MaxThreadPerBlock >> > (d_B,d_C,d_resultnum);
        if (k%4096==0)
        {
            end = clock();
            printf("\ntime=%f min\n", (double)(end - start) / CLOCKS_PER_SEC/60);
           
            cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = 0; i < 23; i++)
            {
                cout << resultnum[i] << "  ";
            }
            cout << endl;
            cout << k << endl;
        }
        k++;
        //cudaThreadSynchronize();

    }
    end = clock();
    printf("\ntime=%f min\n", (double)(end - start) / CLOCKS_PER_SEC/60);

    cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, Blocknum * MaxThreadPerBlock * (M + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<129;i++)
    {
        cout<<h_C[i]<<" ";
    }
    cout<<endl;
    cout << h_B[Blocknum * MaxThreadPerBlock - 1] << endl;
    cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 23; i++)
    {
        cout << resultnum[i] << "  ";
    }
    cout << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_AT);
    free(h_B);
    free(h_C);
    return 0;
}