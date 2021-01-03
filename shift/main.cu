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

uint64_t checkrepeat(FILE *f,int name,int mode)
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
        for(uint64_t t=0;t<sum;t++)
        {
            if(num==results[t])
            {
                k=1;
                    //cout<<num<<endl;
                break;
            }
        }
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
        f=fopen("16.txt","w+");
    }
    else if(name==18)
    {
        f=fopen("18.txt","w+");
    }
    else if(name==20)
    {
        f=fopen("20.txt","w+");
    }
    else if(name==22)
    {
        f=fopen("22.txt","w+");
    }
    for(uint64_t i=0;i<sum;i++)
    {
        fprintf(f,"%ld ",results[i]);
    }
    fprintf(f,"\n\n");
    //fclose(f);
    return sum;
}

__device__ uint64_t shiftr(uint64_t n,int t) {
    n =(n<<(64-t))|(n>>t);
    return n;
}
__device__ uint64_t shiftl(uint64_t n,int t) {
    n =(n>>(64-t))|(n<<t);
    return n;
}

__device__ uint64_t hamming(uint64_t n) {
    n = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555);
    n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333);
    n = (n & 0x0f0f0f0f0f0f0f0f) + ((n >> 4) & 0x0f0f0f0f0f0f0f0f);
    n = (n & 0x00ff00ff00ff00ff) + ((n >> 8) & 0x00ff00ff00ff00ff);
    n = (n & 0x0000ffff0000ffff) + ((n >> 16) & 0x0000ffff0000ffff);
    n = (n & 0x00000000ffffffff) + ((n >> 32) & 0x00000000ffffffff);
    return n;
}

__global__ void simu(uint64_t* a, uint64_t* b, int* c,int t,int* weight)//���ж������
{
    int offset;
    offset = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int k;
    k= offset / 128 + t * MaxThreadPerBlock * Blocknum / 128;
    //c[k * M + offset%128] = hamming(a[offset%128] & b[k])%2;  
    atomicAdd(&weight[k], hamming(a[offset%128] & b[k])%2);
}
__global__ void gpushiftl(uint64_t* c,int t)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    c[k]=shiftl(c[k],t);
}
__global__ void gpushiftr(uint64_t* c,int t)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    c[k]=shiftr(c[k],t);
}
__global__ void countweight(int* c,int* result)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    if (c[k] < 23)
    {
        atomicAdd(&result[c[k]],1);
    }

}
__global__ void hbadd(uint64_t* b)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    b[k] += MaxThreadPerBlock * Blocknum;
}
__global__ void clearzero(int* c)
{
    int k = 0;
    k = threadIdx.x + blockIdx.x * blockDim.x;
    c[k] = 0;
}
int main()
{
    srand(time(NULL));
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
    uint64_t* h_A = (uint64_t*)malloc(sizeof(uint64_t) * M);
    uint64_t* h_A1 = (uint64_t*)malloc(sizeof(uint64_t) * M);
    uint64_t* h_B = (uint64_t*)malloc(sizeof(uint64_t) * Blocknum * MaxThreadPerBlock);
    int* h_C = (int*)malloc(sizeof(int) * Blocknum * MaxThreadPerBlock * M);
    int* weight = (int*)malloc(sizeof(int) * Blocknum * MaxThreadPerBlock);
    int* resultnum = (int*)malloc(sizeof(int) * 23);
   
    for (int i = 0; i < 23; i++)
    {
        resultnum[i] = 0;
    }

    fp_test = fopen("G_PAC.txt", "r");
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
            h_A1[i] = 2 * h_A1[i] + h_AT[j* M + i];
        }
    }
    for (int i = 0; i < Blocknum * MaxThreadPerBlock; i++)
    {
        h_B[i] = (uint64_t)19406846/2*4096*1024+1+i;   
    }

    uint64_t* d_A, * d_B,* d_A1;
    int* d_C, * d_resultnum,* d_weight;
    cudaMalloc((void**)&d_weight, sizeof(int) * Blocknum * MaxThreadPerBlock);
    cudaMalloc((void**)&d_A, sizeof(uint64_t) * M);
    cudaMalloc((void**)&d_A1, sizeof(uint64_t) * M);
    cudaMalloc((void**)&d_B, sizeof(uint64_t) * Blocknum * MaxThreadPerBlock);
    cudaMalloc((void**)&d_C, sizeof(int) * Blocknum * MaxThreadPerBlock * M);
    cudaMalloc((void**)&d_resultnum, sizeof(int) * 23);
   


    cudaMemcpy(d_A, h_A, M * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A1, h_A1, M * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Blocknum * MaxThreadPerBlock * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_resultnum, resultnum, 23 * sizeof(int), cudaMemcpyHostToDevice);
    uint64_t k = 19406846;
    uint64_t k1=k+4096;
    clock_t start = clock();
    clock_t end;
    FILE* fp_16;
    FILE* fp_18;
    FILE* fp_20;
    FILE* fp_22;
    fp_16=fopen("16.txt","a+");
    fp_18=fopen("18.txt","a+");
    fp_20=fopen("20.txt","a+");
    fp_22=fopen("22.txt","a+");
    fp_test=fopen("results.txt","a+");
    int shiftnum=0;
    while (k<k1)//��
    //while(k<256)
    {
            for (int i = 0; i < 128; i++)
            {
                simu << <Blocknum * MaxThreadPerBlock / 128, 128 >> > (d_A, d_B,d_C,i,d_weight);
            }
        
        //cudaThreadSynchronize();
        countweight << < Blocknum, MaxThreadPerBlock >> > (d_weight,d_resultnum);
//       cudaMemcpy(h_C, d_C, Blocknum * MaxThreadPerBlock * (M + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(weight, d_weight, Blocknum * MaxThreadPerBlock * sizeof(int), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(int), cudaMemcpyDeviceToHost);
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

        if(k%2==0)
        {
            shiftnum=rand()%21+10;
            gpushiftl << < Blocknum, MaxThreadPerBlock >> > (d_B,shiftnum);
            //hbadd << < Blocknum, MaxThreadPerBlock >> > (d_B);
        }
        else
        {
            gpushiftr << < Blocknum, MaxThreadPerBlock >> > (d_B,shiftnum);
            //cudaThreadSynchronize();
            hbadd << < Blocknum, MaxThreadPerBlock >> > (d_B);
        }
        clearzero << < Blocknum, MaxThreadPerBlock >> > (d_weight);
        if ((k+1)%1024==0)
        {
            if ((k+1)%4096)
            {
                checkrepeat(fp_16,16,0);
                checkrepeat(fp_18,18,0);
                checkrepeat(fp_20,20,0);
                checkrepeat(fp_22,22,0);
                cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
            }
            else
            {
                srand(time(NULL));
                resultnum[16]=checkrepeat(fp_16,16,1);
                resultnum[18]=checkrepeat(fp_18,18,1);
                resultnum[20]=checkrepeat(fp_20,20,1);
                resultnum[22]=checkrepeat(fp_22,22,1);
            }
            end = clock();
            printf("\ntime=%f min\n", (double)(end - start) / CLOCKS_PER_SEC/60);
           
            //cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
            fprintf(fp_test,"k = %ld\n",k);
            for (int i = 0; i < 23; i++)
            {
                cout << resultnum[i] << "  ";
                fprintf(fp_test,"%d ",resultnum[i]);
            }
            cout << endl;
            cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            printf("last b: %ld\n",h_B[Blocknum * MaxThreadPerBlock - 1]);
            fprintf(fp_test,"\n");
            fprintf(fp_test,"last b: %ld\n",h_B[Blocknum * MaxThreadPerBlock - 1]);
            printf("k: %ld\n",k);            
            fclose(fp_test);
            fp_test=fopen("results.txt","a");
        }
        k++;
        //cudaThreadSynchronize();
        //cout<<k<<endl;
    }
    end = clock();
    printf("\ntime=%f min\n", (double)(end - start) / CLOCKS_PER_SEC/60);
    cudaMemcpy(h_B, d_B, Blocknum * MaxThreadPerBlock * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, Blocknum * MaxThreadPerBlock * M * sizeof(int), cudaMemcpyDeviceToHost);
    printf("last b: %ld\n",h_B[Blocknum * MaxThreadPerBlock - 1]);
    //cudaMemcpy(resultnum, d_resultnum, 23 * sizeof(int), cudaMemcpyDeviceToHost);
    cout<<endl;
    // for(int i=0;i<10;i++)
    // {
    //     cout<<h_B[i]<<" ";
    // }
    // cout<<endl;
    // for(int i=0;i<2000;i++)
    // {
    //     cout<<h_C[i]<<" ";
    //     if((i+1)%128==0)
    //     {
    //         cout<<endl<<endl;
    //     }
    // }
    fprintf(fp_test,"k = %ld\n",k);
    for (int i = 0; i < 23; i++)
    {
        cout << resultnum[i] << "  ";
        fprintf(fp_test,"%d ",resultnum[i]);
    }
    fprintf(fp_test,"\n");
    cout << endl;
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