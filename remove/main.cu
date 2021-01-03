#include<iostream>
#include <stdio.h>
using namespace std;
int main()
{
    FILE* f;
    //FILE* f1;
    int64_t num;
    int64_t k=0;
    int64_t sum=0;
    int64_t results[1000000];
    f=fopen("22.txt","r");
    //f1=fopen("18norepeat.txt","w");
    for(int64_t i=0;!feof(f);i++)
    {
        k=0;
        fscanf(f, "%ld", &num);
        if(!feof(f))
        {
            for(int t=0;t<sum;t++)
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
            }
        }
    }
    fclose(f);
    f=fopen("22.txt","w");
    for(int64_t i=0;i<sum;i++)
    {
        fprintf(f,"%ld ",results[i]);
    }
    cout<<"sum:"<<sum<<endl;
    fclose(f);
    //fclose(f1);
    return 0;
}