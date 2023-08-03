#include "data.h"


int main()
{  
    fstream data;
    data.open("webGoogle.csv");
    string line,word;
    int** edges=new int*[EDGES];
    for(int i=0; i<EDGES; i++){
        edges[i]=new int[2];
    }
    int** cluster=new int*[NODES];
    for(int i=0; i<NODES;i++){
        cluster[i]=new int[2];
    }
    int count=0;

    return 0;
}