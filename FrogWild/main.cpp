#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define EDGES 5066842
#define NODES 855802


int main()
{  
    fstream data;
    data.open("webGoogle.txt");
    if(data.is_open()){
        string line;
        while(getline(data,line)){
            cout<<line<<endl;
        }
    }
    data.close();
    return 0;
}