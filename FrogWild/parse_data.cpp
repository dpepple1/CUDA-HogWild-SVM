#include "data.h"



void return_list(string path, int** arr){
    fstream data;
    data.open(path);
    string line,word;
    int count=0;
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); //Stream Class to operate on strings
            while(getline(stream_data,word,',')){
                //Extract data until ',' is found
                *(arr[count])=word;
                arr[count]++;
            }
            count++;
        }
    }
    data.close();
}