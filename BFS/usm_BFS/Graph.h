#pragma once
#include"./global.h"
#include<map>
#include<CL/sycl.hpp>
using namespace sycl;
template<class EdgeType>
class Graph {
private:
public:
    SIZE_TYPE sourcenode = 0;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE* nodePointers;
    EdgeType* edgeArray;
    SIZE_TYPE* degree;
    sycl::queue mq;
    void setSourceNode(uint s)
    {
        this->sourcenode = s;
    }
    void readDataFromFile(const string& filename, sycl::queue& q);
    ~Graph();
    void printself() {
        cout << "vertexArrSize:" << this->vertexArrSize << " " << "edgeArrSize:" << this->edgeArrSize << "\n";
        cout << "nodePointers:" << "\n";
        for (int i = 0; i < 10; i++)
        {
            cout << nodePointers[i] << " ";
        }
        cout << "\n";
        cout << "edgeArr:" << "\n";
        for (int i = 0; i < 10; i++)
        {
            cout << edgeArray[i] << " ";
        }
    }
    ull findbignode();
    ull Averagedegree();
    ull Bignode_num();
    bool Isdirected();
    void convert2BEL(string input);
    uint findbignode(uint last);

};
template<class EdgeType>
void Graph<EdgeType>::readDataFromFile(const string& fileName, sycl::queue& q)
{
    mq = q;
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char*)&this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char*)&this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";
    degree = new SIZE_TYPE[vertexArrSize];
    //nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    
    this->nodePointers = malloc_shared<EDGE_POINTER_TYPE>(vertexArrSize, mq);
    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    
    //edgeArray = new EdgeType[edgeArrSize];
    this->edgeArray = malloc_shared<EdgeType>(edgeArrSize, mq);

    infile.read((char*)edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << "\n";
}

template<class EdgeType>
uint Graph<EdgeType>::findbignode(uint last)
{
    for (uint i = last+1; i < vertexArrSize - 1; i++)
    {
        uint degree = nodePointers[i + 1] - nodePointers[i];
        if (degree > 3000)
        {
            cout << "find a big node:" << i << "\n";
            return i;
        }
    }
    cout << "find a big node:" << last << "\n";
    return last;
    //cout << "failed to find a big node" << "\n";
    //getchar();
}

template<class EdgeType>
ull Graph<EdgeType>::Averagedegree() {
    ull _degree = 0;
    for (ull i = 0; i < vertexArrSize; i++) {
        _degree += degree[i];
    }
    _degree /= vertexArrSize;
    return _degree;
}

template<class EdgeType>
ull Graph<EdgeType>::Bignode_num() {

    ull num = 0;
    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] >= edgeArrSize) {
            cout << i << "   " << nodePointers[i] << "\n";
            break;
        }

        degree[i] = nodePointers[i + 1] - nodePointers[i];

        if (degree[i] >= 32)
            num++;
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    if (degree[vertexArrSize - 1] >= 32)
        num++;
    return num;
}

template<class EdgeType>
bool Graph<EdgeType>::Isdirected() {
    for (ull i = 0; i < vertexArrSize; i++) {
        ull vertex = i;
        if (degree[vertex] != 0) {
            ull pointer = nodePointers[vertex];
            for (int j = 0; j < degree[vertex]; j++) {
                uint end = edgeArray[pointer + j];
                ull end_p = nodePointers[end];

                for (int m = 0; m < degree[end]; m++) {
                    if (edgeArray[end_p + m] == vertex)
                        return true;
                }
            }
        }
    }
    return false;
}
template<class EdgeType>
void Graph<EdgeType>::convert2BEL(string input) {
    cout << "convert to BEL" << "\n";
    std::ofstream coloutfile(input + ".bel.col", std::ofstream::binary);
    std::ofstream dstoutfile(input + ".bel.dst", std::ofstream::binary);
    std::ofstream valoutfile(input + ".bel.val", std::ofstream::binary);
    ull placeholder = 0;
    ull* edges = new ull[edgeArrSize];
    uint* weights = new uint[edgeArrSize];
    for (uint i = 0; i < edgeArrSize; i++) {
        OutEdgeWeighted edge = edgeArray[i];
        edges[i] = edge.end;
        weights[i] = edge.w8;
    }
    cout << "vertexArrSize " << vertexArrSize << "\n";
    cout << "edgeArrSize " << edgeArrSize << "\n";
    coloutfile.write((char*)&this->vertexArrSize, 8);
    coloutfile.write((char*)&placeholder, 8);
    coloutfile.write((char*)nodePointers, sizeof(ull) * vertexArrSize);
    coloutfile.close();
    dstoutfile.write((char*)&edgeArrSize, 8);
    dstoutfile.write((char*)&placeholder, 8);
    dstoutfile.write((char*)edges, sizeof(ull) * edgeArrSize);
    dstoutfile.close();
    valoutfile.write((char*)&edgeArrSize, 8);
    valoutfile.write((char*)&placeholder, 8);
    valoutfile.write((char*)weights, sizeof(uint) * edgeArrSize);
    valoutfile.close();
    cout << "prepare BEL OK" << "\n";
}
template<class EdgeType>
Graph<EdgeType>::~Graph()
{
    delete[] degree;
    cout << "~Graph" << "\n";
}
