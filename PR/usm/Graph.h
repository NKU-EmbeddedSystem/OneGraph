#pragma once
#include"global.h"
#include<chrono>
#include<fstream>
#include <CL/sycl.hpp>
#include <vector>
template<class EdgeType>
class Graph {
private:
public:
    SIZE_TYPE sourcenode = 0;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE* nodePointers;
    SIZE_TYPE* outDegree;
    EdgeType* edgeArray;
    SIZE_TYPE* degree;
    sycl::queue q;
    void setSourceNode(uint s)
    {
        this->sourcenode = s;
    }
    void readDataFromFile(const string& filename);
    //void readDataFromFileQ(const string& filename);
    void readDataFromFileQ(const string& filename,sycl::queue q);
    ~Graph();
    void printself() {
        cout << "vertexArrSize:" << this->vertexArrSize << " " << "edgeArrSize:" << this->edgeArrSize << "\n";
        cout << "nodePointers:" << "\n";
        for (int i = 0; i < vertexArrSize; i++)
        {
            cout << nodePointers[i] << " ";
        }
        cout << "\n";
        cout << "outDegree:" << "\n";
        for (int i = 0; i < vertexArrSize; i++)
        {
            cout << outDegree[i] << " ";
        }
        cout << "\n";
        cout << "edgeArr:" << "\n";
        for (int i = 0; i < edgeArrSize; i++)
        {
            cout << edgeArray[i] << " ";
        }
        cout << "\n";
        cout << "degree:" << "\n";
        for (int i = 0; i < vertexArrSize; i++)
        {
            cout << degree[i] << " ";
        }
        cout << "\n";
    }
    ull findbignode();
    ull Averagedegree();
    ull Bignode_num();
    bool Isdirected();
    void convert2BEL(string input);
};
//========================read-in c++ new version========================
template<class EdgeType>
void Graph<EdgeType>::readDataFromFile(const string& fileName) {
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char*)&this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char*)&this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));

    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";

    outDegree = new SIZE_TYPE[vertexArrSize];
    infile.read((char*)outDegree, sizeof(uint) * vertexArrSize);

    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);

    edgeArray = new EdgeType[edgeArrSize];
    infile.read((char*)edgeArray, sizeof(EdgeType) * edgeArrSize);
    degree = new SIZE_TYPE[vertexArrSize];
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << "\n";
}

//========================read-in malloc_shared version========================
template<class EdgeType>
void Graph<EdgeType>::readDataFromFileQ(const string& fileName,sycl::queue q) {
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char*)&this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char*)&this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));

    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";

    outDegree = malloc_shared<SIZE_TYPE>(vertexArrSize, q);
    infile.read((char*)outDegree, sizeof(uint) * vertexArrSize);

    nodePointers = malloc_shared<EDGE_POINTER_TYPE>(vertexArrSize, q);
    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);

    edgeArray = malloc_shared<EdgeType>(edgeArrSize, q);
    infile.read((char*)edgeArray, sizeof(EdgeType) * edgeArrSize);

    degree =malloc_shared<SIZE_TYPE>(vertexArrSize, q);
    infile.close();

    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << "\n";
}

//========================read-in c++ malloc version========================
//template<class EdgeType>
//void Graph<EdgeType>::readDataFromFileQ(const string& fileName) {
//    cout << "readDataFromFile" << "\n";
//    auto startTime = chrono::steady_clock::now();
//    ifstream infile(fileName, ios::in | ios::binary);
//    infile.read((char*)&this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
//    infile.read((char*)&this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));
//
//    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";
//
//    outDegree = (SIZE_TYPE*)malloc(vertexArrSize * sizeof(SIZE_TYPE));
//    infile.read((char*)outDegree, sizeof(uint) * vertexArrSize);
//
//    nodePointers = (EDGE_POINTER_TYPE*)malloc(vertexArrSize * sizeof(EDGE_POINTER_TYPE));
//    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
//
//    edgeArray = (EdgeType*)malloc(edgeArrSize * sizeof(EdgeType));
//    infile.read((char*)edgeArray, sizeof(EdgeType) * edgeArrSize);
//
//    degree = (SIZE_TYPE*)malloc(vertexArrSize * sizeof(SIZE_TYPE));
//    infile.close();
//
//    auto endTime = chrono::steady_clock::now();
//    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
//    cout << "readDataFromFile " << duration << " ms" << "\n";
//}

template<class EdgeType>
ull Graph<EdgeType>::findbignode()
{
    for (ull i = 0; i < vertexArrSize - 1; i++)
    {
        ull degree = nodePointers[i + 1] - nodePointers[i];
        if (degree > 3000)
        {
            cout << "find a big node:" << i << "\n";
            return i;
        }
    }
    cout << "failed to find a big node" << "\n";
    getchar();
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
    //free(nodePointers,q);
    //free(outDegree, q);
    //free(edgeArray, q);
    //free(degree, q);
    //delete[]nodePointers;
    //delete[]outDegree;
    //delete[]degree;
    //delete[]edgeArray;

    cout << "~Graph" << "\n";
}
