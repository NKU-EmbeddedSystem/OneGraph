#include <string>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include <thread>
#include "TimeRecord.h"
#include "globals.h"
#pragma once

struct PartEdgeListInfo {
    SIZE_TYPE partActiveNodeNums;
    SIZE_TYPE partEdgeNums;
    SIZE_TYPE partStartIndex;
};

using namespace std;

template<class EdgeType>
class TestMeta {
public:
    ~TestMeta();
};

template<class EdgeType>
TestMeta<EdgeType>::~TestMeta() {

}

template<class EdgeType>
class GraphMeta {
public:
    sycl::queue streamStatic;
    sycl::queue streamDynamic;
    size_t grid = 1024*108;
    size_t block = 1024;
    SIZE_TYPE partOverloadSize;
    EDGE_POINTER_TYPE overloadSize;
    SIZE_TYPE sourceNode = 0;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE *nodePointers;
    EdgeType *edgeArray;
    //special for pr
    SIZE_TYPE *outDegree;
    SIZE_TYPE *degree;
    uint *label;
    float *valuePr;
    SIZE_TYPE *value;
    uint *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    EDGE_POINTER_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    EdgeType *overloadEdgeList;
    //GPU
    uint *resultD;
    uint *prefixSumTemp;
    EdgeType *staticEdgeListD;
    EdgeType *overloadEdgeListD;
    uint *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    SIZE_TYPE *outDegreeD;
    // async need two labels
    uint *isActiveD;
    uint *isStaticActive;
    uint *isOverloadActive;
    SIZE_TYPE *valueD;
    float *valuePrD;
    float *sumD;
    SIZE_TYPE *activeNodeListD;
    EDGE_POINTER_TYPE *activeOverloadNodePointersD;
    EDGE_POINTER_TYPE *activeOverloadDegreeD;
    float adviseRate;
    int paramSize;
    ALG_TYPE algType;    

    void readDataFromFile(const string &fileName, uint isPagerank);

    void transFileUintToUlong(const string &fileName);

    ~GraphMeta();

    void setPrestoreRatio(float adviseK, int paramSize) {
        this->adviseRate = adviseK;
        this->paramSize = paramSize;
    }

    void initGraphHost();

    void initGraphDevice();

    void setAlgType(ALG_TYPE type) {
        algType = type;
    }

    void setSourceNode(SIZE_TYPE sourceNode) {
        this->sourceNode = sourceNode;
    }

    void fillEdgeArrByMultiThread(uint overloadNodeSize);

    void caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum);

private:
    SIZE_TYPE max_partition_size;
    SIZE_TYPE max_static_node;
    SIZE_TYPE total_gpu_size;
    uint fragmentSize = 4096;

    void getMaxPartitionSize();

    void initLableAndValue();

};

template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromFile(const string &fileName, uint isPagerank) {
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char *) &this->edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";
    if (isPagerank) {
        outDegree = new SIZE_TYPE [vertexArrSize];
        infile.read((char *) outDegree, sizeof(uint) * vertexArrSize);
    }
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    edgeArray = new EdgeType[edgeArrSize];
    infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << "\n";
}

template<class EdgeType>
void GraphMeta<EdgeType>::transFileUintToUlong(const string &fileName) {
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(uint));
    infile.read((char *) &this->edgeArrSize, sizeof(uint));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << "\n";
    outDegree = new uint[vertexArrSize];
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(uint) * vertexArrSize);
    edgeArray = new EdgeType[edgeArrSize];
    infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    vector<ulong> transData(edgeArrSize);
    for (int i = 0; i < edgeArrSize; i++) {
        transData[i] = edgeArray[i];
    }

    std::ofstream outfile(fileName.substr(0, fileName.length() - 4) + "lcsr", std::ofstream::binary);

    outfile.write((char *) &vertexArrSize, sizeof(unsigned int));
    outfile.write((char *) &edgeArrSize, sizeof(unsigned int));
    outfile.write((char *) nodePointers, sizeof(unsigned int) * vertexArrSize);
    outfile.write((char *) transData.data(), sizeof(ulong) * edgeArrSize);

    outfile.close();
}

template<class EdgeType>
GraphMeta<EdgeType>::~GraphMeta() {
    delete[] edgeArray;
    delete[] nodePointers;
    delete[] outDegree;
    cout << "~GraphMeta" << "\n";    
}

template<class EdgeType>
void GraphMeta<EdgeType>::initGraphHost() {
    cout << "initGraphHost()" << "\n";
    degree = new SIZE_TYPE[vertexArrSize];
    // static region nodes
    isInStatic = new uint[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    activeOverloadNodePointers = new EDGE_POINTER_TYPE[vertexArrSize];

    // calculate degree
    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];

    getMaxPartitionSize();
    initLableAndValue();
    overloadEdgeList = (EdgeType *) malloc(overloadSize * sizeof(EdgeType));
    staticNodePointer = new SIZE_TYPE[vertexArrSize];
    for (uint i = 0; i < max_static_node; i++) {
        staticNodePointer[i] = nodePointers[i];
    }
}


template<class EdgeType>
void GraphMeta<EdgeType>::initGraphDevice() {
    cout << "initGraphDevice()" << "\n";
    resultD = malloc_device<uint>(grid,streamStatic);
    prefixSumTemp = malloc_device<uint>(vertexArrSize,streamStatic);

    // double queue
    streamDynamic = sycl::queue();
    streamStatic = sycl::queue();
    //pre store
    TimeRecord<chrono::milliseconds> totalProcess("pre move data");
    totalProcess.startRecord();
    staticEdgeListD = malloc_device<EdgeType>(max_partition_size,streamStatic);
    streamStatic.memcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(EdgeType));
    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();

    isInStaticD = malloc_device<uint>(vertexArrSize,streamStatic);
    overloadNodeListD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
    staticNodeListD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
    staticNodePointerD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);

    streamStatic.memcpy(staticNodePointerD, staticNodePointer, vertexArrSize * sizeof(SIZE_TYPE));
    streamStatic.memcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(uint));
    overloadEdgeListD = malloc_device<EdgeType>(partOverloadSize,streamStatic);
    degreeD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
    isActiveD = malloc_device<uint>(vertexArrSize,streamStatic);
    isStaticActive = malloc_device<uint>(vertexArrSize,streamStatic);
    isOverloadActive = malloc_device<uint>(vertexArrSize,streamStatic);

    activeNodeListD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
    activeOverloadNodePointersD = malloc_device<EDGE_POINTER_TYPE>(vertexArrSize,streamStatic);
    activeOverloadDegreeD = malloc_device<EDGE_POINTER_TYPE>(vertexArrSize,streamStatic);

    streamStatic.memcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE));
    streamStatic.memcpy(isActiveD, label, vertexArrSize * sizeof(uint));

    streamStatic.memset(isStaticActive, 0, vertexArrSize * sizeof(uint));
    streamStatic.memset(isOverloadActive, 0, vertexArrSize * sizeof(uint));

    streamStatic.wait();

    if(algType == PR) {
        outDegreeD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
        valuePrD = malloc_device<float>(vertexArrSize,streamStatic);
        sumD = malloc_device<float>(vertexArrSize,streamStatic);

        streamStatic.memcpy(outDegreeD, outDegree, vertexArrSize * sizeof(SIZE_TYPE));
        streamStatic.memcpy(valuePrD, valuePr, vertexArrSize * sizeof(float));
        streamStatic.memset(sumD, 0, vertexArrSize * sizeof(float));
    } else {
        valueD = malloc_device<SIZE_TYPE>(vertexArrSize,streamStatic);
        streamStatic.memcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE));
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::getMaxPartitionSize() {
    int deviceID;
    size_t totalTemp = 1024;
    size_t totalMemory = 12*totalTemp*totalTemp*totalTemp;
    size_t availMemory = totalMemory;
    long reduceMem = paramSize * sizeof(SIZE_TYPE) * (long) vertexArrSize;
    total_gpu_size = (availMemory - reduceMem) / sizeof(EdgeType);

    float adviseK = (10 - (float) edgeArrSize / (float) total_gpu_size) / 9;
    cout << "adviseK " << adviseK << "\n";
    if (adviseK < 0) {
        adviseK = 0.5;
        cout << "adviseK " << adviseK << "\n";
    }
    if (adviseK > 1) {
        adviseK = 1.0;
        cout << "adviseK " << adviseK << "\n";
    }
    cout << "adviseRate " << adviseRate << "\n";
    if (adviseRate > 0) {
        adviseK = adviseRate;
    }

    max_partition_size = adviseK * total_gpu_size;
    if (max_partition_size > edgeArrSize) {
        max_partition_size = edgeArrSize;
    }
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << "\n";
    if (max_partition_size > UINT_MAX) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = UINT_MAX;
    }
    max_static_node = 0;
    SIZE_TYPE edgesInStatic = 0;
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true; 
            if (i > max_static_node) max_static_node = i;
            edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }

    cout << "max_partition_size " << max_partition_size << " nodePointers[vertexArrSize-1]" << nodePointers[vertexArrSize-1] << " edgesInStatic " << edgesInStatic << "\n";

    partOverloadSize = total_gpu_size - max_partition_size; // overload part that cannot be stored in GPU memory
    overloadSize = edgeArrSize - edgesInStatic;
    cout << " partOverloadSize " << partOverloadSize << " overloadSize " << overloadSize << "\n";
}

template<class EdgeType>
void GraphMeta<EdgeType>::initLableAndValue() {

    label = new uint[vertexArrSize];
    if (algType == PR) {
        valuePr = new float[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0;
        }
    } else {
        value = new SIZE_TYPE[vertexArrSize];
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = vertexArrSize + 1;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = 0xFFFFFFFF;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::fillEdgeArrByMultiThread(uint overloadNodeSize) {
    int threadNum = 20;
    if (overloadNodeSize < 50) {
        threadNum = 1;
    }
    thread runThreads[threadNum];

    for (int threadIndex = 0; threadIndex < threadNum; threadIndex++) {
        runThreads[threadIndex] = thread([&, threadIndex] {
            float waitToHandleNum = overloadNodeSize;
            float numThreadsF = threadNum;
            unsigned int chunkSize = ceil(waitToHandleNum / numThreadsF);
            unsigned int left, right;
            left = threadIndex * chunkSize;
            right = min(left + chunkSize, overloadNodeSize);
            unsigned int thisNode;
            unsigned int thisDegree;
            EDGE_POINTER_TYPE fromHere = 0;
            EDGE_POINTER_TYPE fromThere = 0;
            for (unsigned int i = left; i < right; i++) {
                thisNode = overloadNodeList[i];
                thisDegree = degree[thisNode];
                fromHere = activeOverloadNodePointers[i];
                fromThere = nodePointers[thisNode];
                
                for (unsigned int j = 0; j < thisDegree; j++) {
                    overloadEdgeList[fromHere + j] = edgeArray[fromThere + j];
                }
                
            }
        });
    }
    for (unsigned int t = 0; t < threadNum; t++) {
        runThreads[t].join();
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::caculatePartInfoForEdgeList(SIZE_TYPE overloadNodeNum, EDGE_POINTER_TYPE overloadEdgeNum) {
    partEdgeListInfoArr.clear();
    if (partOverloadSize < overloadEdgeNum) {
        uint left = 0;
        uint right = overloadNodeNum - 1;
        while ((activeOverloadNodePointers[right] + degree[overloadNodeList[right]] -
                activeOverloadNodePointers[left]) >
               partOverloadSize) {

            uint start = left;
            uint end = right;
            uint mid;
            while (start <= end) {
                mid = (start + end) / 2;
                EDGE_POINTER_TYPE headDistance = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
                EDGE_POINTER_TYPE tailDistance =
                        activeOverloadNodePointers[mid] + degree[overloadNodeList[mid]] -
                        activeOverloadNodePointers[left];
                if (headDistance <= partOverloadSize && tailDistance > partOverloadSize) {
                    break;
                } else if (tailDistance <= partOverloadSize) {
                    start = mid + 1;
                } else if (headDistance > partOverloadSize) {
                    end = mid - 1;
                }
            }
            
            PartEdgeListInfo info;
            info.partActiveNodeNums = mid - left;
            info.partEdgeNums = activeOverloadNodePointers[mid] - activeOverloadNodePointers[left];
            info.partStartIndex = left;
            partEdgeListInfoArr.push_back(info);
            left = mid;
        }

        PartEdgeListInfo info;
        info.partActiveNodeNums = right - left + 1;
        info.partEdgeNums =
                activeOverloadNodePointers[right] + degree[overloadNodeList[right]] - activeOverloadNodePointers[left];
        info.partStartIndex = left;
        partEdgeListInfoArr.push_back(info);
    } else {
        PartEdgeListInfo info;
        info.partActiveNodeNums = overloadNodeNum;
        info.partEdgeNums = overloadEdgeNum;
        info.partStartIndex = 0;
        partEdgeListInfoArr.push_back(info);
    }
}
