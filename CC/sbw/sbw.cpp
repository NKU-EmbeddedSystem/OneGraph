#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include<malloc.h>
#include <pthread.h>
#include "global.h"
#include "partitioner.h"
#include "partitioner.cpp"
static constexpr size_t sub_group_size = 16;
using namespace sycl;
using namespace std;
const int NUM_THREADS_h = 20;
const int NUM_THREADS = 1024 * 108;
static constexpr size_t N = 1024 * 108;
static constexpr size_t B = 1024;
class my_device_selector : public device_selector {
public:
    my_device_selector(std::string vendorName) : vendorName_(vendorName) {};
    int operator()(const device& dev) const override {
        int rating = 0;
        if (dev.is_gpu() && (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
            rating = 3;
        else if (dev.is_gpu()) rating = 2;
        else if (dev.is_cpu()) rating = 1;
        return rating;
    };
private:
    std::string vendorName_;
};
char* optarg = NULL;    /* pointer to the start of the option argument  */
int   optind = 1;       /* number of the next argv[] to be evaluated    */
int   opterr = 1;       /* non-zero if a question mark should be returned
                           when a non-valid option character is detected */

                           /* handle possible future character set concerns by putting this in a macro */
#define _next_char(string)  (char)(*(string+1))

int getopt(int argc, char* argv[], char* opstring)
{
    static char* pIndexPosition = NULL; /* place inside current argv string */
    char* pArgString = NULL;        /* where to start from next */
    char* pOptString;               /* the string in our program */


    if (pIndexPosition != NULL) {
        /* we last left off inside an argv string */
        if (*(++pIndexPosition)) {
            /* there is more to come in the most recent argv */
            pArgString = pIndexPosition;
        }
    }

    if (pArgString == NULL) {
        /* we didn't leave off in the middle of an argv string */
        if (optind >= argc) {
            /* more command-line arguments than the argument count */
            pIndexPosition = NULL;  /* not in the middle of anything */
            return EOF;             /* used up all command-line arguments */
        }

        /*---------------------------------------------------------------------
         * If the next argv[] is not an option, there can be no more options.
         *-------------------------------------------------------------------*/
        pArgString = argv[optind++]; /* set this to the next argument ptr */

        if (('/' != *pArgString) && /* doesn't start with a slash or a dash? */
            ('-' != *pArgString)) {
            --optind;               /* point to current arg once we're done */
            optarg = NULL;          /* no argument follows the option */
            pIndexPosition = NULL;  /* not in the middle of anything */
            return EOF;             /* used up all the command-line flags */
        }

        /* check for special end-of-flags markers */
        if ((strcmp(pArgString, "-") == 0) ||
            (strcmp(pArgString, "--") == 0)) {
            optarg = NULL;          /* no argument follows the option */
            pIndexPosition = NULL;  /* not in the middle of anything */
            return EOF;             /* encountered the special flag */
        }

        pArgString++;               /* look past the / or - */
    }

    if (':' == *pArgString) {       /* is it a colon? */
        /*---------------------------------------------------------------------
         * Rare case: if opterr is non-zero, return a question mark;
         * otherwise, just return the colon we're on.
         *-------------------------------------------------------------------*/
        return (opterr ? (int)'?' : (int)':');
    }
    else if ((pOptString = strchr(opstring, *pArgString)) == 0) {
        /*---------------------------------------------------------------------
         * The letter on the command-line wasn't any good.
         *-------------------------------------------------------------------*/
        optarg = NULL;              /* no argument follows the option */
        pIndexPosition = NULL;      /* not in the middle of anything */
        return (opterr ? (int)'?' : (int)*pArgString);
    }
    else {
        /*---------------------------------------------------------------------
         * The letter on the command-line matches one we expect to see
         *-------------------------------------------------------------------*/
        if (':' == _next_char(pOptString)) { /* is the next letter a colon? */
            /* It is a colon.  Look for an argument string. */
            if ('\0' != _next_char(pArgString)) {  /* argument in this argv? */
                optarg = &pArgString[1];   /* Yes, it is */
            }
            else {
                /*-------------------------------------------------------------
                 * The argument string must be in the next argv.
                 * But, what if there is none (bad input from the user)?
                 * In that case, return the letter, and optarg as NULL.
                 *-----------------------------------------------------------*/
                if (optind < argc)
                    optarg = argv[optind++];
                else {
                    optarg = NULL;
                    return (opterr ? (int)'?' : (int)*pArgString);
                }
            }
            pIndexPosition = NULL;  /* not in the middle of anything */
        }
        else {
            /* it's not a colon, so just return the letter */
            optarg = NULL;          /* no argument follows the option */
            pIndexPosition = pArgString;    /* point to the letter we're on */
        }
        return (int)*pArgString;    /* return the letter that matched */
    }
}
void validation(uint vertexArrSize, bool* labelD) {
    int cnt = 0;
    for (int i = 0; i < vertexArrSize; i++) {
        if (labelD[i] == 0) cnt++;
    }
    cout << "connetcted component num: "<<cnt<<"\n";
}
template<class T, class E>
T* inclusive_scan(E* current, T* next, uint num, sycl::queue q) {
    unsigned int two_power = 1;
    unsigned int num_iter = ceil(log2(num));
    uint* result = NULL;
    for (unsigned int iter = 0; iter < num_iter; iter++, two_power *= 2) {
        if (iter % 2 == 0) {
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
                    auto tid = item.get_global_id()[0];
                    for (uint i = tid; i < num; i += N) {
                        if (i < two_power) {
                            next[i] = current[i];
                        } else {
                        next[i] = current[i] +current[i - two_power];
                        }
                    }
                    });
                result = next;
                }).wait();
        } else {
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
                    auto tid = item.get_global_id()[0];
                    for (uint i = tid; i < num; i += N) {
                        if (i < two_power) {
                            current[i] = next[i];
                        } else {
                            current[i] = next[i] + next[i - two_power];
                        }
                    }
                    });
                result = current;
                }).wait();
        }
    }
    
    return result;
}
struct threadParam_t
{
	int t_id; 
    EDGE_POINTER_TYPE* subVertex;
    uint* subEdge;
    SIZE_TYPE* offset;
    ull activeNodeNum;
    uint* degree;
    uint* edgeArray;
    EDGE_POINTER_TYPE* nodePointers;
};
void* threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
	int t_id = p -> t_id;

    EDGE_POINTER_TYPE* subVertex = p -> subVertex;
    uint* subEdge = p -> subEdge;
    SIZE_TYPE* offset = p -> offset;
    ull activeNodeNum = p -> activeNodeNum;
    uint* degree = p -> degree;
    uint* edgeArray = p -> edgeArray;
    EDGE_POINTER_TYPE* nodePointers = p -> nodePointers;

    for (uint i = t_id; i < activeNodeNum; i += NUM_THREADS_h) {
        uint v = subVertex[i];
        for (uint k1 = offset[i], k2 = nodePointers[v]; k1 < offset[i] + degree[v]; k1++, k2++) {
            subEdge[k1] = edgeArray[k2];
        }
    }
    pthread_exit(NULL);
    return 0; 
}
void generateSubG(uint* subIndex, uint* subDegree, SIZE_TYPE* subOffset, bool isActive[], sycl::queue q, SIZE_TYPE vertexArrSize, SIZE_TYPE numActiveVertices, EDGE_POINTER_TYPE subVertex[], SIZE_TYPE degree[], SIZE_TYPE degreeD[],SIZE_TYPE offset[], uint* edgeArray, EDGE_POINTER_TYPE* nodePointers, uint* subEdge) {
    //step 1. get subIndex
    uint* isActiveDprefixLable = malloc_device<uint>(vertexArrSize, q);

    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                isActiveDprefixLable[i] = (uint)isActive[i];
            }
            });
        }).wait();

    subIndex = inclusive_scan(isActiveDprefixLable, subIndex, vertexArrSize, q);

    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                subIndex[i] -= isActive[i];
            }
            });
        }).wait();

    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
            auto tid = item.get_global_id()[0];
        for (SIZE_TYPE id = tid; id < vertexArrSize; id += N) {
            //step2. get subVertex
            //step3. get subDegree
            if (isActive[id] == 1) {//可以把step23放在一个parallel_for里，因为它们相互不影响
                subVertex[subIndex[id]] = id;
                subDegree[id] = degreeD[id];
            }
            else {
                subDegree[id] = 0;
            }
            }
            });
        }).wait();

    SIZE_TYPE* subDegreePrefixSumLable = malloc_device<SIZE_TYPE>(vertexArrSize, q);
    q.memcpy(subDegreePrefixSumLable, subDegree, sizeof(uint) * vertexArrSize).wait();
        
    //step4. get subOffset
    subOffset = inclusive_scan(subDegreePrefixSumLable, subOffset, vertexArrSize, q);
    q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
        auto tid = item.get_global_id()[0];
        for (uint i = tid; i < vertexArrSize; i += N) {
            subOffset[i] -= subDegree[i];
        }
        });
    }).wait();
    //step5. get Offset
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
            auto tid = item.get_global_id()[0];
        for (SIZE_TYPE id = tid; id < vertexArrSize; id += N) {
            if (isActive[id] == 1) {
                offset[subIndex[id]] = subOffset[id];
            }
            }
            });
        }).wait();
    
    offset[numActiveVertices] = degree[subVertex[numActiveVertices - 1]] + offset[numActiveVertices - 1];
    //step6. get subEdge
    pthread_t* handles = new pthread_t[NUM_THREADS_h];
    threadParam_t* param = new threadParam_t[NUM_THREADS_h];
    for (int t_id = 0; t_id < NUM_THREADS_h; t_id++)
    {
        param[t_id].t_id = t_id;
        param[t_id].subVertex = subVertex;
        param[t_id].subEdge = subEdge;
        param[t_id].offset = offset;
        param[t_id].activeNodeNum = numActiveVertices;
        param[t_id].degree = degree;
        param[t_id].edgeArray = edgeArray;
        param[t_id].nodePointers = nodePointers;

        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
    }


        for (int t_id = 0; t_id < NUM_THREADS_h; t_id++)
            pthread_join(handles[t_id], NULL);
    free(isActiveDprefixLable, q);
    free(subDegreePrefixSumLable, q);
            
}

int main(int argc, char* argv[]) {
    default_selector selector;
    sycl::queue q(selector);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    int ch;
    string dataname;
    uint source;
    char* c = new char[6];
    c[0] = 'f'; c[1] = ':'; c[2] = 'r'; c[3] = ':'; c[4] = 'h'; c[5] = '\0';
    while ((ch = getopt(argc, argv, c)) != -1) {
        switch (ch) {
        case 'f':
            dataname = optarg;
            break;
        case 'r':
            source = atoll(optarg);
            break;
        case 'h':
            printf("8-byte edge BFS\n");
            printf("\t-f | input file name\n");
            printf("\t-r | BFS root \n");
            printf("\t-h | help message\n");
            return 0;
        case '?':
            break;
        default:
            break;
        }
    }
    string path = dataname;
    SIZE_TYPE vertexArrSize;
    EDGE_POINTER_TYPE edgeArrSize;
    EDGE_POINTER_TYPE* nodePointers;
    uint* edgeArray;
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(dataname, ios::in | ios::binary);
    infile.read((char*)&vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char*)&edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << "\n";
    nodePointers = (EDGE_POINTER_TYPE*)malloc(sizeof(EDGE_POINTER_TYPE)*vertexArrSize);
    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    edgeArray = (uint*)malloc(sizeof(uint)*edgeArrSize);
    infile.read((char*)edgeArray, sizeof(uint) * edgeArrSize);
    infile.close();
    auto endreadTime = chrono::steady_clock::now();
    auto readTime = chrono::duration_cast<chrono::milliseconds>(endreadTime - startTime).count();
    cout << "readfile Time: " << readTime << " ms" << "\n";

    uint* degree = (uint*)malloc(sizeof(uint)*vertexArrSize);
    uint* degreeD = malloc_device<uint>(vertexArrSize, q);

    uint* value = (uint*)malloc(sizeof(uint)*vertexArrSize);
    uint* valueD = malloc_device<uint>(vertexArrSize, q);

    bool* label = (bool*)malloc(sizeof(bool)*vertexArrSize);
    bool* labelD = malloc_device<bool>(vertexArrSize, q);
    q.memset(labelD, 0, vertexArrSize * sizeof(bool)).wait();

    bool* isActiveHost = (bool*)malloc(sizeof(bool)*vertexArrSize);
    bool* isActive = malloc_device<bool>(vertexArrSize, q);
    q.memset(isActive, 1, vertexArrSize * sizeof(bool)).wait();

    bool* isActivenew = malloc_device<bool>(vertexArrSize, q);
    q.memset(isActivenew, 0, vertexArrSize * sizeof(bool)).wait();

    auto startiniTime = chrono::steady_clock::now();

    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        value[i] = i;
        if (i + 1 < vertexArrSize) {
            degree[i] = nodePointers[i + 1] - nodePointers[i];
        }
        else {
            degree[i] = edgeArrSize - nodePointers[i];
        }
    }
    auto endiniTime = chrono::steady_clock::now();
    auto iniTime = chrono::duration_cast<chrono::milliseconds>(endiniTime - startiniTime).count();
    cout << "Initdata Time: " << iniTime << " ms\n";

    q.memcpy(degreeD, degree, sizeof(uint) * vertexArrSize).wait();
    q.memcpy(valueD, value, sizeof(uint) * vertexArrSize).wait();

    uint memTemp = 1024;
    ull memorys = (ull) 12 * memTemp * memTemp * memTemp;
    uint max_partition_size = 0.9 *(memorys - 8 * 4 * vertexArrSize) / sizeof(uint);
    EDGE_POINTER_TYPE* subVertex = malloc_shared<EDGE_POINTER_TYPE>(vertexArrSize, q);
    SIZE_TYPE* offset = malloc_shared<SIZE_TYPE>(vertexArrSize + 1, q);
    uint* subEdge = (uint*)malloc(sizeof(uint)*edgeArrSize);
    uint* d_subEdge = malloc_device<uint>(max_partition_size, q);
    uint* subIndex = malloc_device<uint>(vertexArrSize, q);
    uint* subDegree = malloc_device<uint>(vertexArrSize, q);
    SIZE_TYPE* subOffset = malloc_device<SIZE_TYPE>(vertexArrSize, q);
    q.memset(subIndex, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subDegree, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subOffset, 0 , sizeof(SIZE_TYPE)*vertexArrSize).wait();
    
    uint activenodes = vertexArrSize;
    Partitioner<EDGE_POINTER_TYPE> partitioner;
    uint testnodesnum = 0;
    auto genTime = 0;
    auto kernelTime = 0;
    auto calStart = chrono::steady_clock::now();
    while (activenodes > 0) {
        auto startgenTime = chrono::steady_clock::now();
        generateSubG(subIndex, subDegree, subOffset, isActive, q, vertexArrSize, activenodes, subVertex, degree, degreeD, offset, edgeArray, nodePointers, subEdge);
        auto endgenTime = chrono::steady_clock::now();
        auto genDuration = chrono::duration_cast<chrono::milliseconds>(endgenTime - startgenTime).count();
        genTime += genDuration;
        
        partitioner.partition(offset, activenodes, max_partition_size);
        for (int outi = 0; outi < partitioner.numPartitions; outi++)
        {
            uint subVertex_from = partitioner.fromNode[outi];
            q.memcpy(d_subEdge, subEdge + partitioner.fromEdge[outi], (partitioner.partitionEdgeSize[outi]) * sizeof(uint)).wait();
            uint edge_from = partitioner.fromEdge[outi];
            uint edge_size = partitioner.partitionEdgeSize[outi];
            uint offset_from = partitioner.fromNode[outi];
            ull newactivenodes = partitioner.partitionNodeSize[outi];
            auto kernelstartTime = chrono::steady_clock::now();
            q.submit([&](handler& h) {
                h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
                    auto tid = item.get_global_id()[0];
                    for (SIZE_TYPE id = tid; id < newactivenodes; id += N) {
                        auto vid = subVertex[subVertex_from + id];//overloadNodeList
                        uint sourceValue = valueD[vid];
                        //i is equal to edgeIndex
                        for (uint i = offset[offset_from + id]; i < offset[offset_from + id + 1]; i++) {
                            uint edgeindex = i - edge_from;    //thisFrom                                 
                            uint nbr = d_subEdge[edgeindex];//vertexId
                            if (sourceValue < valueD[nbr]) {
                                valueD[nbr] = sourceValue;
                                labelD[nbr] = true;
                                isActivenew[nbr] = 1;
                            }                                    
                        }
                    }
                    });
                }).wait();
            auto kernelendTime = chrono::steady_clock::now();
	        kernelTime += chrono::duration_cast<chrono::milliseconds>(kernelendTime - kernelstartTime).count();
        }
        auto kernelstartTime = chrono::steady_clock::now();
        activenodes = 0;
        q.memcpy(isActive, isActivenew, vertexArrSize * sizeof(bool)).wait();
        q.memcpy(isActiveHost, isActivenew, sizeof(bool) * vertexArrSize).wait();
        q.memset(isActivenew, 0, vertexArrSize * sizeof(bool)).wait();
        for (uint id = 0; id < vertexArrSize; id++) {
            if (isActiveHost[id]){
                activenodes++;
            }    
        }
        testnodesnum += activenodes;
        auto kernelendTime = chrono::steady_clock::now();
        auto kernelDuration = chrono::duration_cast<chrono::milliseconds>(kernelendTime - kernelstartTime).count();

	    kernelTime+=kernelDuration;
    }
    testnodesnum += vertexArrSize;
    auto calEnd = chrono::steady_clock::now();
    auto calDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - calStart).count();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - startTime).count();
    cout << "Generate subGraph Time: " << genTime << " ms" << "\n";
    std::cout << "Kernel Time: " << kernelTime << " ms\n";
            
    std::cout << "Calculate Time: " << calDuration << " ms\n";
    std::cout << "Total Time: " << totalDuration << " ms\n";

    // q.memcpy(label, labelD, sizeof(bool) * vertexArrSize).wait();
    // validation(vertexArrSize, label);
    
    free(nodePointers);
    free(edgeArray);
    free(degree);
    free(degreeD, q);
    free(value);
    free(valueD, q);
    free(label);
    free(labelD, q);
    free(isActiveHost);
    free(isActive, q);
    free(isActivenew, q);
    cout << "nodeSum : "<<testnodesnum<<"\n";
    return 0;
}
