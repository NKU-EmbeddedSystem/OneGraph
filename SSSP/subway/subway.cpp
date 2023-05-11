#include <stdio.h>                  /* for EOF */
#include <string.h>                 /* for strchr() */
#include<CL/sycl.hpp>

using namespace sycl;

bool debug1006 = false;
int gpu_iteration = 0;

/* static (global) variables that are specified as exported by getopt() */
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
//======================================================================================================================================
#include"global.h"
#include"partition.h"
#include <pthread.h>

#define MYINFINITY 0xFFFFFFFF
using namespace std;

const int NUM_THREADS_h = 20;
const int NUM_THREADS = 1024 * 108;
static constexpr size_t N = 1024 * 108;
static constexpr size_t B = 1024;
static constexpr size_t sub_group_size = 16;

ull H2D = 0;
ull D2H = 0;

//=====================================================================================================
//Graph.h conversion begin

typedef EdgeWithWeight EdgeType;
SIZE_TYPE vertexArrSize;
EDGE_POINTER_TYPE edgeArrSize;
EDGE_POINTER_TYPE* nodePointers;
EdgeType* edgeArray;

void readDataFromFile(const string& fileName, sycl::queue& q)
{
    cout << "readDataFromFile" << "\n";
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char*)&vertexArrSize, sizeof(EDGE_POINTER_TYPE));
    infile.read((char*)&edgeArrSize, sizeof(EDGE_POINTER_TYPE));
    cout << "vertex num: " << vertexArrSize << " edge num: " << edgeArrSize << "\n";
    nodePointers = new EDGE_POINTER_TYPE[vertexArrSize];

    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    edgeArray = new EdgeType[edgeArrSize];

    infile.read((char*)edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << "\n";
}
//Graph.h conversion end
//=====================================================================================================

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

void generateSubG(uint* subIndex, uint* subDegree, SIZE_TYPE* subOffset, bool isActive[], sycl::queue q, SIZE_TYPE vertexArrSize,
    SIZE_TYPE numActiveVertices, EDGE_POINTER_TYPE subVertex[], SIZE_TYPE degree[],
    SIZE_TYPE offset[], EdgeType* edgeArray, EDGE_POINTER_TYPE* nodePointers, uint* isActiveDprefixLable, SIZE_TYPE* subDegreePrefixSumLable){
    //==============================================================================================
    //step1 get subIntex

    //copy isActive to isActiveDprefixLable
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                isActiveDprefixLable[i] = (uint)isActive[i];
            }
        });
    }).wait();
    //call inclusive_scan(.) to compute prefix summary
    subIndex = inclusive_scan(isActiveDprefixLable, subIndex, vertexArrSize, q);
    //fix the result of prefix summary
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                subIndex[i] -= isActive[i];
            }
        });
    }).wait();

    //==============================================================================================
    //step2、3 get subVertex and subDegree
    //the 2 step can be put into the same parallel_for, for they don't influence each other

    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (SIZE_TYPE id = tid; id < vertexArrSize; id += N) {
                if (id < vertexArrSize) {
                    if (isActive[id] == 1) {
                        subVertex[subIndex[id]] = id;
                        subDegree[id] = degree[id];
                    }
                    else {
                        subDegree[id] = 0;
                    }
                }
            }
        });
    }).wait();

    //==============================================================================================
    //step4 get subOffset

    //copy subDegree to subDegreePrefixSumLable, the latter is just a 'tool' array
    q.memcpy(subDegreePrefixSumLable, subDegree, sizeof(uint) * vertexArrSize).wait();
    //call inclusive_scan(.) to compute prefix summary
    subOffset = inclusive_scan(subDegreePrefixSumLable, subOffset, vertexArrSize, q);
    //fix the prefix summary
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                subOffset[i] -= subDegree[i];
            }
        });
    }).wait();

    //==============================================================================================
    //step5 get offset
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id();
            for (uint id = tid; id < vertexArrSize; id += N) {
                if (isActive[id] == 1) {
                    offset[subIndex[id]] = subOffset[id];
                }
            }
        });
    }).wait();
}

struct threadParam_t
{
	int t_id; //thread id
    EDGE_POINTER_TYPE* subVertex;
    EdgeType* subEdge;
    SIZE_TYPE* offset;
    ull activeNodeNum;
    uint* degree;
};

void* threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
	int t_id = p -> t_id;

    EDGE_POINTER_TYPE* subVertex = p -> subVertex;
    EdgeType* subEdge = p -> subEdge;
    SIZE_TYPE* offset = p -> offset;
    ull activeNodeNum = p -> activeNodeNum;
    uint* degree = p -> degree;

    //step6 get subEdge
    for (uint i = t_id; i < activeNodeNum; i += NUM_THREADS_h) {
        uint v = subVertex[i];
        for (uint k1 = offset[i], k2 = nodePointers[v]; k1 < offset[i] + degree[v]; k1++, k2++) {
            subEdge[k1] = edgeArray[k2];
            if(debug1006 && gpu_iteration == 0)std::cout<<"edgeArray[k2].toNode = "<<edgeArray[k2].toNode;
            if(debug1006 && gpu_iteration == 0)std::cout<<" edgeArray[k2].weight = "<<edgeArray[k2].weight<<"\n";
        }
    }
    pthread_exit(NULL);
    return 0; 
}

void runGPUbfs_subway4(uint source, const uint& m_INFINITY, sycl::queue& q)
{
    auto startTime = chrono::steady_clock::now();
   //===============================================================================
    //variable definition begin
    auto total_init_time_start = chrono::steady_clock::now();
    //define the number of nodes and edges in the graph
    uint nodeSize = vertexArrSize;
    uint edgeSize = edgeArrSize;
    //define variables related to 'partition'(the graph partition)
    Partitioner partitioner;
    uint k = 1024;
    unsigned long long memorys = k * k * 12;
    memorys *= k;
    cout<<"memorys = "<<memorys<<"\n";
    uint reduceMem = 12 * sizeof(uint) * (long) nodeSize;
    cout<<"reduceMem = "<<reduceMem<<"\n";cout<<"sizeof(EdgeType) = "<<sizeof(EdgeType)<<"\n";
    ull total_gpu_size = (memorys - reduceMem) / sizeof(EdgeType);
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
    float adviseRate = 0.0f;
    cout << "adviseRate " << adviseRate << "\n";
    if (adviseRate > 0) {
        adviseK = adviseRate;
    }

    cout<<"total_gpu_size = "<<total_gpu_size<<"\n";
    unsigned long long max_partition_size = adviseK * total_gpu_size;
    if (max_partition_size > edgeArrSize) {
        max_partition_size = edgeArrSize;
    }
    if (max_partition_size > UINT_MAX) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = UINT_MAX;
    }
    std::cout << "max_partition_size   " << max_partition_size << "\n";
    //define degree array, active node array and value array
    uint* degree = new uint[nodeSize];
    uint* degreeD = malloc_device<uint>(nodeSize, q);
    bool* isActiveNodeList = new bool[nodeSize];
    bool* isActiveNodeListD = malloc_device<bool>(nodeSize, q);
    bool* nextActiveNodeListD = malloc_device<bool>(nodeSize, q);
    uint* value = new uint[nodeSize];
    uint* valueD = malloc_device<uint>(nodeSize, q);
    //define 'tool' array to genSubGraph
    EDGE_POINTER_TYPE* subVertex = malloc_device<EDGE_POINTER_TYPE>(nodeSize, q);
    EDGE_POINTER_TYPE* subVertexC = new EDGE_POINTER_TYPE[nodeSize];
    SIZE_TYPE* offset = malloc_device<SIZE_TYPE>(nodeSize + 1, q);
    SIZE_TYPE* offsetC = new SIZE_TYPE[nodeSize + 1];
    EdgeType* subEdge = new EdgeType[edgeArrSize];
    EdgeType* d_subEdge = malloc_device<EdgeType>(max_partition_size, q);
    uint* subIndex = malloc_device<uint>(vertexArrSize, q);
    uint* subDegree = malloc_device<uint>(vertexArrSize, q);
    SIZE_TYPE* subOffset = malloc_device<SIZE_TYPE>(vertexArrSize, q);
    uint* isActiveDprefixLable = malloc_device<uint>(vertexArrSize, q);//for generating subIndex by prefix summary
    SIZE_TYPE* subDegreePrefixSumLable = malloc_device<SIZE_TYPE>(vertexArrSize, q);//for generating subVertex by prefix summary

    //variable definition end
    //===============================================================================
    //===============================================================================
    //variable initialize begin

    int max_degree = 0;
    //compute degrees of each node and the max degree
    //to prevent the partition threshold value be less than the max degree
    for (uint i = 0; i < nodeSize; i++)
    {
        if (i + 1 < nodeSize) {
            degree[i] = nodePointers[i + 1] - nodePointers[i];
        }
        else {
            degree[i] = edgeSize - nodePointers[i];
        }
        if (degree[i] > max_degree) {
            max_degree = degree[i];
        }
    }
    auto total_init_time_end = chrono::steady_clock::now();
    auto total_init_time_duration = chrono::duration_cast<chrono::milliseconds>(total_init_time_end - total_init_time_start).count();
    cout<<"total_init_time_duration = "<<total_init_time_duration<<"\n";
    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
    unsigned long long data_transport = 0;
    auto test_init_time_start = chrono::steady_clock::now();
    for (int i = 0; i < nodeSize; i++) {
        isActiveNodeList[i] = false;
        value[i] = UINT_MAX;
    }
    //initialize with 'source'
    isActiveNodeList[source] = true;
    value[source] = 1;
    int iteration = 0;
    ull activeNodeNum = 1;
    //initialize 'tool' arrays to genSubGraph
    q.memset(subIndex, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subDegree, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subOffset, 0 , sizeof(SIZE_TYPE)*vertexArrSize).wait();
    //initialize active nodes, degree and value arrays
    q.memcpy(isActiveNodeListD, isActiveNodeList, sizeof(bool) * nodeSize).wait();
    H2D += sizeof(bool) * nodeSize / 1024;
    q.memset(nextActiveNodeListD, 0 , sizeof(bool)*nodeSize).wait();
    q.memcpy(degreeD, degree, sizeof(uint) * nodeSize).wait();
    H2D += sizeof(uint) * nodeSize / 1024;
    q.memcpy(valueD, value, sizeof(uint) * nodeSize).wait();
    H2D += sizeof(uint) * nodeSize / 1024;

    //variable initialize end
    //===============================================================================
    
    auto test_init_time_end = chrono::steady_clock::now();
    auto test_init_time_duration = chrono::duration_cast<chrono::milliseconds>(test_init_time_end - test_init_time_start).count();
    cout<<"test_init_time_duration = "<<test_init_time_duration<<"\n";
    //===============================================================================
    //while, the outermost loop begin

    uint64_t genSubG_duration = 0;
    uint64_t transfer_duration = 0;
    uint64_t calcu_duration = 0;
    uint64_t reset_duration = 0;

    uint activeNodeSum = 0;
    data_transport += degree[source] * sizeof(EdgeWithWeight);

    while (true) {

        //define the condition to break
        if (activeNodeNum <= 0) {
            break;
        }
        else{
            activeNodeSum += activeNodeNum;
            data_transport += activeNodeNum * sizeof(uint);
        }
        //===============================================================================
        //genSubGraph begin
        auto genSubG_begin = chrono::steady_clock::now();
        //genSubGraph step1~5（offset->sub_nodePointers, subEdge->sub_Edges）
        generateSubG(subIndex, subDegree, subOffset, isActiveNodeListD, q, nodeSize, activeNodeNum, subVertex, 
        degreeD, offset, edgeArray, nodePointers,isActiveDprefixLable, subDegreePrefixSumLable);
        q.memcpy(subVertexC, subVertex, sizeof(EDGE_POINTER_TYPE) * nodeSize).wait();
        D2H += sizeof(EDGE_POINTER_TYPE) * nodeSize / 1024;
        q.memcpy(offsetC, offset, sizeof(SIZE_TYPE) * (nodeSize + 1)).wait();
        D2H += sizeof(SIZE_TYPE) * (nodeSize + 1) / 1024;
        offsetC[activeNodeNum] = degree[subVertexC[activeNodeNum - 1]] + offsetC[activeNodeNum - 1];
        auto genSubG_end = chrono::steady_clock::now();
        genSubG_duration += chrono::duration_cast<chrono::milliseconds>(genSubG_end - genSubG_begin).count();
        auto transfer_begin = chrono::steady_clock::now();
        //genSubGraph step6
        pthread_t* handles = new pthread_t[NUM_THREADS_h];// create corresponding Handle
        threadParam_t* param = new threadParam_t[NUM_THREADS_h];// create corresponding thread data structure
        for (int t_id = 0; t_id < NUM_THREADS_h; t_id++){
            param[t_id].t_id = t_id;
            param[t_id].subVertex = subVertexC;
            param[t_id].subEdge = subEdge;
            param[t_id].offset = offsetC;
            param[t_id].activeNodeNum = activeNodeNum;
            param[t_id].degree = degree;

            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS_h; t_id++){
            pthread_join(handles[t_id], NULL);
        }
        if(debug1006 && iteration == 0){
            std::cout<<"check subEdge after thread join\n";
            for(int i = 0; i < 40; i ++){
                std::cout<<subEdge[i].toNode<<" ";
                std::cout<<subEdge[i].weight<<"\n";
            }
            std::cout<<"\n";
        }
        auto transfer_end = chrono::steady_clock::now();
        transfer_duration += chrono::duration_cast<chrono::milliseconds>(transfer_end - transfer_begin).count();
        //genSubGraph end
        //===============================================================================

        partitioner.partition(offsetC, activeNodeNum, max_partition_size);//tile the graph

        //===============================================================================
        //traverse all the partition inner loop begin
        auto calcu_begin = chrono::steady_clock::now();
        if(debug1006 && iteration == 0)std::cout<<"partitioner.numPartitions = "<<partitioner.numPartitions<<"\n";
        for (int myi = 0; myi < partitioner.numPartitions; myi++){
            //debug//cout << "fromNode:" << partitioner.fromNode[myi] << "  " << "fromEdge:" << partitioner.fromEdge[myi] << "  " << "partitionEdgeSize:" << partitioner.partitionEdgeSize[myi] << "  " << "numPartitions:" << partitioner.numPartitions << "\n";
            
            //copy all the edges likely to be used by this partition
            q.memcpy(d_subEdge, subEdge + partitioner.fromEdge[myi], (partitioner.partitionEdgeSize[myi]) * sizeof(EdgeType)).wait();
            H2D += (partitioner.partitionEdgeSize[myi]) * sizeof(EdgeType) / 1024;
            //define common values for partition to simplify code
            unsigned int from = partitioner.fromNode[myi];//define the number of the start node for this partition in subGraph
            unsigned int partitionNodeSize = partitioner.partitionNodeSize[myi];//difine number of nodes the partition covers in subGraph
            unsigned int numPartitionedEdges = partitioner.fromEdge[myi];//define number of edges the partition covers in subGraph

            //===============================================================================
            //traverse the nodes of this partition inner inner loop begin
            q.parallel_for(sycl::range(NUM_THREADS), [=](sycl::id<1> ind) {
                for (int index = ind; index < partitionNodeSize; index += NUM_THREADS){
                    //for(int index = 0; index < partitionNodeSize; index ++){ //serial for
                    uint nodeIndex = subVertex[from + index];//get the node index in original graph
                    uint thisFrom = offset[from + index] - numPartitionedEdges;//get the index in partition edgeSet of the first edge corresponding to the node
                    //debug//std::cout<<"thisFrom = "<<thisFrom<<"\n";
                    uint sourceValue = valueD[nodeIndex];//get the value of the current node from the valueD array of original graph
                    //debug//std::cout<<"sourceValue = "<<sourceValue<<"\n";
                    uint finalValue;
                    //traverse all the outDegree of the current node(nodeIndex)
                    for (uint i = thisFrom; i < thisFrom + degreeD[nodeIndex]; i++) {
                        finalValue = sourceValue + d_subEdge[i].weight;
                        uint vertexId = d_subEdge[i].toNode;//get the number of the node in another end of the edge
                        if (finalValue < valueD[vertexId]) {
                            //here atomic
                            auto min_atomic = atomic_ref<uint, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space>(valueD[vertexId]);
                            min_atomic.fetch_min(finalValue);
                            nextActiveNodeListD[vertexId] = true;//set as active node
                        }
                    }
                    //}//serial for
                }//origin: for...NUM_THREADS
            }).wait();//origin: q.parallel_for

            //traverse the nodes of this partition inner inner loop end
            //===============================================================================
        }
        auto calcu_end = chrono::steady_clock::now();
        calcu_duration += chrono::duration_cast<chrono::milliseconds>(calcu_end - calcu_begin).count();
        //traverse all the partition inner loop end
        //===============================================================================
        auto reset_begin = chrono::steady_clock::now();
        //prepare arrays of active nodes for next iteration
        q.memcpy(isActiveNodeList, nextActiveNodeListD, nodeSize * sizeof(bool)).wait();
        D2H += nodeSize * sizeof(bool) / 1024;
        q.memcpy(isActiveNodeListD, isActiveNodeList, nodeSize * sizeof(bool)).wait();
        H2D += nodeSize * sizeof(bool) / 1024;        
        q.memset(nextActiveNodeListD, 0,  nodeSize* sizeof(bool)).wait();

        //count active nodes
        // compute number of edges for next iteration
        activeNodeNum = 0;
        for (int ii = 0; ii < nodeSize; ii++) {
            if (isActiveNodeList[ii]) {
                activeNodeNum++;
                data_transport += degree[ii] * sizeof(EdgeWithWeight);
            }
        }
        if(debug1006 && iteration == 0)std::cout<<"activeNodeNum = "<<activeNodeNum<<"\n";

        iteration++;
        gpu_iteration = iteration;

        auto reset_end = chrono::steady_clock::now();
        reset_duration += chrono::duration_cast<chrono::milliseconds>(reset_end - reset_begin).count();

        if(debug1006 && iteration == 1){
            std::cout<<value[0]<<"\n";
            std::cout<<value[1]<<"\n";
            std::cout<<value[2]<<"\n";
            std::cout<<"...\n";
            std::cout<<value[nodeSize - 1]<<"\n";
            std::cout<<"iteration = "<<iteration<<"\n";
        }

    }

    //while, the outermost loop end
    //===============================================================================
    
    cout<<"genSubG_duration = "<<genSubG_duration<<"\n";
    cout<<"transfer_duration = "<<transfer_duration<<"\n";
    cout<<"calcu_duration = "<<calcu_duration<<"\n";
    cout<<"reset_duration = "<<reset_duration<<"\n";

    cout<<"activeNodeSum = "<<activeNodeSum<<"\n";
    cout<<"H2D = "<<H2D<<"\n";
    cout<<"D2H = "<<D2H<<"\n";

    auto test_reset_begin = chrono::steady_clock::now();
    q.memcpy(value, valueD, sizeof(uint) * nodeSize).wait();//copy valueD to CPU side for easy display and verification

    int nodesum = 0;
    for (uint i = 0; i < vertexArrSize; i++) {
        if (value[i] != m_INFINITY)
            nodesum++;
    }
    cout << "nodesum: " << nodesum << "\n";

    std::cout<<value[0]<<"\n";
    std::cout<<value[1]<<"\n";
    std::cout<<value[2]<<"\n";
    std::cout<<"...\n";
    std::cout<<value[nodeSize - 1]<<"\n";
    std::cout<<"iteration = "<<iteration<<"\n";
    //debug//cout << "end print value" << "\n";
    for (uint i = source+1; i < vertexArrSize - 1; i++)
    {
        uint degree = nodePointers[i + 1] - nodePointers[i];
        if (degree > 3000)
        {
        cout << "find a big node1:" << i << "\n";
        source = i;
        cout<<"has break;\n";
        break;
        }
    }
    cout << "find a big node2:" << source << "\n";
    
    auto test_reset_end = chrono::steady_clock::now();
    auto test_reset_duration = chrono::duration_cast<chrono::milliseconds>(test_reset_end - test_reset_begin).count();
    cout<<"test_reset_duration = "<<test_reset_duration<<"\n";
    }

    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "SSSP_GPU: " << duration << "ms" << "\n";

    //free arrays
    free(degreeD, q);
    free(isActiveNodeListD, q);
    free(nextActiveNodeListD, q);
    free(d_subEdge, q);
    free(valueD, q);
    free(subIndex, q);
    free(subDegree, q);
    free(subOffset, q);
    free(isActiveDprefixLable, q);
    free(subDegreePrefixSumLable, q);

}

int main(int argc, char* argv[])
{
    const uint m_INFINITY = 0xFFFFFFFF;
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

    sycl::queue q;
    string filename = dataname;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    //readin data
    readDataFromFile(filename, q);

    std::cout << "after readDataFromFile" << "\n";
    //=====================================================
    //subway GPU start
    runGPUbfs_subway4(source, m_INFINITY, q);
    //subway CPU end
    //=====================================================

    return 0;
}