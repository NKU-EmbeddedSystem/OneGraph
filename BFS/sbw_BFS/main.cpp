//trasform from BFS

#include <stdio.h>                  /* for EOF */
#include <string.h>                 /* for strchr() */
#include<CL/sycl.hpp>
//#include<omp.h>

using namespace sycl;

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
#include <ctime>


#define MYINFINITY 0xFFFFFFFF
using namespace std;

const int NUM_THREADS_h = 20;
const int NUM_THREADS = 1024 * 108;
static constexpr size_t N = 1024 * 108;
static constexpr size_t B = 1024;
static constexpr size_t sub_group_size = 16;
unsigned long long int dataTrans=0;

//----------------------------------------------- Graph.h转换 begin -----------------------------------

struct EdgeWithWeight {
    uint toNode;
    uint weight;
};

typedef uint EdgeType;
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

uint findbignode(uint last)
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


//----------------------------------------------- Graph.h转换 end -----------------------------------

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

    q.memcpy(subDegreePrefixSumLable, subDegree, sizeof(uint) * vertexArrSize).wait();
    dataTrans+=vertexArrSize * sizeof(uint)/1024/1024;

    

    subOffset = inclusive_scan(subDegreePrefixSumLable, subOffset, vertexArrSize, q);
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < vertexArrSize; i += N) {
                subOffset[i] -= subDegree[i];
            }
        });
    }).wait();

    //==============================================================================================
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
	int t_id;
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

    //step6.求subEdge
    for (uint i = t_id; i < activeNodeNum; i += NUM_THREADS_h) {
        uint v = subVertex[i];
        for (uint k1 = offset[i], k2 = nodePointers[v]; k1 < offset[i] + degree[v]; k1++, k2++) {
            subEdge[k1] = edgeArray[k2];
        }
    }
    pthread_exit(NULL);
    return 0; 
}

void runGPUbfs_subway(uint source, uint*& distance, const uint& m_INFINITY, sycl::queue& q, ofstream& out_res)
{
    auto startGPUInitTime = chrono::steady_clock::now();//For Time: start GPUInitTime
 
    unsigned long long int dataTrans=0;
    uint nodeSize = vertexArrSize;
    uint edgeSize = edgeArrSize;
    Partitioner partitioner;
    ull mem1024 = 1024;
    ull memorys = mem1024 * mem1024 * mem1024 * 12;// 12G mem
    std::cout<<"memorys: "<<memorys<<"\n";
    ull max_partition_size = 0.9 * (memorys - 8 * 4 * nodeSize) / sizeof(uint);//定义划分partition的临界大小
    std::cout << "max_partition_size   " << max_partition_size << "\n";
    uint* degree = new uint[nodeSize];
    uint* degreeD = malloc_device<uint>(nodeSize, q);
    bool* isActiveNodeList = new bool[nodeSize];
    bool* isActiveNodeListD = malloc_device<bool>(nodeSize, q);
    bool* nextActiveNodeListD = malloc_device<bool>(nodeSize, q);
    uint* value = new uint[nodeSize];
    uint* valueD = malloc_device<uint>(nodeSize, q);
    EDGE_POINTER_TYPE* subVertex = malloc_shared<EDGE_POINTER_TYPE>(nodeSize, q);
    SIZE_TYPE* offset = malloc_shared<SIZE_TYPE>(nodeSize + 1, q);
    EdgeType* subEdge = new EdgeType[edgeArrSize];
    EdgeType* d_subEdge = malloc_device<EdgeType>(max_partition_size, q);
    uint* subIndex = malloc_device<uint>(vertexArrSize, q);
    uint* subDegree = malloc_device<uint>(vertexArrSize, q);
    SIZE_TYPE* subOffset = malloc_device<SIZE_TYPE>(vertexArrSize, q);
    uint* isActiveDprefixLable = malloc_device<uint>(vertexArrSize, q);//用于前缀和生成subIndex
    SIZE_TYPE* subDegreePrefixSumLable = malloc_device<SIZE_TYPE>(vertexArrSize, q);//用于前缀和生成subVertex


    int max_degree = 0;
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
    for (int i = 0; i < nodeSize; i++) {
        isActiveNodeList[i] = false;
        value[i] = UINT_MAX;
    }
    isActiveNodeList[source] = true;
    value[source] = 1;
    int iteration = 0;
    ull activeNodeNum = 1;

    q.memset(subIndex, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subDegree, 0 , sizeof(uint)*vertexArrSize).wait();
    q.memset(subOffset, 0 , sizeof(SIZE_TYPE)*vertexArrSize).wait();
    q.memcpy(isActiveNodeListD, isActiveNodeList, sizeof(bool) * nodeSize).wait();
    dataTrans +=sizeof(bool) * nodeSize/1024/1024;
    q.memset(nextActiveNodeListD, 0 , sizeof(bool)*nodeSize).wait();
    q.memcpy(degreeD, degree, sizeof(uint) * nodeSize).wait();
    dataTrans +=sizeof(uint) * nodeSize/1024/1024;
    q.memcpy(valueD, value, sizeof(uint) * nodeSize).wait();
    dataTrans +=sizeof(uint) * nodeSize/1024/1024;


    auto endGPUInitTime = chrono::steady_clock::now();//For Time: end GPUInitTime
    auto GPUInitTime = chrono::duration_cast<chrono::milliseconds>(endGPUInitTime - startGPUInitTime).count();
    cout << "GPUInitTime =  " << GPUInitTime << " ms" << "\n";
    out_res << "GPUInitTime =  " << GPUInitTime << " ms" << "\n";


    auto GPUCalTime = 0;
    auto SubGraphTime = 0;


    while (true) {
        if (activeNodeNum <= 0) {
            break;
        }

        auto startSubGraphTime = chrono::steady_clock::now();//For Time: start SubGraphTime

        generateSubG(subIndex, subDegree, subOffset, isActiveNodeListD, q, nodeSize, activeNodeNum, subVertex, 
        degreeD, offset, edgeArray, nodePointers,isActiveDprefixLable, subDegreePrefixSumLable);
        offset[activeNodeNum] = degree[subVertex[activeNodeNum - 1]] + offset[activeNodeNum - 1];
        pthread_t* handles = new pthread_t[NUM_THREADS_h];// 创建对应的 Handle
        threadParam_t* param = new threadParam_t[NUM_THREADS_h];// 创建对应的线程数据结构
        for (int t_id = 0; t_id < NUM_THREADS_h; t_id++){
            param[t_id].t_id = t_id;
            param[t_id].subVertex = subVertex;
            param[t_id].subEdge = subEdge;
            param[t_id].offset = offset;
            param[t_id].activeNodeNum = activeNodeNum;
            param[t_id].degree = degree;

            pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
        }
        for (int t_id = 0; t_id < NUM_THREADS_h; t_id++){
            pthread_join(handles[t_id], NULL);
        }
        partitioner.partition(offset, activeNodeNum, max_partition_size);

        auto endSubGraphTime = chrono::steady_clock::now();//For Time: end SubGraphTime
        SubGraphTime+=chrono::duration_cast<chrono::milliseconds>(endSubGraphTime - startSubGraphTime).count();


        auto startGPUCalTime = chrono::steady_clock::now();//For Time: start GPUCalTime

        for (int myi = 0; myi < partitioner.numPartitions; myi++){
            
            q.memcpy(d_subEdge, subEdge + partitioner.fromEdge[myi], (partitioner.partitionEdgeSize[myi]) * sizeof(EdgeType)).wait();
            dataTrans += (partitioner.partitionEdgeSize[myi]) * sizeof(EdgeType) / 1024 / 1024;


            unsigned int from = partitioner.fromNode[myi];//定义该partition在子图中的开始节点序号
            unsigned int partitionNodeSize = partitioner.partitionNodeSize[myi];//定义该partition在子图中cover的节点数
            unsigned int numPartitionedEdges = partitioner.fromEdge[myi];//定义该partition在子图中cover的边数

            q.parallel_for(sycl::range(NUM_THREADS), [=](sycl::id<1> ind) {
                for (int index = ind; index < partitionNodeSize; index += NUM_THREADS){
                    uint nodeIndex = subVertex[from + index];//获得原图中该节点的下标
                    uint thisFrom = offset[from + index] - numPartitionedEdges;//获得该节点对应的第一条边在partition下的边集中的下标
                    uint sourceValue = valueD[nodeIndex];//从原图的valueD数组中获取当前节点的值
                    uint finalValue;
                    for (uint i = thisFrom; i < thisFrom + degreeD[nodeIndex]; i++) {
                        finalValue = sourceValue + 1;
                        EdgeType vertexId = d_subEdge[i];//获取该边另一端的节点序号
                        if (finalValue < valueD[vertexId]) {
                            valueD[vertexId] = finalValue;
                            nextActiveNodeListD[vertexId] = true;//设置为活跃节点
                        }
                    }
                }
            }).wait();//原q.parallel_for

        }

        q.memcpy(isActiveNodeList, nextActiveNodeListD, nodeSize * sizeof(bool)).wait();
        dataTrans+=nodeSize * sizeof(bool)/1024/1024;
        q.memcpy(isActiveNodeListD, isActiveNodeList, nodeSize * sizeof(bool)).wait();
        dataTrans+=nodeSize * sizeof(bool)/1024/1024;
        q.memset(nextActiveNodeListD, 0,  nodeSize* sizeof(bool)).wait();

        activeNodeNum = 0;
        for (int ii = 0; ii < nodeSize; ii++) {
            if (isActiveNodeList[ii]) {
                activeNodeNum++;
            }
        }
        // cout<<"iteration: "<<iteration<<";  activeNodeNum: "<<activeNodeNum<<"\n";

        iteration++;
        gpu_iteration = iteration;

        auto endGPUCalTime = chrono::steady_clock::now();//For Time: end GPUCalTime
        GPUCalTime+=chrono::duration_cast<chrono::milliseconds>(endGPUCalTime - startGPUCalTime).count();
    }

    q.memcpy(value, valueD, sizeof(uint) * nodeSize).wait();//将valueD拷回cpu端，方便展示和核对
    for (int i = 0; i < nodeSize; i++) {
        distance[i] = value[i];
    }

    //释放数组空间
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

    cout << "SubGraphTime: " << SubGraphTime << "ms" << "\n";
    out_res << "SubGraphTime: " << SubGraphTime << "ms" << "\n";
    cout << "GPUCalTime: " << GPUCalTime << "ms" << "\n";
    out_res << "GPUCalTime: " << GPUCalTime << "ms" << "\n";

    cout << "move overload size : " << dataTrans << "MB \n";
    out_res << "move overload size : " << dataTrans << "MB \n";

}



int testTimes = 5;
int main(int argc, char* argv[])
{
    const uint m_INFINITY = 0xFFFFFFFF;
    int ch;
    string dataname;
    uint source;
    char* c = new char[6];
    c[0] = 'f'; c[1] = ':'; c[2] = 'r'; c[3] = ':'; c[4] = 'h'; c[5] = '\0';
    //while ((ch = getopt(argc, argv, "f:r:h")) != -1) {
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

    ofstream out_res;//将执行时间结果输出到同目录下的res.txt中
    out_res.open("./res.txt", ios::app);
    if (!out_res)cout << "out_res open failed!" << "\n";

    //读入数据
    readDataFromFile(filename, q);
    out_res<<"\n\n\n";
    out_res<<filename<<"\n";
    out_res << "vertexArrSize = " << vertexArrSize << " edgeArrSize = " << edgeArrSize << "\n";
    std::cout << "after readDataFromFile" << "\n";


    //设置source
    uint *src=new uint[testTimes];
    uint ttt=0;
    for(int i=0;i<testTimes;i++)
    {
        src[i]=findbignode(ttt);
        ttt = src[i];

        // srand(time(0));
        // src[i] = rand() % vertexArrSize;
        // cout<< "node "<<i<<" = "<<src[i]<<"\n";
    }

    //subway GPU start
    for (int testIndex = 0; testIndex < testTimes; testIndex++) 
    {
        cout<<"----------testIndex = "<<testIndex<<"-----------\n";
        out_res<<"----------testIndex = "<<testIndex<<"-----------\n";
        source = src[testIndex];
        uint* D_distance = new uint[vertexArrSize];
        runGPUbfs_subway(source, D_distance, m_INFINITY, q, out_res);

        uint nodesum = 0;
        for (uint i = 0; i < vertexArrSize; i++) {
            if (D_distance[i] != m_INFINITY)
                nodesum++;
        }
        cout << "nodesum: " << nodesum << "\n";
        out_res << "nodesum: " << nodesum << "\n";
        delete[] D_distance;
    }

    delete []src;
    return 0;
}
