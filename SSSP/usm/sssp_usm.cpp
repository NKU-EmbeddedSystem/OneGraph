#include <algorithm>
#include <stdio.h>                  /* for EOF */
#include <string.h>                 /* for strchr() */
#include<CL/sycl.hpp>
#include<atomic_ref.hpp>

using namespace sycl;

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
//===============================================================================================================
#include"Graph.h"

using namespace std;

bool debug4_on = false;
const int N =  1024*108;
const int B = 1024;
int NUM_THREADS0 = N;
#define MYINFINITY 0xFFFFFFFF

namespace {

    template <typename T>
    using local_atomic_ref = sycl::ext::oneapi::atomic_ref<
        T,
        sycl::ext::oneapi::memory_order::relaxed,
        sycl::ext::oneapi::memory_scope::work_group,
        access::address_space::local_space>;

    template <typename T>
    using global_atomic_ref = sycl::ext::oneapi::atomic_ref<
        T,
        sycl::ext::oneapi::memory_order::relaxed,
        sycl::ext::oneapi::memory_scope::system,
        access::address_space::global_space>;
}

void runGPUsssp3(Graph<EdgeWithWeight>& G, uint source, uint*& distance, const uint& m_INFINITY, sycl::queue& q)
{
    auto startTime = chrono::steady_clock::now();
    auto total_init_time_start = chrono::steady_clock::now();

    uint nodeSize = G.vertexArrSize;
    uint edgeSize = G.edgeArrSize;
    uint* degreeD = malloc_shared<uint>(nodeSize, q);
    bool* isActiveNodeListD = malloc_shared<bool>(nodeSize, q);

    for (uint i = 0; i < nodeSize; i++) {
        isActiveNodeListD[i] = false;
        if (i + 1 < nodeSize) {
            degreeD[i] = G.nodePointers[i + 1] - G.nodePointers[i];
        }
        else {
            degreeD[i] = edgeSize - G.nodePointers[i];
        }
    }

    bool* nextActiveNodeListD = malloc_shared<bool>(nodeSize, q);
    EdgeWithWeight* d_edgeArray = malloc_shared<EdgeWithWeight>(edgeSize, q);
    EDGE_POINTER_TYPE* d_nodePointers = malloc_shared<EDGE_POINTER_TYPE>(nodeSize, q);;
    uint* valueD = malloc_shared<uint>(nodeSize, q);

    q.memcpy(d_nodePointers, G.nodePointers, sizeof(EDGE_POINTER_TYPE) * nodeSize).wait();
    q.memcpy(d_edgeArray, G.edgeArray, sizeof(EdgeWithWeight) * edgeSize).wait();

    auto total_init_time_end = chrono::steady_clock::now();
    auto total_init_time_duration = chrono::duration_cast<chrono::milliseconds>(total_init_time_end - total_init_time_start).count();
    cout<<"total_init_time_duration = "<<total_init_time_duration<<"\n";

    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
    auto test_init_time_start = chrono::steady_clock::now();
    for (int i = 0; i < nodeSize; i++) {
        nextActiveNodeListD[i] = 0;
        isActiveNodeListD[i] = false;
        valueD[i] = UINT_MAX;
    }

    isActiveNodeListD[source] = true;
    valueD[source] = 1;
    uint activeSum = 0;
    int iteration = 0;

    uint activeNodeNum = 1;
    auto test_init_time_end = chrono::steady_clock::now();
    auto test_init_time_duration = chrono::duration_cast<chrono::milliseconds>(test_init_time_end - test_init_time_start).count();
    cout<<"test_init_time_duration = "<<test_init_time_duration<<"\n";
    uint64_t calcu_duration = 0;
    auto calcu_begin = chrono::steady_clock::now();
    while (true) {
        if (activeNodeNum <= 0) {
            break;
        }
        else {
            activeSum += activeNodeNum;
        }
    int NUM_THREADS = NUM_THREADS0;
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N,B), [=](nd_item<1> item) {
            //core loop begin
            int ind = item.get_global_id(0);
            for (int index = ind; index < nodeSize; index += NUM_THREADS) {
                uint nodeIndex = 0 + index;
                if (isActiveNodeListD[nodeIndex]) {
                    uint edgeIndex = d_nodePointers[nodeIndex] - 0;
                    uint sourceValue = valueD[nodeIndex];
                    uint finalValue;
                    //below traverse all the outDegree of the ndoe(nodeIndex)
                    for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                        //each 'edgeListD[i]' is connected with current node by an edge
                        finalValue = sourceValue + d_edgeArray[i].weight;
                        uint vertexId = d_edgeArray[i].toNode;
                        if (finalValue < valueD[vertexId]) {
                            auto min_atomic = atomic_ref<uint, sycl::memory_order::relaxed, memory_scope::device, 
                                                            access::address_space::global_space>(valueD[vertexId]);
                            min_atomic.fetch_min(finalValue);
                            nextActiveNodeListD[vertexId] = true;
                        }
                    }
                }
            }
            //core loop end
        });
    }).wait();
        q.memcpy(isActiveNodeListD, nextActiveNodeListD, nodeSize * sizeof(bool)).wait();
        activeNodeNum = 0;
        for (int ii = 0; ii < nodeSize; ii++) {
            if (isActiveNodeListD[ii]) {
                activeNodeNum++;
            }
            nextActiveNodeListD[ii] = 0;
        }
        iteration++;
    }
    auto calcu_end = chrono::steady_clock::now();
    calcu_duration += chrono::duration_cast<chrono::milliseconds>(calcu_end - calcu_begin).count();
    cout<<"calcu_duration = "<<calcu_duration<<"\n";
    //==========================================================================
    auto test_reset_begin = chrono::steady_clock::now();
    cout << "start print valueD" << "\n";
    for (int i = 0; i < nodeSize; i++) {
        if(debug4_on)cout << valueD[i] << "\n";
        distance[i] = valueD[i];
    }
    if(debug4_on == false){
        cout<<valueD[0]<<"\n";
        cout<<valueD[1]<<"\n";
        cout<<valueD[2]<<"\n";
        cout<<"...\n";
        cout<<valueD[nodeSize - 1]<<"\n";
    }
    cout << "end print valueD" << "\n";

    for (uint i = source+1; i < nodeSize - 1; i++)
    {
        uint degree = G.nodePointers[i + 1] - G.nodePointers[i];
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

    free(degreeD, q);
    free(isActiveNodeListD, q);
    free(nextActiveNodeListD, q);
    free(d_edgeArray, q);
    free(d_nodePointers, q);
    free(valueD, q);

    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "ssspGPU: " << duration << "ms" << "\n";

}


int main(int argc, char* argv[])
{
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
    
    Graph<EdgeWithWeight> G;
    sycl::gpu_selector gs;
    sycl::queue q;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    string filename = dataname;
    
    G.readDataFromFile(filename, q);
    cout << "nodePointers" << "\n";
    cout << "edgeArray" << "\n";
    cout << "after readDataFromFile" << "\n";
    auto startCPUInitTime = chrono::steady_clock::now();
    const uint m_INFINITY = 0xFFFFFFFF;    

    auto startGPUInitTime = chrono::steady_clock::now();
    vector<uint>distance(G.vertexArrSize + 1, m_INFINITY);
    vector<uint>parent(G.vertexArrSize + 1);
    
    //GPU start
    uint* D_distance = new uint[G.vertexArrSize];
    bool* D_visited = new bool[G.vertexArrSize];
    for (uint i = 0; i < G.vertexArrSize; i++)
    {
        D_distance[i] = m_INFINITY;
        D_visited[i] = false;
    }
    D_distance[source] = 0;
    auto endGPUInitTime = chrono::steady_clock::now();
    auto duration_gpu = chrono::duration_cast<chrono::milliseconds>(endGPUInitTime - startGPUInitTime).count();
    cout << "GPUInitTime =  " << duration_gpu << " ms" << "\n";
    cout << "Initlization finished!" << "\n";
    runGPUsssp3(G, source, D_distance, m_INFINITY, q);

    int nodesum = 0;
    for (uint i = 0; i < G.vertexArrSize; i++) {
        if (D_distance[i] != m_INFINITY)
            nodesum++;
    }
    cout << "nodesum: " << nodesum << "\n";
    delete[] D_distance;
    delete[] D_visited;
    //GPU end

    return 0;
}
