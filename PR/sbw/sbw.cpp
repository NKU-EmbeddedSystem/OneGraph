#include <CL/sycl.hpp>
#include <algorithm>
#include <math.h>
#include <numeric>
#include <vector>
#include "subgraph.h"
#include "partitioner.h"
#include "subgraph_generator.h"
using namespace sycl;
#include "Graph.h"
#include "global.h"

//#include "/home/lsy/wzr/oneTBB/include/oneapi/tbb.h"
//#include "/home/lsy/wzr/oneTBB/include/oneapi/tbb/partitioner.h"
//#include "oneapi/tbb/parallel_for.h"
// static constexpr size_t N = 1024*56;
// static constexpr size_t B = 1024;
static constexpr size_t sub_group_size = 32;
//const bool debug=false;
static bool GPUon = true;
class my_device_selector : public device_selector {
public:
    my_device_selector(std::string vendorName) : vendorName_(vendorName) {};
    int operator()(const device& dev) const override {
        int rating = 0;
        if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
            rating = 3;
        else if (dev.is_gpu()) rating = 2;
        else if (dev.is_cpu()) rating = 1;
        return rating;
    };

private:
    std::string vendorName_;
};

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


template <typename T>
void pr_init(uint vertexArrSize, uint edgeArrSize, uint* edgeArray, ull* nodePointers, uint* degree, T* output) {
    auto startTime = chrono::steady_clock::now();

    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        degree[i] = nodePointers[i + 1] - nodePointers[i];
        output[i] = (T)1.0 / vertexArrSize;
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    output[vertexArrSize - 1] = (T)1.0 / vertexArrSize;
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "init time: " << duration << " ms\n";

}


template <typename T, typename E>
void pr_calQ(Graph<E>graph,int& round, float beta, T* output, sycl::queue& q) {

    auto calStartTime2 = chrono::steady_clock::now();
    
    graph.d_outDegree = malloc_device<uint>(graph.vertexArrSize, q);
    graph.d_degree = malloc_device<uint>(graph.vertexArrSize, q);

    q.memcpy(graph.d_degree, graph.degree, sizeof(uint) * graph.vertexArrSize);
    q.memcpy(graph.d_outDegree, graph.outDegree, sizeof(uint) * graph.vertexArrSize);

    T* d_output = malloc_device<T>(graph.vertexArrSize, q);
    q.memcpy(d_output, output, sizeof(T) * graph.vertexArrSize);

    bool* inactiveNodeD;
    T* valueD = malloc_device<T>(graph.vertexArrSize, q);

    int activeNodeSum = graph.vertexArrSize;

    inactiveNodeD = malloc_device<bool>(graph.vertexArrSize, q);

    uint* d_outDegree=graph.d_outDegree;
    uint* d_degree=graph.d_degree;
    
    Subgraph<E> subgraph(graph,q);
	SubgraphGenerator<E> subgen(graph,q);
    Partitioner<E> partitioner;

    subgen.generate(graph, subgraph, inactiveNodeD,q);	

    double KernelTime=0;
    double genTime=0;
    //double GenTime=0;
    cout<<"srart while\n";
    q.wait();
    while (activeNodeSum!=0) {
        round++;
        //cout << "\n=======round "<<round << "=======\n";

        partitioner.partition(subgraph, subgraph.numActiveNodes);

        activeNodeSum=subgraph.numActiveNodes;         
        //cout<<" numActiveNodes = "<<activeNodeSum<<"\n";
        for(int i=0; i<partitioner.numPartitions; i++){
            q.memcpy(subgraph.d_activeEdgeList, subgraph.activeEdgeList + partitioner.fromEdge[i], 
                    (partitioner.partitionEdgeSize[i]) * sizeof(OutEdge)).wait();

            uint NodeStart=partitioner.fromNode[i];
            uint NodeSize=partitioner.partitionNodeSize[i];
            uint EdgeStart=partitioner.fromEdge[i];
            
            uint *d_activeNodes=subgraph.d_activeNodes;
            ull *d_activeNodesPointer=subgraph.d_activeNodesPointer;

            uint *d_activeEdgeList=subgraph.d_activeEdgeList;

        q.wait();

        auto calStartTime = chrono::steady_clock::now();
        try {
            
            q.submit([&](handler& h) {

                h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)  {
                    auto tid = item.get_global_id()[0];

                    for (SIZE_TYPE i = NodeStart+tid; i < NodeStart+NodeSize; i += N) {
                        ull edgeIndex = d_activeNodesPointer[i]-EdgeStart;
                        
                        T tempSum = 0;
                        for (uint j = edgeIndex; j < edgeIndex + d_degree[d_activeNodes[ i]]; j++) {
                            uint srcNodeIndex = d_activeEdgeList[j];

                            if (d_outDegree[srcNodeIndex]>0) {

                                tempSum += d_output[srcNodeIndex] / d_outDegree[srcNodeIndex];
                                
                            }
                        }
                        
                        valueD[d_activeNodes[i]] = 0.15 + 0.85 * tempSum;
                    }
                    for (SIZE_TYPE i = NodeStart+tid; i <NodeStart+ NodeSize; i += N) {
                        T diff = abs(valueD[d_activeNodes[ i]] - d_output[d_activeNodes[i]]);
                        d_output[d_activeNodes[ i]] = valueD[d_activeNodes[ i]];

                        if (diff < 0.001) {
                            inactiveNodeD[d_activeNodes[ i]] = true;
                        }
                    }
                     });
                }).wait();
        }
        catch (sycl::exception& e) {
            std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
        }
        catch (std::exception& e) {
            std::cout << "Caught std exception: " << e.what() << "\n";
        }
        catch (...) {
            std::cout << "Caught unknown exception\n";
        }
        auto endTime = chrono::steady_clock::now();
        auto calDuration = chrono::duration_cast<chrono::milliseconds>(endTime - calStartTime).count();
	    KernelTime+=calDuration;
        }

        auto genStart = chrono::steady_clock::now();
        subgen.generate(graph,subgraph,inactiveNodeD,q);
        auto genEnd = chrono::steady_clock::now();
        auto genDuration = chrono::duration_cast<chrono::milliseconds>(genEnd - genStart).count();
	    genTime+=genDuration;

    }
    q.memcpy(output, d_output, sizeof(T) * graph.vertexArrSize).wait();

    auto endTime2 = chrono::steady_clock::now();
    auto calDuration2 = chrono::duration_cast<chrono::milliseconds>(endTime2 - calStartTime2).count();
    cout << "GPU cal time: " << calDuration2 << " ms\n";
    cout<<"genTime = "<<genTime<<"\n";
    cout<<"KernelTime = "<<KernelTime<<"\n";
    cout<<"calDuration = "<<calDuration2<<"\n";

    free(inactiveNodeD, q);
}

int main(int argc, char** argv) {
    double* output_cpu;
    int vertexArrSize;
    int ch;
    string dataname;
    string resname;
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
            printf("options as following:\n");
            printf("\t-f | input file name\n");
            printf("\t-r | BFS&sssp root \n");
            printf("\t-h | help message\n");
            return 0;
        case 'n'://for result check
            vertexArrSize = stoi(optarg);
            break;
        case 'c'://for result check
            resname = optarg;
            break;
        case '?':
            break;
        default:
            break;
        }
    }


    Graph<uint> G_gpu;
    float beta = 0.85;

    double* output;

    if (GPUon) {
        cout << "==gpu pr START\n";
        int round_gpu = 0;
        string vendor_name = "Nvidia";
        my_device_selector selector(vendor_name);
        sycl::queue q(selector);
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        auto startTime2 = chrono::steady_clock::now();
        G_gpu.readDataFromFile(dataname);

        output = (double *)malloc(G_gpu.vertexArrSize*sizeof(double));
        pr_init(G_gpu.vertexArrSize, G_gpu.edgeArrSize, G_gpu.edgeArray, G_gpu.nodePointers, G_gpu.degree, output);

        pr_calQ<double,uint>(G_gpu,round_gpu, beta, output, q);

        auto endTime2 = chrono::steady_clock::now();

        auto allDuration2 = chrono::duration_cast<chrono::milliseconds>(endTime2 - startTime2).count();

        cout << "total time: " << allDuration2 << " ms\n";

    }

    return 0;
}

