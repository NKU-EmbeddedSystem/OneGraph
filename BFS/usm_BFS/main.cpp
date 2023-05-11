#include <stdio.h>                  /* for EOF */
#include <string.h>                 /* for strchr() */
#include<CL/sycl.hpp>

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
//======================================================================================================================================
#include"Graph.h"


using namespace std;

static constexpr size_t sub_group_size = 16;
static constexpr size_t N = 1024 * 108;
static constexpr size_t B = 1024;

void runGPUBFS(Graph<myEdgeType>& G, uint source, uint*& distance, const uint& m_INFINITY, sycl::queue& q, ofstream& out_res)
{
    auto startGPUInitTime = chrono::steady_clock::now();//For Time: start GPUInitTime
 
    uint nodeSize = G.vertexArrSize;
    uint edgeSize = G.edgeArrSize;
    uint* degreeD = malloc_shared<uint>(nodeSize, q);
    bool* isActiveNodeListD = malloc_shared<bool>(nodeSize, q);
    bool* nextActiveNodeListD = malloc_shared<bool>(nodeSize, q);
    uint* valueD = malloc_shared<uint>(nodeSize, q);

    for (uint i = 0; i < nodeSize; i++) 
    {
        if (i + 1 < nodeSize) {
            degreeD[i] = G.nodePointers[i + 1] - G.nodePointers[i];
        }
        else {
            degreeD[i] = edgeSize - G.nodePointers[i];
        }
    }

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


    auto endGPUInitTime = chrono::steady_clock::now();//For Time: end GPUInitTime
    auto GPUInitTime = chrono::duration_cast<chrono::milliseconds>(endGPUInitTime - startGPUInitTime).count();
    cout << "GPUInitTime =  " << GPUInitTime << " ms" << "\n";
    out_res << "GPUInitTime =  " << GPUInitTime << " ms" << "\n";

    auto startGPUCalTime = chrono::steady_clock::now();//For Time: start GPUCalTime
    while (true) {
        if (activeNodeNum <= 0) 
            break;
        else
            activeSum += activeNodeNum;
        myEdgeType* d_edgeArray = G.edgeArray;
        EDGE_POINTER_TYPE* d_nodePointers = G.nodePointers;

        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
            auto tid = item.get_global_id()[0];
            for (int index = tid; index < nodeSize; index += N) 
            {
                uint nodeIndex = 0 + index;
                if (isActiveNodeListD[nodeIndex]) 
                {
                    uint edgeIndex = d_nodePointers[nodeIndex] - 0;
                    uint sourceValue = valueD[nodeIndex];
                    uint finalValue;
                    for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++)
                    {
                        finalValue = sourceValue + 1;
                        myEdgeType vertexId = d_edgeArray[i];
                        if (finalValue < valueD[vertexId]) {
                           valueD[vertexId] = finalValue;
                            nextActiveNodeListD[vertexId] = true;
                        }
                    }
                }
            }
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
            //cout<<"iteration: "<<iteration<<";  activeNodeNum: "<<activeNodeNum<<"\n";
            iteration++;
    }

    for (int i = 0; i < nodeSize; i++) {
        distance[i] = valueD[i];
    }
    free(degreeD, q);
    free(isActiveNodeListD, q);
    free(nextActiveNodeListD, q);
    free(valueD, q);

    //=====================================================================================================
    auto endGPUCalTime = chrono::steady_clock::now();//For Time: end GPUCalTime
    auto GPUCalTime = chrono::duration_cast<chrono::milliseconds>(endGPUCalTime - startGPUCalTime).count();
    cout << "GPUCalTime: " << GPUCalTime << "ms" << "\n";
    out_res << "GPUCalTime: " << GPUCalTime << "ms" << "\n";
}




#define MYINFINITY 0xFFFFFFFF

int testTimes = 5;
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

    Graph<myEdgeType> G;
    sycl::queue q;
    string filename = dataname;

    //start display
    ofstream out_res;
    out_res.open("./res.txt", ios::app);
    if (!out_res)cout << "out_res open failed!" << "\n";

    out_res<<"\n\n\n";
    out_res<<filename<<"\n";


    G.readDataFromFile(filename, q);
    out_res << "vertexArrSize = " << G.vertexArrSize << " edgeArrSize = " << G.edgeArrSize << "\n";
    cout << "after readDataFromFile" << "\n";

    //设置source
    uint *src=new uint[testTimes];
    uint ttt=0;
    for(int i=0;i<testTimes;i++)
    {
        src[i]=G.findbignode(ttt);
        ttt = src[i];
    }

    //GPU start
    for (int testIndex = 0; testIndex < testTimes; testIndex++) 
    {
        cout<<"----------testIndex = "<<testIndex<<"-----------\n";
        out_res<<"----------testIndex = "<<testIndex<<"-----------\n";
        source = src[testIndex];
        uint* D_distance = new uint[G.vertexArrSize];
        runGPUBFS(G, source, D_distance, m_INFINITY, q, out_res);

        uint nodesum = 0;
        for (uint i = 0; i < G.vertexArrSize; i++) {
            if (D_distance[i] != m_INFINITY)
                nodesum++;
        }
        cout << "nodesum: " << nodesum << "\n";
        out_res << "nodesum: " << nodesum << "\n";
        delete[] D_distance;
    }
    out_res.close();
}