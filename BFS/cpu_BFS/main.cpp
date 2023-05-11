#include <stdio.h>                  /* for EOF */
#include <string.h>                 /* for strchr() */
#include<CL/sycl.hpp>

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

void runCPUBFS(Graph<myEdgeType>& G, uint source, uint*& distance, const uint& m_INFINITY, ofstream& out_res)
{


    auto startCPUInitTime = chrono::steady_clock::now();//For Time: start CPUInitTime

    uint nodeSize = G.vertexArrSize;
    uint edgeSize = G.edgeArrSize;
    uint* degree = new uint[nodeSize];
    bool* isActiveNodeList = new bool[nodeSize];
    
    for (uint i = 0; i < nodeSize; i++) 
    {
        if (i + 1 < nodeSize) {
            degree[i] = G.nodePointers[i + 1] - G.nodePointers[i];
        }
        else {
            degree[i] = edgeSize - G.nodePointers[i];
        }
    }

    bool* nextActiveNodeListD = new bool[nodeSize];
    uint* value = new uint[nodeSize];

    for (int i = 0; i < nodeSize; i++) {
        nextActiveNodeListD[i] = 0;
        isActiveNodeList[i] = false;
        value[i] = UINT_MAX;
    }

    isActiveNodeList[source] = true;
    value[source] = 1;
    uint activeSum = 0;
    int iteration = 0;
    uint activeNodeNum = 1;


    auto endCPUInitTime = chrono::steady_clock::now();//For Time: end CPUInitTime
    auto CPUInitTime = chrono::duration_cast<chrono::milliseconds>(endCPUInitTime - startCPUInitTime).count();
    cout << "CPUInitTime =  " << CPUInitTime << " ms" << "\n";
    out_res << "CPUInitTime =  " << CPUInitTime << " ms" << "\n";

    auto startCPUCalTime = chrono::steady_clock::now();//For Time: start CPUCalTime
    while (true) {
        if (activeNodeNum <= 0) {
            break;
        }
        else {
            activeSum += activeNodeNum;
        }

        for (int index = 0; index < nodeSize; index++) {
            uint nodeIndex = 0 + index;
            if (isActiveNodeList[nodeIndex]) {
                uint edgeIndex = G.nodePointers[nodeIndex] - 0;
                uint sourceValue = value[nodeIndex];
                uint finalValue;
                for (uint i = edgeIndex; i < edgeIndex + degree[nodeIndex]; i++) 
                {
                    finalValue = sourceValue + 1;
                    myEdgeType vertexId = G.edgeArray[i];
                    if (finalValue < value[vertexId]) {
                        value[vertexId] = finalValue;
                        nextActiveNodeListD[vertexId] = true;
                    }
                }
            }
        }

        activeNodeNum = 0;
        for (int ii = 0; ii < nodeSize; ii++) {
            isActiveNodeList[ii] = nextActiveNodeListD[ii];
            if (isActiveNodeList[ii]) {
                activeNodeNum++;
            }
            nextActiveNodeListD[ii] = 0;
        }
        
        //cout<<"iteration: "<<iteration<<";  activeNodeNum: "<<activeNodeNum<<"\n";
        iteration++;
    }

    for (int i = 0; i < nodeSize; i++) {
        
        distance[i] = value[i];
    }

    delete[] degree;
    delete[]isActiveNodeList;
    delete[]nextActiveNodeListD;
    delete[]value;
    auto endCPUCalTime = chrono::steady_clock::now();//For Time: end CPUCalTime
    auto CPUCalTime = chrono::duration_cast<chrono::milliseconds>(endCPUCalTime - startCPUCalTime).count();
    cout << "CPUCalTime: " << CPUCalTime << "ms" << "\n";
    out_res << "CPUCalTime: " << CPUCalTime << "ms" << "\n";
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
    string filename = dataname;

    ofstream out_res;
    out_res.open("./res.txt", ios::app);
    if (!out_res)cout << "out_res open failed!" << "\n";
    
    out_res<<"\n\n\n";
    out_res<<filename<<"\n";


    G.readDataFromFile(filename);
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

    //CPU start
    for (int testIndex = 0; testIndex < testTimes; testIndex++) 
    {
        cout<<"----------testIndex = "<<testIndex<<"-----------\n";
        out_res<<"----------testIndex = "<<testIndex<<"-----------\n";
        source = src[testIndex];
        uint* distance_s = new uint[G.vertexArrSize];
        runCPUBFS(G, source, distance_s, m_INFINITY, out_res);

        uint nodesum = 0;
        for (uint i = 0; i < G.vertexArrSize; i++) {
            if (distance_s[i] != m_INFINITY)
                nodesum++;
        }
        cout << "nodesum: " << nodesum << "\n";
        out_res << "nodesum: " << nodesum << "\n";
        delete[] distance_s;
    }
    delete []src;
}