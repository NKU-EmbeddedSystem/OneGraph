#include "Graph.h"
#include "global.h"
#include <vector>
#include <cstring>


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
void pr_init(uint vertexArrSize, uint edgeArrSize, uint* edgeArray, ull* nodePointers, uint* degree,T * output) {
   auto startTime = chrono::steady_clock::now();

    for (int i = 0; i < vertexArrSize-1; i++){
        output[i] = (T)1.0 / vertexArrSize;
	degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    output[vertexArrSize - 1] = (T)1.0 / vertexArrSize;
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "init time: " << duration << " ms\n";
}

template <typename T>
void pr_cal(uint vertexArrSize, uint* edgeArray, ull* nodePointers, uint* degree, uint* outDegree,
    int& round, double beta, T* output) {

    vector<short int> activeNode(vertexArrSize, 1);
    T* value = (T*)calloc(vertexArrSize, sizeof(T));

    bool stopIter = false;
    int activeNodeSum = vertexArrSize;

    while (activeNodeSum) {
        for (SIZE_TYPE i = 0; i < activeNode.size(); i++) {
            if (activeNode[i] == 0)continue;
            uint edgeIndex = nodePointers[i];
            T tempSum = 0;
            for (uint j = edgeIndex; j < edgeIndex + degree[i]; j++) {
                uint srcNodeIndex = edgeArray[j];
                if (outDegree[srcNodeIndex]) {
                    T tempValue = output[srcNodeIndex] / outDegree[srcNodeIndex];
                    tempSum += tempValue;
                }
            }
            value[i] = 0.15 + 0.15 * tempSum;
        }


        for (SIZE_TYPE i = 0; i < activeNode.size(); i++) {
            if (activeNode[i] == 0)continue;
            T diff = abs(value[i] - output[i]);
            output[i] = value[i];
            value[i] = 0;

            if (diff < 0.001) {
                activeNode[i] = 0;
                activeNodeSum--;
            }
        }round++;
        //cout<<"round = "<<round<<"\t active = "<<activeNodeSum<<"\n";
    }
}

int main(int argc, char** argv) {

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
            printf("options as following:\n");
            printf("\t-f | input file name\n");
            printf("\t-r | BFS&sssp root \n");
            printf("\t-h | help message\n");
            return 0;
        case '?':
            break;
        default:
            break;
        }
    }
    
    Graph<uint> G;
    int round = 0;
    double beta = 0.85;
    auto startTime = chrono::steady_clock::now();

    G.readDataFromFile(dataname);
    double* output = (double*)calloc(G.vertexArrSize, sizeof(double));

    pr_init(G.vertexArrSize,G.edgeArrSize, G.edgeArray, G.nodePointers, G.degree, output);
    //G.printself();
    auto calStart = chrono::steady_clock::now();
    pr_cal<double>(G.vertexArrSize, G.edgeArray, G.nodePointers, G.degree,G.outDegree,
        round, beta, output);
    auto endTime = chrono::steady_clock::now();
    auto calDuration = chrono::duration_cast<chrono::milliseconds>(endTime - calStart).count();
    auto allDuration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "caltime: " << calDuration << " ms\n"<< "time: " << allDuration << " ms\n";



    return 0;
}
