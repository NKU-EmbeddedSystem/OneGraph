#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include "Graph.h"
#include <chrono>
#include <string>
using namespace std;

char* optarg = NULL;    /* pointer to the start of the option argument  */
int   optind = 1;       /* number of the next argv[] to be evaluated    */
int   opterr = 1;       /* non-zero if a question mark should be returned
                           when a non-valid option character is detected */

                           /* handle possible future character set concerns by putting this in a macro */
#define _next_char(string)  (char)(*(string+1))
void validation(uint vertexArrSize, bool* labelD) {
    int cnt = 0;
    for (int i = 0; i < vertexArrSize; i++) {
        if (labelD[i] == 0) cnt++;
    }
    std::cout<<"connected component num : "<<cnt<<"\n";
}
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

int main(int argc, char* argv[]) {

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
    Graph<uint> G;
    auto startTime = chrono::steady_clock::now();
    G.readDataFromFile(path);
    auto endreadTime = chrono::steady_clock::now();
    auto readTime = chrono::duration_cast<chrono::milliseconds>(endreadTime - startTime).count();
    cout << "readfile Time: " << readTime << " ms\n";

    uint nodeSize = G.vertexArrSize;
    ull edgeSize = G.edgeArrSize;

    uint* value = new uint[G.vertexArrSize];

    bool* label = new bool[G.vertexArrSize];
    bool* isActiveNodeList = new bool[nodeSize];
    bool* nextActiveNodeListD = new bool[nodeSize];
    auto startiniTime = chrono::steady_clock::now();

    for(SIZE_TYPE i = 0; i < G.vertexArrSize; i++){
        label[i] = 0;
        isActiveNodeList[i] = true;
        nextActiveNodeListD[i] = false;
    }
    
    for (SIZE_TYPE i = 0; i < G.vertexArrSize; i++) {
        value[i] = i;
        if (i + 1 < G.vertexArrSize) {
            G.degree[i] = G.nodePointers[i + 1] - G.nodePointers[i];
        }
        else {
            G.degree[i] = G.edgeArrSize - G.nodePointers[i];
        }
    }
    auto endiniTime = chrono::steady_clock::now();
    auto iniTime = chrono::duration_cast<chrono::milliseconds>(endiniTime - startiniTime).count();
    cout << "Initdata Time: " << iniTime << " ms\n";
    uint activeNodeNum = 1;
    uint testnodesnum = 0;
    auto calStart = chrono::steady_clock::now();
    while (true) {
        if (activeNodeNum <= 0) {
            break;
        }
        for (int index = 0; index < nodeSize; index++) {
            if (isActiveNodeList[index]) {
                uint edgeIndex = G.nodePointers[index];
                uint sourceValue = value[index];
                for (uint i = edgeIndex; i < edgeIndex + G.degree[index]; i++){
                    if (sourceValue < value[G.edgeArray[i]]) {
                        testnodesnum ++;
                        value[G.edgeArray[i]] = sourceValue;
                        nextActiveNodeListD[G.edgeArray[i]] = true;
                        label[G.edgeArray[i]] = true;
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
        
    } 

    auto calEnd = chrono::steady_clock::now();
    auto calDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - calStart).count();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - startTime).count();
    std::cout << "Calculate Time: " << calDuration << " ms\n";
    std::cout << "Total Time: " << totalDuration << " ms\n";

    free(value);        
    // validation(G.vertexArrSize, label);
    free(label);
    
    cout << "nodeSum : "<<testnodesnum<<"\n";
    return 0;
}
