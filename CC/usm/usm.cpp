#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include "global.h"
#include<chrono>
#include <ctime>
#include <stdio.h>
#include <string>
#include <cstring>
#include <cstdlib>
static constexpr size_t sub_group_size = 16;
using namespace sycl;
using namespace std;
static constexpr size_t N = 1024 * 108;
static constexpr size_t B = 1024;
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
void validation(int vertexArrSize, bool* labelD) {
    int cnt = 0;
    for (int i = 0; i < vertexArrSize; i++) {
        if (labelD[i] == 0) cnt++;
    }
    cout << "connetcted component num: "<<cnt<<"\n";
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
    nodePointers = malloc_shared<EDGE_POINTER_TYPE>(vertexArrSize, q);
    infile.read((char*)nodePointers, sizeof(EDGE_POINTER_TYPE) * vertexArrSize);
    edgeArray = malloc_shared<uint>(edgeArrSize, q);

    infile.read((char*)edgeArray, sizeof(uint) * edgeArrSize);
    infile.close();
    auto endreadTime = chrono::steady_clock::now();
    auto readTime = chrono::duration_cast<chrono::milliseconds>(endreadTime - startTime).count();
    cout << "readfile Time: " << readTime << " ms" << "\n";
   
    uint* degreeD = malloc_shared<uint>(vertexArrSize, q);

    uint* valueD = malloc_shared<uint>(vertexArrSize, q);

    bool* labelD = malloc_shared<bool>(vertexArrSize, q);
    q.memset(labelD, 0, vertexArrSize * sizeof(bool)).wait();

    bool* isActive = malloc_shared<bool>(vertexArrSize, q);
    q.memset(isActive, 1, vertexArrSize * sizeof(bool)).wait();

    bool* isActivenew = malloc_shared<bool>(vertexArrSize, q);
    q.memset(isActivenew, 0, vertexArrSize * sizeof(bool)).wait();

    auto startiniTime = chrono::steady_clock::now();
    
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        valueD[i] = i;
        if (i + 1 < vertexArrSize) {
            degreeD[i] = nodePointers[i + 1] - nodePointers[i];
        }
        else {
            degreeD[i] = edgeArrSize - nodePointers[i];
        }
    }
    auto endiniTime = chrono::steady_clock::now();
    auto iniTime = chrono::duration_cast<chrono::milliseconds>(endiniTime - startiniTime).count();
    cout << "Initdata Time: " << iniTime << " ms\n";

    ull activenodes = vertexArrSize;
    uint testnodesnum = 0;
    auto calStart = chrono::steady_clock::now();
    
    while (activenodes != 0) {
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
                auto tid = item.get_global_id()[0];
                for (SIZE_TYPE id = tid; id < vertexArrSize; id += N) {
                    if (!isActive[id]) {
                        continue;
                    }
                    uint edgeIndex = nodePointers[id];
                    uint sourceValue = valueD[id];
                    for (uint i = edgeIndex; i < edgeIndex + degreeD[id]; i++) {
                        if (sourceValue < valueD[edgeArray[i]]) {
                            valueD[edgeArray[i]] = sourceValue;
                            labelD[edgeArray[i]] = true;
                            isActivenew[edgeArray[i]] = 1;
                        }
                    }
                }
                for (SIZE_TYPE id = tid; id < vertexArrSize; id += N) {
                    isActive[id] = isActivenew[id];
                    isActivenew[id] = 0;
                }
                });
            }).wait();
        activenodes = 0;
        for (uint id = 0; id < vertexArrSize; id++) {
            activenodes += (int)isActive[id];
        }
        testnodesnum += activenodes;
    }
    testnodesnum += vertexArrSize;
    auto calEnd = chrono::steady_clock::now();
    auto calDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - calStart).count();        
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(calEnd - startTime).count();
    std::cout << "Calculate Time: " << calDuration << " ms\n";
    std::cout << "Total Time: " << totalDuration << " ms\n";

    // validation(vertexArrSize, labelD);
    free(nodePointers, q);
    free(edgeArray, q);
    free(degreeD, q);
    free(valueD, q);
    free(labelD, q);
    free(isActive, q);
    free(isActivenew, q);
    
    cout << "nodeSum : "<<testnodesnum<<"\n";
    return 0;
}
