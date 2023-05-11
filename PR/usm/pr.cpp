#include <CL/sycl.hpp>

using namespace sycl;
#include "Graph.h"
#include "global.h"

static constexpr size_t N = 1024*56;
static constexpr size_t B = 1024;
static constexpr size_t sub_group_size = 32;
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

template <typename T>
void pr_cal_cpu(uint vertexArrSize, uint* edgeArray, ull* nodePointers, uint* degree, uint* outDegree,
    int& round, float beta, T* output) {

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
            value[i] = 0.15 + 0.85 * tempSum;
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
        cout <<"round: "<<round<<"\nnode: "<<activeNodeSum<<"\n";
    }
}


template <typename T>
void pr_calQ(uint vertexArrSize, uint edgeArrSize, uint* edgeArray, ull* nodePointers, uint* degree, uint* outDegree,
    int& round, float beta, T* output, sycl::queue& q) {

    bool * inactiveNodeD = malloc_device<bool>(vertexArrSize, q);
    T* valueD = malloc_device<T>(vertexArrSize, q);
    SIZE_TYPE * disenableD= malloc_shared<SIZE_TYPE>(N / 32, q);
    int activeNodeSum = vertexArrSize;

    while (activeNodeSum) {
        try {

        q.submit([&](handler& h) {
            //auto out = stream(10240, 7680, h);
            h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]] {
                auto sg = item.get_sub_group();
                auto tid = item.get_global_id()[0];
                int gid = sg.get_group_id()[0] + B / 32 * floor(tid / B);

                SIZE_TYPE cnt_disenableD = 0;

                for (SIZE_TYPE i = tid ; i < vertexArrSize; i += N) {
                    if (inactiveNodeD[i])continue;
                    uint edgeIndex = nodePointers[i];
                    T tempSum = 0;
                    for (uint j = edgeIndex; j < edgeIndex + degree[i]; j++) {
                        uint srcNodeIndex = edgeArray[j];
                        if (outDegree[srcNodeIndex]) {
                            T tempValue = output[srcNodeIndex] / outDegree[srcNodeIndex];
                            tempSum += tempValue;
                        }
                    }
                    valueD[i] = 0.15 + 0.85 * tempSum;
                }
                for (SIZE_TYPE i = tid; i < vertexArrSize; i += N) {
                    if (inactiveNodeD[i])continue;
                    T diff = abs(valueD[i] - output[i]);
                    output[i] = valueD[i];
                    //valueD[i] = 0;

                    if (diff < 0.001) {
                        inactiveNodeD[i] = true;
                        cnt_disenableD++;
                        
                    }
                }
                item.barrier(access::fence_space::local_space);
                SIZE_TYPE sum = reduce_over_group(sg, cnt_disenableD, sycl::plus<>());
                if (sg.get_local_id()[0] == 0)disenableD[gid] = sum;
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
        for(int i = 0; i < N/32; i ++)activeNodeSum -= disenableD[i];
            
        round++;

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

    Graph<uint> G_gpu;
    float beta = 0.85;
    double* output_cpu;
    double* output;

    
    if (GPUon) {
        cout << "==gpu pr START\n";
        int round_gpu = 0;
        string vendor_name = "Nvidia";
        my_device_selector selector(vendor_name);
        sycl::queue q(selector);
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        auto startTime2 = chrono::steady_clock::now();
        G_gpu.readDataFromFileQ(dataname,q);
        //G.printself();
        
        output = malloc_shared<double>(G_gpu.vertexArrSize, q);
        pr_init(G_gpu.vertexArrSize, G_gpu.edgeArrSize, G_gpu.edgeArray, G_gpu.nodePointers, G_gpu.degree, output);

        auto calStartTime2 = chrono::steady_clock::now();
        pr_calQ<double>(G_gpu.vertexArrSize, G_gpu.edgeArrSize, G_gpu.edgeArray, G_gpu.nodePointers, G_gpu.degree, G_gpu.outDegree,
            round_gpu, beta, output, q);

        auto endTime2 = chrono::steady_clock::now();
        auto calDuration2 = chrono::duration_cast<chrono::milliseconds>(endTime2 - calStartTime2).count();
        auto allDuration2 = chrono::duration_cast<chrono::milliseconds>(endTime2 - startTime2).count();
        cout << "GPU cal times: " << calDuration2 << " ms\n" << "total time: " << allDuration2 << " ms\n";

    }

    return 0;
}

