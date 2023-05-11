#include "CalculateOpt.h"
//#include "constants.h"
#include "ArgumentParser.h"
int main(int argc, char** argv) {

    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "sssp";
    }
    cout << "arguments.algo " << arguments.algo << "\n";
    if (arguments.algo == "sssp") {
        sssp_opt(arguments.input, arguments.sourceNode, arguments.adviseK);
     } 
    
    return 0;
}

