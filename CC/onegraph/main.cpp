#include "CalculateOpt.h"
#include "ArgumentParser.h"
int main(int argc, char** argv) {
    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "cc";
    }
    cout << "arguments.algo " << arguments.algo << "\n";
    if (arguments.algo == "cc") {
        cc_opt(arguments.input, arguments.adviseK);
     } 
    
    return 0;
}

