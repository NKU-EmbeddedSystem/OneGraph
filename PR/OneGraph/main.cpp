#include "CalculateOpt.h"
//#include "constants.h"
#include "ArgumentParser.h"
int main(int argc, char** argv) {

    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "pr";
    }
    cout << "arguments.algo " << arguments.algo << "\n";
    if (arguments.algo == "pr") {
        pr_opt(arguments.input, arguments.adviseK);
     } 
    
    return 0;
}

