#include "CalculateOpt.h"
#include "ArgumentParser.h"
int main(int argc, char** argv) {
    ArgumentParser arguments(argc, argv, true);
    if (arguments.algo.empty()) {
        arguments.algo = "bfs";
    }
    cout << "arguments.algo " << arguments.algo << "\n";
    if (arguments.algo == "bfs") {
        bfs_opt(arguments.input, arguments.sourceNode, arguments.adviseK);
     } 
    
    return 0;
}

