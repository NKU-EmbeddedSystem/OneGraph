#include "globals.h"
#pragma once
class ArgumentParser {
private:

public:
    int argc;
    char **argv;

    bool canHaveSource;

    bool hasInput;
    bool hasSourceNode;
    string input;
    float adviseK = 0.0f;
    int sourceNode;
    int method;
    string algo;

    ArgumentParser(int argc, char **argv, bool canHaveSource);

    bool Parse();

    string GenerateHelpString();

};

ArgumentParser::ArgumentParser(int argc, char **argv, bool canHaveSource) {
    this->argc = argc;
    this->argv = argv;
    this->canHaveSource = canHaveSource;

    this->sourceNode = 0;

    hasInput = false;
    hasSourceNode = false;
    Parse();
}

bool ArgumentParser::Parse() {
    try {
        for (int i = 1; i < argc - 1; i = i + 2) {
            //argv[i]

            if (strcmp(argv[i], "-f") == 0) {
                input = string(argv[i + 1]);
                hasInput = true;
            } else if (strcmp(argv[i], "--type") == 0) {
                algo = string(argv[i + 1]);
            } else if (strcmp(argv[i], "-r") == 0 && canHaveSource) {
                sourceNode = atoi(argv[i + 1]);
                hasSourceNode = true;
            } else if (strcmp(argv[i], "--method") == 0) {
                method = atoi(argv[i + 1]);
            } else if (strcmp(argv[i], "--adviseK") == 0) {
                adviseK = atof(argv[i + 1]);
            }
        }

        if (hasInput)
            return true;
    }
    catch (const std::exception &strException) {
        std::cerr << strException.what() << "\n";
        GenerateHelpString();
        exit(0);
    }
    catch (...) {
        std::cerr << "An exception has occurred.\n";
        GenerateHelpString();
        exit(0);
    }
}

string ArgumentParser::GenerateHelpString() {
    string str = "\nRequired arguments:";
    str += "\n    [-f]: Input graph file. E.g., -f FacebookGraph.txt";
    str += "\nOptional arguments";
    if (canHaveSource)
        str += "\n    [-r]:  Begins from the source (Default: 0). E.g., -r 10";
    return str;
}