#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
using namespace std;
typedef unsigned int uint;
typedef unsigned long long ull;
typedef uint SIZE_TYPE;
typedef ull EDGE_POINTER_TYPE;

struct OutEdge {
    uint end;
};

struct OutEdgeWeighted {
    uint end;
    uint w8;
};

struct Edge {
    uint source;
    uint end;
};

struct EdgeWeighted {
    uint source;
    uint end;
    uint w8;
};

struct llOutEdge {
    ull end;
};

struct llOutEdgeWeighted {
    ull end;
    ull w8;
};

