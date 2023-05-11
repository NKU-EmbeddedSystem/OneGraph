#ifndef PARTITIONER_CUH
#define PARTITIONER_CUH
#include "global.h"

template <class E>
class Partitioner
{
private:

public:
	uint numPartitions;
	vector<uint> fromNode;
	vector<uint> fromEdge;
	vector<uint> partitionNodeSize;
	vector<uint> partitionEdgeSize;
	Partitioner();
    void partition(SIZE_TYPE* activeNodesPointer, uint numActiveNodes, uint max_partition_size);
    void reset();
};

#endif



