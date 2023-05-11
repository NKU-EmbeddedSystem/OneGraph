#include "global.h"
#include <CL/sycl.hpp>
#include <stdio.h>

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
	void partition(SIZE_TYPE* activeNodesPointer, uint numActiveNodes,ull max_partition_size);
	void reset();
};



Partitioner::Partitioner()
{
	reset();
}


void Partitioner::partition(SIZE_TYPE* activeNodesPointer, uint numActiveNodes,ull max_partition_size )
{
	reset();
	unsigned int from, to;
	unsigned int left, right, mid;
	unsigned int partitionSize;
	unsigned int numNodesInPartition;
	unsigned int numPartitionedEdges;
	bool foundTo;
	unsigned int accurCount;


	from = 0;
	to = numActiveNodes;
	numPartitionedEdges = 0;

	do
	{
		left = from;
		right = numActiveNodes;
		partitionSize = activeNodesPointer[right] - activeNodesPointer[left];
		if (partitionSize <= max_partition_size)
		{
			to = right;
		}
		else
		{
			foundTo = false;
			accurCount = 10;
			while (foundTo == false || accurCount > 0)
			{
				mid = (left + right) / 2;
				partitionSize = activeNodesPointer[mid] - activeNodesPointer[from];
				if (foundTo == true)
					accurCount--;
				if (partitionSize <= max_partition_size)
				{
					left = mid;
					to = mid;
					foundTo = true;
				}
				else
				{
					right = mid;
				}
			}

			if (to == numActiveNodes)
			{
				cout << "Error in Partitioning...\n";
				exit(-1);
			}
		}
		partitionSize = activeNodesPointer[to] - activeNodesPointer[from];
		numNodesInPartition = to - from;
		fromNode.push_back(from);
		fromEdge.push_back(numPartitionedEdges);
		partitionNodeSize.push_back(numNodesInPartition);
		partitionEdgeSize.push_back(partitionSize);
		from = to;
		numPartitionedEdges += partitionSize;

	} while (to != numActiveNodes);

	numPartitions = fromNode.size();
}


void Partitioner::reset()
{
	fromNode.clear();
	fromEdge.clear();
	partitionNodeSize.clear();
	partitionEdgeSize.clear();
	numPartitions = 0;
}
