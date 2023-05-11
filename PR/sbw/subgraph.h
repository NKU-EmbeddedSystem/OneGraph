#ifndef SUBGRAPH_H
#define SUBGRAPH_H

#include <CL/sycl.hpp>
#include "global.h"
#include "Graph.h"
using namespace std;
using namespace sycl;

template <class E>
class Subgraph
{
private:

public:
	uint num_nodes;
	uint num_edges;
	uint numActiveNodes;
	sycl::queue q;
	uint *activeNodes;
	ull *activeNodesPointer;
	E *activeEdgeList;
	
	uint *d_activeNodes;
	ull *d_activeNodesPointer;
	E *d_activeEdgeList;
	
	ull max_partition_size;
	

	Subgraph(Graph<E> graph,sycl::queue q)
{

	ull n1024=1024;
    ull memorys = (ull)(n1024 * n1024 * n1024* 10);
	this->num_nodes = graph.vertexArrSize;
	this->num_edges = graph.edgeArrSize;

    max_partition_size = (0.75* (memorys -8*19*num_nodes) / sizeof(E));
	cout<<"max_partition_size = "<<max_partition_size<<"\n";

    activeNodes = malloc_host<uint>(num_nodes, q);
	activeNodesPointer = malloc_host<ull>(num_nodes+1, q);
	activeEdgeList = malloc_host<uint>(num_edges, q);

	d_activeNodes = malloc_device<uint>(num_nodes, q);
	d_activeNodesPointer = malloc_device<ull>(num_nodes+1, q);
	d_activeEdgeList = malloc_device<uint>(max_partition_size, q);

}
};

#endif	//	SUBGRAPH_H
