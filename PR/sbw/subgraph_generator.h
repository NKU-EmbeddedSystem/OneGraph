#ifndef SUBGRAPH_GENERATOR_H
#define SUBGRAPH_GENERATOR_H

#include "global.h"
#include "Graph.h"
#include "subgraph.h"
#include <CL/sycl.hpp>
#include <pthread.h>
#include <thread>
const unsigned int NUM_THREADS = 56;
static constexpr size_t N = 1024*56*10;
static constexpr size_t B = 1024;
const uint global_size=1024*56*10;
const unsigned int THRESHOLD_THREAD = 50000;
using namespace std;
using namespace sycl;
const bool debug=false;

struct params{
	int tId;
	uint numThreads;	
	uint numActiveNodes;
	uint *activeNodes;
	uint *outDegree; 
	ull *activeNodesPointer;
	ull *nodePointer; 
	uint *activeEdgeList;
	uint *edgeList;};
// pthread
void* dynamic(void *p)
{
	params* thread_p = (params*) p;  
	unsigned int chunkSize = ceil(thread_p->numActiveNodes / thread_p->numThreads);
	unsigned int left, right;
	left = thread_p->tId * chunkSize;
	right = std::min(left+chunkSize, thread_p->numActiveNodes);	
	
	unsigned int thisNode;
	unsigned int thisDegree;
	unsigned int fromHere;
	unsigned int fromThere;

	for(unsigned int i=left; i<right; i++)
	{
		thisNode = thread_p->activeNodes[i];
		thisDegree = thread_p->outDegree[thisNode];
		fromHere = thread_p->activeNodesPointer[i];
		fromThere = thread_p->nodePointer[thisNode];
		for(unsigned int j=0; j<thisDegree; j++)
		{
			thread_p->activeEdgeList[fromHere+j] = thread_p->edgeList[fromThere+j];
		}
	}
	return 0; // 必须返回
}

void prePrefix(uint *activeNodesLabeling, unsigned int *activeNodesDegree, 
				unsigned int *degree, bool *inactiveNodeD, unsigned int numNodes,sycl::queue q)
{
	q.submit([&](handler& h) {
		//auto out = stream(10240, 7680, h);
        //h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
		h.parallel_for(global_size, [=](id<1> tid) {
            //auto tid = item.get_global_id()[0];
            for(int id=tid;id<numNodes;id+=N){
                if(inactiveNodeD[id]==false){
                    activeNodesLabeling[id] = 1;
					activeNodesDegree[id] = degree[id];
				}
                else{
                    activeNodesLabeling[id] = 0;
					activeNodesDegree[id] = 0;
				}	
            }});
        
    }).wait();
      
}

void makeQueue(unsigned int *activeNodes, bool *activeNodesLabeling,
							uint *prefixLabeling, unsigned int numNodes,sycl::queue q)
{

	q.submit([&](handler& h) {
		h.parallel_for(global_size, [=](id<1> tid) {
            for(int id=tid;id<numNodes;id+=N){
                if(activeNodesLabeling[id]==false){
                    activeNodes[prefixLabeling[id]-1] = id;
				}
            }
        });
    }).wait();
}

void makeActiveNodesPointer(ull *activeNodesPointer, bool *activeNodesLabeling, 
											uint *prefixLabeling, uint *prefixSumDegrees, 
											unsigned int numNodes,sycl::queue q)
{
    q.submit([&](handler& h) {

		h.parallel_for(global_size, [=](id<1> tid) {
            for(int id=tid;id<numNodes;id+=N){
                if(activeNodesLabeling[id]==false){
                    activeNodesPointer[prefixLabeling[id]] = prefixSumDegrees[id];

				}
            }
        });
    }).wait();

}



uint* inclusive_scan(uint* current,uint* next, unsigned int N, queue& q) {
unsigned int two_power = 1;
unsigned int num_iter = ceil(log2(N));

uint* result = NULL;
    // Iterate over the necessary iterations.
    for (unsigned int iter = 0; iter < num_iter; iter++, two_power *= 2) {
      // Submit command group for execution
	  if (iter % 2 == 0) {
      q.submit([&](auto& h) {
        
          h.parallel_for(global_size, [=](id<1> j) {
            for(int i=j;i<N;i+=global_size){
            if (i < two_power) {
              next[i] = current[i];
            } else {
              next[i] = current[i] +current[i - two_power];
            }

            }
          });  // end parallel for loop in kernel
          result = next;
		  }).wait();

        } else {
			q.submit([&](auto& h) {
          	h.parallel_for(global_size, [=](id<1> j) {
            for(int i=j;i<N;i+=global_size){
            if (i < two_power) {
              current[i] = next[i];
            } else {
              current[i] = next[i] + next[i - two_power];
            }

            }
          });  // end parallel for loop in kernel
          result = current;
      }).wait();  // end device queue
    }      // end iteration
	}
  q.wait_and_throw();

  return result;
}


template <class E>
class SubgraphGenerator
{
private:

public:
	unsigned int *activeNodesLabeling;
	unsigned int *activeNodesDegree;
	unsigned int *prefixLabeling;
	unsigned int *prefixSumDegrees;
	unsigned int *d_activeNodesLabeling;
	unsigned int *d_activeNodesDegree;
	unsigned int *d_prefixLabeling;
	unsigned int *d_prefixSumDegrees;



SubgraphGenerator(Graph<E> &graph,sycl::queue q)
{

    d_activeNodesLabeling=malloc_device<uint>(graph.vertexArrSize, q);
    d_activeNodesDegree=malloc_device<uint>(graph.vertexArrSize, q);
    d_prefixLabeling=malloc_device<uint>(graph.vertexArrSize, q);
    d_prefixSumDegrees=malloc_device<uint>(graph.vertexArrSize+1, q);

}


void generate(Graph<E> &graph, Subgraph<E> &subgraph,bool *inactiveNodeD,sycl::queue q)
{


	uint vertexArrSize=graph.vertexArrSize;
	prePrefix(d_activeNodesLabeling, d_activeNodesDegree, graph.d_degree, inactiveNodeD, vertexArrSize,q);

	uint *anl=d_activeNodesLabeling;
    //reduction
    uint* activeSum = malloc_shared<uint>(1, q);
    *activeSum = 0;
	q.submit([&](handler& h) {
		//auto out = stream(10240, 7680, h);
    	h.parallel_for(nd_range<1>{N, B}, 
    	    reduction(activeSum, sycl::plus<>()), [=](nd_item<1> it, auto& temp) {
    	        auto i = it.get_global_id(0);
    	        for(int id=i;id<vertexArrSize;id+=N)
    	            {temp.combine(anl[id]);}
    	    }); }).wait();
	subgraph.numActiveNodes=*activeSum;
	if(activeSum==0)return;

	uint *next = malloc_device<uint>(vertexArrSize, q);

    d_prefixLabeling = inclusive_scan(d_activeNodesLabeling, next, vertexArrSize, q);

	makeQueue(subgraph.d_activeNodes, inactiveNodeD, d_prefixLabeling, vertexArrSize,q);
    q.memcpy(subgraph.activeNodes, subgraph.d_activeNodes, subgraph.numActiveNodes*sizeof(unsigned int)).wait();

	uint *next2 = malloc_device<uint>(vertexArrSize, q);
	
	d_prefixSumDegrees = inclusive_scan(d_activeNodesDegree, next2, vertexArrSize, q);
	
	makeActiveNodesPointer(subgraph.d_activeNodesPointer, inactiveNodeD, d_prefixLabeling, d_prefixSumDegrees,vertexArrSize, q);

     q.memcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, subgraph.numActiveNodes*sizeof(ull)).wait();


	unsigned int numActiveEdges = 0;
	if(subgraph.numActiveNodes>0)
		numActiveEdges = subgraph.activeNodesPointer[subgraph.numActiveNodes-1] + graph.degree[subgraph.activeNodes[subgraph.numActiveNodes-1]];	
	unsigned int last = numActiveEdges;


	q.memcpy(subgraph.d_activeNodesPointer+subgraph.numActiveNodes, &last, sizeof(ull));
	q.memcpy(subgraph.activeNodesPointer, subgraph.d_activeNodesPointer, (subgraph.numActiveNodes+1)*sizeof(ull));

	free(next,q);
	free(next2,q);

	unsigned int numThreads = NUM_THREADS;

	if(subgraph.numActiveNodes < 50000){
		for(unsigned int i=0; i<subgraph.numActiveNodes; i++)
		{
			unsigned int thisNode = subgraph.activeNodes[i];
			unsigned int thisDegree = graph.degree[thisNode];
			ull fromHere = subgraph.activeNodesPointer[i];
			ull fromThere = graph.nodePointers[thisNode];
			for(unsigned int j=0; j<thisDegree; j++)
			{
				subgraph.activeEdgeList[fromHere+j] = graph.edgeArray[fromThere+j];
			}
		}



	}

	else{

	pthread_t* pthread_p=new pthread_t[numThreads];
	params* params_p = new params[numThreads];
	
	for(int t=0; t<numThreads; t++)
	{
		params_p[t].tId=t;
		params_p[t].numThreads=numThreads;	
		params_p[t].numActiveNodes=subgraph.numActiveNodes;
		params_p[t].activeNodes=subgraph.activeNodes;
		params_p[t].outDegree=graph.outDegree; 
		params_p[t].activeNodesPointer=subgraph.activeNodesPointer;
		params_p[t].nodePointer=graph.nodePointers; 
		params_p[t].activeEdgeList=subgraph.activeEdgeList;
		params_p[t].edgeList=graph.edgeArray;
		pthread_create(&pthread_p[t], 0,dynamic,&params_p[t]);
	}
		
	for (int t=0; t<numThreads; t++)
    	pthread_join(pthread_p[t], NULL);
	}
	q.wait_and_throw();
}

};

#endif	//	SUBGRAPH_GENERATOR_H
