#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H


#include "globals.h"
#pragma once

using namespace sycl;
const int global_size = 1024*108;

typedef unsigned long long ull;
 void
setLabelDefault(uint activeNum, uint *activeNodes, uint *labelD);

template<class T>
 void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, T *labelD,sycl::queue q);

 void
mixStaticLabel(uint activeNum, uint *activeNodes, uint *labelD1, uint *labelD2, uint *isInD);

 void
mixDynamicPartLabel(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1, uint *labelD2);

 void
setDynamicPartLabelTrue(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1,
                        uint *labelD2);

 void
mixCommonLabel(uint testNodeNum, uint *labelD1, uint *labelD2);

 void
cleanStaticAndOverloadLabel(uint vertexNum, uint *staticLabel, uint *overloadLabel);

 void
setStaticAndOverloadLabel(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, uint *isInD);

 void
setStaticAndOverloadLabelBool(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, uint *isInD);

 void
setStaticAndOverloadLabel4Pr(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, uint *isInD,
                             uint *fragmentRecordD, uint *nodePointersD, uint fragment_size, uint *degreeD,
                             uint *isFragmentActiveD);

 void
setOverloadActiveNodeArray(uint vertexNum, uint *activeNodes, uint *overloadLabel,
                           uint *activeLabelPrefix);
template <typename T>
 void
setStaticActiveNodeArray(uint vertexNum, uint *activeNodes, T *staticLabel,
                         uint *activeLabelPrefix);

 void
setLabeling(uint vertexNum, uint *labelD, uint *labelingD);

 void
setActiveNodeArray(uint vertexNum, uint *activeNodes, uint *activeLabel, uint *activeLabelPrefix);

 void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

 void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          uint *activeLabel, uint *activeLabelPrefix, uint *isInList, uint *degreeD);

 void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

 void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, uint *isInD);

template <class T, typename E>
 void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, E *activeNodePointers, T *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD);

 void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, uint *isInStatic);

 void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                      uint *staticFragmentDataD);

 void
setFragmentDataOpt(uint *staticFragmentData, uint staticFragmentNum, uint *staticFragmentVisitRecordsD);

 void
recordFragmentVisit(uint *activeNodeListD, uint activeNodeNum, uint *nodePointersD, uint *degreeD, uint fragment_size,
                    uint *fragmentRecordsD);

 void
bfsKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const uint *isActiveNodeListD,
                          const uint *nodePointersD,
                          const uint *edgeListD, const uint *degreeD, uint *valueD, uint *nextActiveNodeListD);


 void
setStaticAndOverloadLabelAndRecord(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel,
                                   uint *isInD, uint *vertexVisitRecordD);

template<typename T, typename E>
 void
sssp_kernelDynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                      const uint *degreeD,
                      uint *valueD,
                      T *isActiveD, EdgeWithWeight *edgeListOverloadD,
                      const E *activeOverloadNodePointersD,
                      sycl::queue q) {
        
        q.submit([&](handler& h) {
            h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
                auto tid = item.get_global_id()[0];
                for (uint i = tid; i < overloadNodeNum; i += N) {
                    uint traverseIndex = overloadStartNode + i;
                    //get the node index from original graph
                    uint id = overloadNodeListD[traverseIndex];
                    //get the value of current node from original graph
                    uint sourceValue = valueD[id];
                    uint finalValue = sourceValue + 1;
                    for (uint i = 0; i < degreeD[id]; i++) {
                        EdgeWithWeight checkNode{};          
                        checkNode = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                          activeOverloadNodePointersD[overloadStartNode] + i];
                        finalValue = sourceValue + checkNode.weight;
                        uint vertexId = checkNode.toNode;
                        
                        if (finalValue < valueD[vertexId]) {
                            isActiveD[vertexId] = 1;
                            //here atomic
                            auto min_atomic = atomic_ref<uint, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space>(valueD[vertexId]);
                            min_atomic.fetch_min(finalValue);
                        }
                    }
                }
            });
    }).wait();
}

template<typename T, typename E>
 void
sssp_kernelDynamic2(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                      const uint *degreeD,
                      uint *valueD,
                      T *isActiveD, EdgeWithWeight *edgeListOverloadD,
                      const E *activeOverloadNodePointersD,
                      sycl::queue q) {
        
        for(uint i = 0; i < overloadNodeNum; i ++){
            uint traverseIndex = overloadStartNode + i;
            uint id = overloadNodeListD[traverseIndex];
            int id_count = 0;
            uint sourceValue = valueD[id];
            uint finalValue = sourceValue + 1;
            for (uint i = 0; i < degreeD[id]; i++) {
                EdgeWithWeight checkNode{};          
                checkNode = edgeListOverloadD[activeOverloadNodePointersD[traverseIndex] -
                                    activeOverloadNodePointersD[overloadStartNode] + i];
                finalValue = sourceValue + checkNode.weight;
                uint vertexId = checkNode.toNode;
                
                if (finalValue < valueD[vertexId]) {
                    id_count ++;
                    isActiveD[vertexId] = 1;
                    auto min_atomic = atomic_ref<uint, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space>(valueD[vertexId]);
                    min_atomic.fetch_min(finalValue);
                }
            }
        }
}


template<class T>
 void
sssp_kernel(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, 
        EdgeWithWeight *edgeListD, uint *valueD, T *labelD,sycl::queue q) 
{
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint i = tid; i < nodeNum; i += N) {
                uint id = activeNodesD[i];
                uint edgeIndex = nodePointersD[id];
                uint sourceValue = valueD[id];
                uint finalValue;
                for (uint i = edgeIndex; i < edgeIndex +degreeD[id]; i++) {
                    finalValue = sourceValue + edgeListD[i].weight;
                    uint vertexId;
                    vertexId = edgeListD[i].toNode;
                    if (finalValue < valueD[vertexId]) {
                        auto min_atomic = atomic_ref<uint, sycl::memory_order::relaxed, memory_scope::device, access::address_space::global_space>(valueD[vertexId]);
                        min_atomic.fetch_min(finalValue);
                        labelD[vertexId] = 1;
                    }
                }
            }
        });
    });
}


void setStaticAndOverloadLabelBool(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, uint *isInD,sycl::queue q) {

    q.submit([&](handler& h) {
        h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < vertexNum; i += global_size) {
                if (activeLabel[i]) {
                    if (isInD[i]) {
                        staticLabel[i] = 1;
                    } else {
                        overloadLabel[i] = 1;
                    }
                }
            }
        });
    }).wait();
}

template<class T>
 void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, T *labelD,sycl::queue q) {
    q.submit([&](handler& h) {
        h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < activeNum; i += global_size) {
                if (labelD[activeNodes[i]]) 
                    labelD[activeNodes[i]] = 0;
            }
        });
    }).wait();
}


template <class T, typename E>
 void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, E *activeNodePointers, T *activeLabel,uint *activeLabelPrefix, uint *degreeD, sycl::queue q) {
    q.submit([&](handler& h) {
        h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < vertexNum; i += global_size) {
            if (activeLabel[i]) {
                activeNodes[activeLabelPrefix[i]] = i;
                activeNodePointers[activeLabelPrefix[i]] = degreeD[i];
                activeLabel[i] = 0;
                }
            }
        });
    }).wait();
}


template <typename T>
 void setStaticActiveNodeArray(uint vertexNum, uint *activeNodes, T *staticLabel,
                         uint *activeLabelPrefix,sycl::queue q) {
    q.submit([&](handler& h) {
        h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < vertexNum; i += global_size) {
                if (staticLabel[i]) {
                activeNodes[activeLabelPrefix[i]] = i;
                staticLabel[i] = 0;
        }
            }
        });
    }).wait();
}
uint* myExclusive_scan(uint* current, uint* next, unsigned int N, queue& q) {
    unsigned int two_power = 1;
    unsigned int num_iter = ceil(log2(N));
    uint* result = NULL;
    // Iterate over the necessary iterations.
    for (unsigned int iter = 0; iter < num_iter; iter++, two_power *= 2) {
        if (iter == 0) {
        q.submit([&](auto& h) {
            
            h.parallel_for(global_size, [=](sycl::id<1> j) {
                if(j == 0){
                    next[0] = 0;
                }
                for(int i=j;i<N-1;i+=global_size){
                if (i < two_power) {
                next[i+1] = current[i];
                } else {
                next[i+1] = current[i] +current[i - two_power];
                }}
            });  
            result = next;
            }).wait();

            }
        else if (iter % 2 == 0 && iter!=0) {
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
                });//end parallel for loop in kernel
                result = current;
            }).wait();//end device queue
        }//end iteration
	}
    q.wait_and_throw();

  return result;
}
uint myReduction(uint * arr, uint vertexArrSize,sycl::queue q,SIZE_TYPE N,SIZE_TYPE B){
  uint* sum = malloc_shared<uint>(1, q);
  *sum = 0;
  // nd-range kernel parallel_for with reduction parameter
	q.submit([&](handler& h) {
    	h.parallel_for(nd_range<1>{N, B}, 
    	    reduction(sum, sycl::plus<>()), [=](nd_item<1> it, auto& temp) {
    	        auto i = it.get_global_id(0);
    	        for(int id=i;id<vertexArrSize;id+=N)
    	            {temp.combine(arr[id]);}
    	    }); }).wait();
  return *sum;
}

template <typename T, typename K>
void prKernel_Opt(uint nodeNum, K *valueD, K *sumD, T *isActiveNodeList,sycl::queue q) {
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
            auto tid = item.get_global_id()[0];
            for (uint index = tid; index < nodeNum; index += N) {
                if (isActiveNodeList[index]) {
                    K tempValue = 0.15 + 0.85 * sumD[index];
                    K diff = tempValue > valueD[index] ? (tempValue - valueD[index]) :  (valueD  [index] - tempValue);

                    if (diff > 0.001) {
                        isActiveNodeList[index] = 1;
                        valueD[index] = tempValue;
                        sumD[index] = 0;
                    } else {
                        isActiveNodeList[index] = 0;
                        sumD[index] = 0;
                    }
                }
            }
        });
    }).wait();

}

#endif