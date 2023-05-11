#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H


#include "globals.h"
#pragma once

using namespace sycl;
const int global_size = 1024*56;


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


template <typename E, typename K> void
prSumKernel_dynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                    const E *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, K *valueD,
                    K *sumD,uint * isactiveD,sycl::queue q) {
    q.submit([&](handler& h) {
        h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < overloadNodeNum; i += global_size) {
                uint traverseIndex = overloadStartNode + i;
                uint nodeIndex = overloadNodeListD[traverseIndex];

                uint edgeOffset = nodePointersD[traverseIndex] - nodePointersD[overloadStartNode];
            K tempSum = 0;
            for (uint i = edgeOffset; i < edgeOffset + degreeD[nodeIndex]; i++) {
                uint srcNodeIndex = edgeListD[i];
                if(outDegreeD[srcNodeIndex]!=0){
                    K tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                    //printf("src %d dest %d value %f \n", srcNodeIndex,nodeIndex )
                    tempSum += tempValue;
                }
            }
            sumD[nodeIndex] = tempSum;
        }

        });
    }).wait();
}

template<typename T> void
prSumKernel_static(uint activeNum,  uint *activeNodeList,
                    uint *nodePointersD, uint *degreeD,
                    uint *edgeListD, uint *outDegreeD,  T *valueD,
                   T *sumD,uint * isactiveD,sycl::queue q) {
    q.submit([&](handler& h) {
       h.parallel_for(global_size, [=](id<1> tid) {
            for (uint i = tid; i < activeNum; i += global_size) {
            uint nodeIndex = activeNodeList[i];
            uint edgeIndex = nodePointersD[nodeIndex];
            T tempSum = 0;
            for (uint i = edgeIndex; i < edgeIndex + degreeD[nodeIndex]; i++) {
                uint srcNodeIndex = edgeListD[i];
                if(outDegreeD[srcNodeIndex]!=0){
                    T tempValue = valueD[srcNodeIndex] / outDegreeD[srcNodeIndex];
                    tempSum += tempValue;
                }
            }
            sumD[nodeIndex] = tempSum;
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
                // else{
                //     staticLabel[i] = 0;
                //     overloadLabel[i] = 0;
                // }
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
            }}
          });  // end parallel for loop in kernel
          result = current;
      }).wait();  // end device queue
    }      // end iteration
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