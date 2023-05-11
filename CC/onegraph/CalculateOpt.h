#include "GraphMeta.h"
#include "gpu_kernels.h"
#include "TimeRecord.h"
#pragma once

void cc_opt(string path, float adviseRate) {

    uint *temp_isStaticActive;
    uint *scan_temp;

    cout << "======cc_opt=======" << "\n";
    GraphMeta<uint> graph;
    graph.setAlgType(CC);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 17);
    graph.initGraphHost();
    graph.initGraphDevice();

    uint activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);
    
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");


    EDGE_POINTER_TYPE overloadEdges = 0; 

    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        uint nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        temp_isStaticActive = malloc_device<uint>(graph.vertexArrSize, graph.streamStatic);
        scan_temp = malloc_device<uint>(graph.vertexArrSize,graph.streamStatic);
        while (activeNodesNum) {
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD,graph.streamStatic);

            uint staticNodeNum = myReduction(graph.isStaticActive,graph.vertexArrSize,graph.streamStatic,N,B);

            if (staticNodeNum > 0) {
                graph.streamStatic.memcpy(temp_isStaticActive, graph.isStaticActive, sizeof(uint) * graph.vertexArrSize).wait();
                graph.prefixSumTemp=myExclusive_scan(temp_isStaticActive, scan_temp, graph.vertexArrSize,graph.streamStatic);

                setStaticActiveNodeArray<uint>(graph.vertexArrSize, graph.staticNodeListD,
                                               graph.isStaticActive,graph.prefixSumTemp,graph.streamStatic);
            }
            uint overloadNodeNum = myReduction(graph.isOverloadActive,graph.vertexArrSize,graph.streamStatic,N,B);
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            
            if (overloadNodeNum > 0) {
                graph.streamStatic.memcpy(temp_isStaticActive, graph.isOverloadActive, sizeof(uint) * graph.vertexArrSize).wait();
                graph.prefixSumTemp=myExclusive_scan(temp_isStaticActive,scan_temp, graph.vertexArrSize,graph.streamStatic);
                setOverloadNodePointerSwap(graph.vertexArrSize, graph.overloadNodeListD, graph.activeOverloadDegreeD,graph.isOverloadActive,graph.prefixSumTemp, graph.degreeD,graph.streamDynamic);
                graph.streamDynamic.memcpy(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint)).wait();
                EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                graph.streamDynamic.memcpy(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                unsigned long long ankor = 0;
                graph.streamDynamic.wait();
                
                for(unsigned i = 0; i < overloadNodeNum; ++i) {
                    overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                    if(i > 0) {
                        ankor += overloadDegree[i - 1];
                    }
                    graph.activeOverloadNodePointers[i] = ankor;
                }
                graph.streamDynamic.memcpy(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                overloadEdges += overloadEdgeNum;
            }

            if (staticNodeNum > 0) {
                setLabelDefaultOpt(staticNodeNum,graph.staticNodeListD,graph.isActiveD,graph.streamStatic);
            }
            if (overloadNodeNum > 0) {
                setLabelDefaultOpt(overloadNodeNum,graph.overloadNodeListD,graph.isActiveD,graph.streamDynamic);
            }
            preProcess.endRecord();
            staticProcess.startRecord();
            
            if(staticNodeNum > 0){
                cc_kernelStaticSwap(staticNodeNum, graph.staticNodeListD,
                                    graph.staticNodePointerD,
                                    graph.degreeD,
                                    graph.staticEdgeListD, graph.valueD,
                                    graph.isActiveD, graph.isInStaticD,graph.streamStatic);
            }

            if (overloadNodeNum > 0) {
                graph.streamDynamic.memcpy(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint)).wait();
                graph.streamDynamic.memcpy(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                graph.streamDynamic.wait();
                graph.streamStatic.wait();
                staticProcess.endRecord();
                overloadProcess.startRecord();
                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) {
                    overloadMoveProcess.startRecord();

                    graph.streamDynamic.memcpy(graph.overloadEdgeListD, graph.overloadEdgeList +graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint)).wait();
                    overloadMoveProcess.endRecord();
                    
                    cc_kernelDynamicSwap(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, graph.degreeD,
                            graph.valueD, graph.isActiveD,
                            graph.overloadEdgeListD,
                            graph.activeOverloadNodePointersD,
                            graph.streamDynamic);
                }
                overloadProcess.endRecord();

            } else {
                graph.streamStatic.wait();
                staticProcess.endRecord();
            }
            graph.streamStatic.wait();
            preProcess.startRecord();
            activeNodesNum = myReduction(graph.isActiveD, graph.vertexArrSize,graph.streamStatic,N,B);
            nodeSum += activeNodesNum;
            preProcess.endRecord();
        }
        free(temp_isStaticActive, graph.streamStatic);
        free(scan_temp,graph.streamStatic);
        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << "\n";
        cout << "move overload size : " << overloadEdges * sizeof(uint) << "\n";
    }
    
}
