            

//#ifndef PTGRAPH_CALCULATEOPT_CUH
//#define PTGRAPH_CALCULATEOPT_CUH
#include "GraphMeta.h"
#
#include "gpu_kernels.h"
#include "TimeRecord.h"
#pragma once


void sssp_opt(string path, uint sourceNode, double adviseRate) {

    // Auxiliary array
    uint *temp_isOverloadActive;
    uint *scan_temp;
    
    cout << "======sssp_opt=======" << "\n";
    GraphMeta<EdgeWithWeight> graph;
    graph.setAlgType(SSSP);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 15 + 2);
    graph.initGraphHost();
    graph.initGraphDevice();

    uint activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);

    EDGE_POINTER_TYPE overloadEdges = 0; 

    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {

        TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
        TimeRecord<chrono::milliseconds> preProcess("preProcess");
        TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
        TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
        TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");
        TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
        totalProcess.startRecord();
        totalProcess.endRecord();
        totalProcess.print();
        totalProcess.clearRecord();
        cout<<"graph.sourceNode = "<<graph.sourceNode<<"\n";

        graph.refreshLabelAndValue();

        activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);
        scan_temp = malloc_device<uint>(graph.vertexArrSize,graph.streamStatic);

        uint nodeSum = activeNodesNum;
        int iter = 0;
        totalProcess.startRecord();
        temp_isOverloadActive = malloc_device<uint>(graph.vertexArrSize, graph.streamStatic);

        while (activeNodesNum) {
            
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD,graph.streamStatic);

            uint staticNodeNum = myReduction(graph.isStaticActive,graph.vertexArrSize,graph.streamStatic,N,B);

            if (staticNodeNum > 0) {
                
                graph.streamStatic.memcpy(temp_isOverloadActive, graph.isStaticActive, sizeof(uint) * graph.vertexArrSize).wait();

                graph.prefixSumTemp=myExclusive_scan(temp_isOverloadActive, scan_temp, graph.vertexArrSize,graph.streamStatic);

                setStaticActiveNodeArray<uint>(graph.vertexArrSize, graph.staticNodeListD,
                                               graph.isStaticActive,graph.prefixSumTemp,graph.streamStatic);
            }


            activeNodesNum = myReduction(graph.isActiveD, graph.vertexArrSize,graph.streamStatic,N,B);
            uint overloadNodeNum = myReduction(graph.isOverloadActive,graph.vertexArrSize,graph.streamStatic,N,B);
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            
            if (overloadNodeNum > 0) {
                
                graph.streamStatic.memcpy(temp_isOverloadActive, graph.isOverloadActive, sizeof(uint) * graph.vertexArrSize).wait();
                graph.prefixSumTemp=myExclusive_scan(temp_isOverloadActive,scan_temp, graph.vertexArrSize,graph.streamStatic);

                setOverloadNodePointerSwap(graph.vertexArrSize, graph.overloadNodeListD, graph.activeOverloadDegreeD,graph.isOverloadActive,graph.prefixSumTemp, graph.degreeD,graph.streamDynamic);


                forULLProcess.startRecord();
                // change wait status
                graph.streamDynamic.memcpy(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint)).wait();
                EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                graph.streamDynamic.memcpy(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                unsigned long long ankor = 0;

                for(unsigned i = 0; i < overloadNodeNum; ++i) {
                    overloadEdgeNum += graph.degree[graph.overloadNodeList[i]];
                    if(i > 0) {
                        ankor += overloadDegree[i - 1];
                    }
                    graph.activeOverloadNodePointers[i] = ankor;
                    if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                        cout << i << " : " << graph.activeOverloadNodePointers[i];
                    }
                }

                graph.streamDynamic.memcpy(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE));

                forULLProcess.endRecord();
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
                sssp_kernel(staticNodeNum, graph.staticNodeListD,graph.staticNodePointerD, graph.degreeD,graph.staticEdgeListD, graph.valueD,graph.isActiveD,graph.streamStatic);
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

                    graph.streamDynamic.memcpy(graph.overloadEdgeListD, graph.overloadEdgeList +graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(EdgeWithWeight)).wait();
                    overloadMoveProcess.endRecord();

                    sssp_kernelDynamic(
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

            preProcess.startRecord();
            activeNodesNum = myReduction(graph.isActiveD, graph.vertexArrSize,graph.streamStatic,N,B);
            nodeSum += activeNodesNum;
            preProcess.endRecord();

            graph.streamStatic.wait(); 

        }
        
        free(temp_isOverloadActive, graph.streamStatic);
        free(scan_temp,graph.streamStatic);
        cout<<"total iter = "<<iter<<"\n";
        uint* v=new uint[graph.vertexArrSize];
        graph.streamDynamic.memcpy(v, graph.valueD, graph.vertexArrSize * sizeof(uint)).wait();
        cout<<"below print first 10 nodes and the last node as a reference for correctness\n";
        for(uint i=0;i<10;i++)
        cout<<v[i]<<"\t";
        cout<<"..."<<v[graph.vertexArrSize-1]<<"\n";
        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();
        cout << "nodeSum : " << nodeSum << "\n";
        cout<<"nodeSum is the summary of nodes that have been traversed, which represents the amount of calculation";
        graph.sourceNode = graph.findbignode(sourceNode);
        cout << "move overload size : " << overloadEdges * sizeof(uint) << "\n";
    }
    
}
