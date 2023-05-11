            

//#ifndef PTGRAPH_CALCULATEOPT_CUH
//#define PTGRAPH_CALCULATEOPT_CUH
#include "GraphMeta.h"
#
#include "gpu_kernels.h"
#include "TimeRecord.h"
#pragma once


void pr_opt(string path, uint sourceNode, double adviseRate);
void pr_opt(string path, double adviseRate) {

    // Auxiliary array
    uint *temp_isActive;
    uint *scan_temp;
    
    cout << "======pr_opt=======" << "\n";
    GraphMeta<uint> graph;
    graph.setAlgType(PR);
    graph.readDataFromFile(path, true);
    graph.setPrestoreRatio(adviseRate, 19);
    graph.initGraphHost();
    graph.initGraphDevice();

    uint activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);
    //cout<<activeNodesNum<<"   ======\n";

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

    EDGE_POINTER_TYPE overloadEdges = 0; 

    int testTimes = 1;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        uint nodeSum = activeNodesNum;
        int iter = 1;
        totalProcess.startRecord();
        temp_isActive = malloc_device<uint>(graph.vertexArrSize, graph.streamStatic);
        scan_temp = malloc_device<uint>(graph.vertexArrSize,graph.streamStatic);

        while (activeNodesNum) {
            
            //cout<<"ier = "<<iter<<"\t==== activeNodesNum = "<<activeNodesNum<<"\n";
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD,graph.streamStatic);

            uint staticNodeNum = myReduction(graph.isStaticActive,graph.vertexArrSize,graph.streamStatic,N,B);

            //cout<<"staticNodeNum "<<staticNodeNum<<"\n";
            if (staticNodeNum > 0) {
                
                graph.streamStatic.memcpy(temp_isActive, graph.isStaticActive, sizeof(uint) * graph.vertexArrSize).wait();

                graph.prefixSumTemp=myExclusive_scan(temp_isActive, scan_temp, graph.vertexArrSize,graph.streamStatic);

                setStaticActiveNodeArray<uint>(graph.vertexArrSize, graph.staticNodeListD,
                                               graph.isStaticActive,graph.prefixSumTemp,graph.streamStatic);
                
                
            }


            activeNodesNum = myReduction(graph.isActiveD, graph.vertexArrSize,graph.streamStatic,N,B);
            uint overloadNodeNum = myReduction(graph.isOverloadActive,graph.vertexArrSize,graph.streamStatic,N,B);
            //cout<<" overloadNodeNum is " << overloadNodeNum << "\n";
            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            
            if (overloadNodeNum > 0) {
                
                graph.streamStatic.memcpy(temp_isActive, graph.isOverloadActive, sizeof(uint) * graph.vertexArrSize).wait();
                graph.prefixSumTemp=myExclusive_scan(temp_isActive,scan_temp, graph.vertexArrSize,graph.streamStatic);

                setOverloadNodePointerSwap(graph.vertexArrSize, graph.overloadNodeListD, graph.activeOverloadDegreeD,graph.isOverloadActive,graph.prefixSumTemp, graph.degreeD,graph.streamDynamic);


                forULLProcess.startRecord();
                // change wait status
                graph.streamDynamic.memcpy(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint)).wait();
                EDGE_POINTER_TYPE * overloadDegree = new EDGE_POINTER_TYPE[overloadNodeNum];
                graph.streamDynamic.memcpy(overloadDegree, graph.activeOverloadDegreeD, overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                unsigned long long ankor = 0;

                // Try Asyn
                // graph.streamDynamic.wait();
                // graph.streamStatic.wait();

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

            preProcess.endRecord();
            staticProcess.startRecord();

            if(staticNodeNum > 0){
                prSumKernel_static(staticNodeNum, graph.staticNodeListD,graph.staticNodePointerD, graph.degreeD,graph.staticEdgeListD,graph.outDegreeD,graph.valuePrD,graph.sumD,graph.isActiveD,graph.streamStatic);
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

                    prSumKernel_dynamic(
                            graph.partEdgeListInfoArr[i].partStartIndex,
                            graph.partEdgeListInfoArr[i].partActiveNodeNums,
                            graph.overloadNodeListD, 
                            graph.activeOverloadNodePointersD,
                            graph.overloadEdgeListD,
                            graph.degreeD, graph.outDegreeD,
                            graph.valuePrD,
                            graph.sumD,
                            graph.isActiveD,
                            graph.streamDynamic);

                }
                overloadProcess.endRecord();

            } else {
                graph.streamStatic.wait();
                staticProcess.endRecord();
            }

            preProcess.startRecord();
            prKernel_Opt(graph.vertexArrSize, graph.valuePrD, graph.sumD, graph.isActiveD,graph.streamStatic);
            activeNodesNum = myReduction(graph.isActiveD, graph.vertexArrSize,graph.streamStatic,N,B);
            nodeSum += activeNodesNum;
            preProcess.endRecord();

            graph.streamStatic.wait(); 

        }
        
        free(temp_isActive, graph.streamStatic);
        free(scan_temp,graph.streamStatic);
        totalProcess.endRecord();
        totalProcess.print();
        preProcess.print();
        staticProcess.print();
        overloadProcess.print();
        forULLProcess.print();
        overloadMoveProcess.print();

        cout << "move overload size : " << overloadEdges * sizeof(uint) << "\n";
    }
    
}
