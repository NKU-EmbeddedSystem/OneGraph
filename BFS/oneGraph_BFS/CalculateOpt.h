#include "GraphMeta.h"
#include "gpu_kernels.h"
#include "TimeRecord.h"
#pragma once

bool debug1020 = false;
int testTimes = 5;

void bfs_opt(string path, uint sourceNode, float adviseRate) {

    //uint *temp_isOverloadActive;
    uint *temp_isStaticActive;
    uint *scan_temp;

    cout << "======bfs_opt=======" << "\n";
    GraphMeta<myEdgeType> graph;
    graph.setAlgType(BFS);
    graph.setSourceNode(sourceNode);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(adviseRate, 17);
    graph.initGraphHost();
    graph.initGraphDevice();


    //设置source
    uint *src=new uint[testTimes];
    uint ttt=0;
    for(int i=0;i<testTimes;i++)
    {
        src[i]=graph.findbignode(ttt);
        ttt = src[i];
    }

    uint activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);
    //cout<<activeNodesNum<<"   ======\n";
    
    //TimeRecord<chrono::milliseconds> forULLProcess("forULLProcess");
    TimeRecord<chrono::milliseconds> preProcess("preProcess");
    TimeRecord<chrono::milliseconds> staticProcess("staticProcess");
    TimeRecord<chrono::milliseconds> overloadProcess("overloadProcess");
    TimeRecord<chrono::milliseconds> overloadMoveProcess("overloadMoveProcess");

    TimeRecord<chrono::milliseconds> totalProcess("totalProcess");
    totalProcess.startRecord();

    cout << "vertexArrSize = " << graph.vertexArrSize << " edgeArrSize = " << graph.edgeArrSize << "\n";

    totalProcess.endRecord();
    totalProcess.print();
    totalProcess.clearRecord();

    EDGE_POINTER_TYPE overloadEdges = 0; 

    
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {

        cout<<"----------testIndex = "<<testIndex<<"-----------\n";
        //设置source
        //scout<<"set source = "<<src[testIndex]<<"\n";
        graph.setSourceNode(src[testIndex]);
        graph.refreshLabelAndValue();
        activeNodesNum = myReduction((graph.isActiveD), graph.vertexArrSize,graph.streamStatic,N,B);


        uint nodeSum = activeNodesNum;
        uint staticnodeSum = 0; 
        int iter = 0;
        totalProcess.startRecord();

        temp_isStaticActive = malloc_device<uint>(graph.vertexArrSize, graph.streamStatic);
        scan_temp = malloc_device<uint>(graph.vertexArrSize,graph.streamStatic);
        uint source = src[testIndex];
        // cout<<"degree[source]: "<<graph.degree[source]<<std::endl;

        while (activeNodesNum) {

            //cout<<"\t==== activeNodesNum = "<<activeNodesNum<<"\n";
            iter++;
            preProcess.startRecord();
            setStaticAndOverloadLabelBool(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD,graph.streamStatic);

            uint staticNodeNum = myReduction(graph.isStaticActive,graph.vertexArrSize,graph.streamStatic,N,B);

            staticnodeSum += staticNodeNum;

            if (staticNodeNum > 0) {
                graph.streamStatic.memcpy(temp_isStaticActive, graph.isStaticActive, sizeof(uint) * graph.vertexArrSize).wait();
                graph.prefixSumTemp=myExclusive_scan(temp_isStaticActive, scan_temp, graph.vertexArrSize,graph.streamStatic);

                setStaticActiveNodeArray<uint>(graph.vertexArrSize, graph.staticNodeListD,
                                               graph.isStaticActive,graph.prefixSumTemp,graph.streamStatic);//求subVertex, 或者叫overLoadNodeList
                //cout << "iter " << iter << " staticNodeNum is " << staticNodeNum<<"\n";
            }

            uint overloadNodeNum = myReduction(graph.isOverloadActive,graph.vertexArrSize,graph.streamStatic,N,B);

            EDGE_POINTER_TYPE overloadEdgeNum = 0;
            if (overloadNodeNum > 0) {

                graph.streamStatic.memcpy(temp_isStaticActive, graph.isOverloadActive, sizeof(uint) * graph.vertexArrSize).wait();

                graph.prefixSumTemp=myExclusive_scan(temp_isStaticActive,scan_temp, graph.vertexArrSize,graph.streamStatic);

                setOverloadNodePointerSwap(graph.vertexArrSize, graph.overloadNodeListD, graph.activeOverloadDegreeD,graph.isOverloadActive,graph.prefixSumTemp, graph.degreeD,graph.streamStatic);
                //forULLProcess.startRecord();
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
                    if(graph.activeOverloadNodePointers[i] > graph.edgeArrSize) {
                        //cout << i << " : " << graph.activeOverloadNodePointers[i];
                    }
                }
                graph.streamDynamic.memcpy(graph.activeOverloadNodePointersD, graph.activeOverloadNodePointers, overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                //forULLProcess.endRecord();
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
                bfs_kernelStatic(staticNodeNum, graph.staticNodeListD,graph.staticNodePointerD, graph.degreeD,graph.staticEdgeListD, graph.valueD,graph.isActiveD,graph.streamStatic);
            }

            if (overloadNodeNum > 0) {
                graph.streamDynamic.memcpy(graph.overloadNodeList, graph.overloadNodeListD, overloadNodeNum * sizeof(uint)).wait();
                graph.streamDynamic.memcpy(graph.activeOverloadNodePointers, graph.activeOverloadNodePointersD,overloadNodeNum * sizeof(EDGE_POINTER_TYPE)).wait();
                
                graph.fillEdgeArrByMultiThread(overloadNodeNum);
                graph.caculatePartInfoForEdgeList(overloadNodeNum, overloadEdgeNum);

                // add wait()
                graph.streamDynamic.wait();
                graph.streamStatic.wait();

                staticProcess.endRecord();

                overloadProcess.startRecord();
                for (int i = 0; i < graph.partEdgeListInfoArr.size(); i++) 
                {
                    overloadMoveProcess.startRecord();
                    graph.streamDynamic.memcpy(graph.overloadEdgeListD, graph.overloadEdgeList +graph.activeOverloadNodePointers[graph.partEdgeListInfoArr[i].partStartIndex],graph.partEdgeListInfoArr[i].partEdgeNums * sizeof(uint)).wait();
                    overloadMoveProcess.endRecord();
                    //cout<<"start bfs_kernelDynamicPart\n";
                    //cout<<"graph.partEdgeListInfoArr[i].partStartIndex = "<<graph.partEdgeListInfoArr[i].partStartIndex<<"\n";
                    //cout<<"graph.partEdgeListInfoArr[i].partActiveNodeNums = "<<graph.partEdgeListInfoArr[i].partActiveNodeNums<<"\n";
                
                    bfs_kernelDynamicPart(
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
            // cout<<"iter "<<iter<<" staticNodeNum "<<staticNodeNum<<" overloadNodeNum "<<overloadEdgeNum<< " activeNodesNum "<< activeNodesNum<<std::endl;
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
        //forULLProcess.print();
        overloadMoveProcess.print();

        totalProcess.clearRecord();
        preProcess.clearRecord();
        staticProcess.clearRecord();
        overloadProcess.clearRecord();
        overloadMoveProcess.clearRecord();

        cout << "nodeSum : " << nodeSum << "\n";
        cout << "move overload size : " << overloadEdges * sizeof(uint) /1024/1024<< "MB \n";
        cout << "staticnodeSum : " << staticnodeSum << "\n";
    }
    
}
