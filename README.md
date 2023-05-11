# OneGraph: A Cross-Architecture Framework for Large-scale Graph Computing on GPUs based on oneAPI.
OneGrpah is a framework for large-scale Graph Computing based on oneAPI. It can process out-of-memory graph data with GPU. 
It significantly reduces the data transfer between GPU and CPU and masks the latency by asynchronous transfer.

OneGraph could be ported to multiple types of accelerators without code modification and performance loss. It follows the design of Ascetic[<sup>1</sup>](#refer-anchor-1) and re-implements it with oneAPI[<sup>2</sup>](#refer-anchor-2). 

The input data should be in CSR format, to convert a txt graph data file into CSR, you can just use the converter offered in [Ascetic](https://github.com/NKU-EmbeddedSystem/Ascetic). 

We re-built in another two out-of-memory graph computing approaches in OneGraph, the SubWay[<sup>3</sup>](#refer-anchor-2) and Unifide Shared Memory based approach.
You can just choose an approach according to your requirement.

## Compilation
To compile OneGrpah, you need to install the oneAPI base toolkit firstly. You can get it at https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

Then, for intel GPU you don't need any other requrements.
For NVIDIA and AMD GPU, a plugin from CodePlay is necessary, you can find it at https://developer.codeplay.com/products/oneapi/nvidia/home/ and https://developer.codeplay.com/products/oneapi/amd/home/ respectively.

After confirming all requirements are ready, you can make the program in cmd(**for NVIDIA or AMD GPU**), or just compile the program with the dpcpp compiler(**for Intel GPU**) in oneAPI base toolkit as follows.
 ```shell
dpcpp main.cpp -o OneGraph
```
*We do not recommend using the -O3 optimization parameters. Because we find it will just lead to negative optimization for OneGraph. The reason for this is unknown and you can try again on your own platform.* 

## Running applications
You can run the application as follows
```
./PR_OneGraph(different for different algo) 
    --type bfs/sssp/cc/pr 
    -f inputfile path 
    -r sourcenode(SSSP and BFS needed)
    --advisek 0.8(the ratio of static region)
```
## Publication
[CCF THPC] Shiyang Li, Jingyu Zhu, Yuting Peng, Jiaxun Han, Zhuoran Wang, Xiaoli Gong, Gang Wang, Jin Zhang, and Xuqiang Wang. OneGraph: A Cross-Architecture Framework for Large-scale Graph Computing on GPUs based on oneAPI.

## References

<div id="refer-anchor-1"></div>

- [1] [ICPP'21] Ruiqi Tang, Ziyi Zhao, Kailun Wang, Xiaoli Gong, Jin Zhang, Wen-wen Wang, and Pen-Chung Yew. [Ascetic: Enhancing Cross-Iterations Data Efficiency in Out-of-Memory Graph Processing on GPUs.]((https://doi.org/10.1145/3472456.3472457))
- [2] [oneAPI: A New Era of Accelerated Computing.](https://www.intel.com/content/www/us/en/software/oneapi.html)
- [3] [EUROSYS'20] Amir Hossein Nodehi Sabet, Zhijia Zhao, and Rajiv Gupta. [Subway: minimizing data transfer during out-of-GPU-memory graph processing.](https://dl.acm.org/doi/abs/10.1145/3342195.3387537) In Proceedings of the Fifteenth European Conference on Computer Systems.

