#pragma once

#include "../physics/params_pod.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cudaprob3 {

// RAII wrapper around a CUDA executable graph.
// Stores a reference to the kernel node so params can be updated without
// re-instantiating the graph (cudaGraphExecKernelNodeSetParams).
struct CUDAGraphHandle {
    cudaGraphExec_t exec  = nullptr;
    cudaGraph_t     graph = nullptr;
    cudaGraphNode_t kernelNode = nullptr;

    CUDAGraphHandle() noexcept = default;

    CUDAGraphHandle(const CUDAGraphHandle&) = delete;
    CUDAGraphHandle& operator=(const CUDAGraphHandle&) = delete;

    CUDAGraphHandle(CUDAGraphHandle&& o) noexcept
        : exec(o.exec), graph(o.graph), kernelNode(o.kernelNode) {
        o.exec = nullptr; o.graph = nullptr; o.kernelNode = nullptr;
    }

    CUDAGraphHandle& operator=(CUDAGraphHandle&& o) noexcept {
        destroy();
        exec = o.exec; graph = o.graph; kernelNode = o.kernelNode;
        o.exec = nullptr; o.graph = nullptr; o.kernelNode = nullptr;
        return *this;
    }

    ~CUDAGraphHandle() { destroy(); }

    [[nodiscard]] bool valid() const noexcept { return exec != nullptr; }

    void destroy() noexcept {
        if (exec)  { cudaGraphExecDestroy(exec);  exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph);      graph = nullptr; }
        kernelNode = nullptr;
    }

    void replay(cudaStream_t stream) const {
        if (cudaGraphLaunch(exec, stream) != cudaSuccess)
            throw std::runtime_error("cudaGraphLaunch failed");
    }
};

} // namespace cudaprob3
