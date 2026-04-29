#include "single_gpu_calculator.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <expected>
#include <stdexcept>

namespace cudaprob3 {

std::expected<ResultView, std::string>
SingleGPUCalculator::calculate(const OscillationParams& params, NeutrinoType type) {
    cudaSetDevice(deviceId_);
    const bool antineutrino = (type == NeutrinoType::Antineutrino);
    const OscParamsPOD pod = params.computePOD(antineutrino);

    if (useCUDAGraphs_ && graphCaptured_) {
        cudaMemcpy(thrust::raw_pointer_cast(d_params_.data()),
                   &pod, sizeof(OscParamsPOD), cudaMemcpyHostToDevice);
        graph_.replay(computeStream_);
    } else if (useCUDAGraphs_ && !graphCaptured_) {
        cudaStreamBeginCapture(computeStream_, cudaStreamCaptureModeGlobal);
        launchKernel(pod, type, 1, computeStream_);
        cudaStreamEndCapture(computeStream_, &graph_.graph);
        cudaGraphInstantiate(&graph_.exec, graph_.graph, nullptr, nullptr, 0);
        graphCaptured_ = true;
        cudaGraphNode_t nodes[64]; std::size_t nNodes = 64;
        cudaGraphGetNodes(graph_.graph, nodes, &nNodes);
        for (std::size_t i = 0; i < nNodes; ++i) {
            cudaGraphNodeType t;
            cudaGraphNodeGetType(nodes[i], &t);
            if (t == cudaGraphNodeTypeKernel) { graph_.kernelNode = nodes[i]; break; }
        }
        graph_.replay(computeStream_);
    } else {
        launchKernel(pod, type, 1, computeStream_);
    }

    cudaEventRecord(computeDone_, computeStream_);
    cudaStreamWaitEvent(xferStream_, computeDone_, 0);
    issueD2H(xferStream_);
    cudaEventRecord(xferDone_, xferStream_);
    cudaEventSynchronize(xferDone_);

    return makeResultView();
}

std::expected<ResultView, std::string> SingleGPUCalculator::waitForResults() {
    cudaSetDevice(deviceId_);
    if (cudaEventSynchronize(xferDone_) != cudaSuccess)
        return std::unexpected("waitForResults: cudaEventSynchronize failed");
    return makeResultView();
}

} // namespace cudaprob3
