#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>

namespace cudaprob3 {

// Standard C++ allocator backed by cudaMallocHost/cudaFreeHost.
// Allows std::vector<T, PinnedAllocator<T>> to use page-locked memory
// for fast async DMA to/from device.
template <typename T>
struct PinnedAllocator {
    using value_type = T;

    PinnedAllocator() noexcept = default;
    template <typename U> PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (cudaMallocHost(&ptr, n * sizeof(T)) != cudaSuccess)
            throw std::bad_alloc{};
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFreeHost(ptr);
    }

    template <typename U>
    bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const PinnedAllocator<U>&) const noexcept { return false; }
};

} // namespace cudaprob3
