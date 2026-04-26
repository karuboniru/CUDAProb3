#ifndef CUDAPROB3_ENGINE_MEMORY_POOL_HPP
#define CUDAPROB3_ENGINE_MEMORY_POOL_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace cudaprob3 {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0), deviceId_(0), own_memory_(false) {}

    DeviceBuffer(int deviceId, size_t count) : ptr_(nullptr), size_(0), deviceId_(deviceId), own_memory_(false) {
        allocate(deviceId, count);
    }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_),
          deviceId_(other.deviceId_), own_memory_(other.own_memory_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.own_memory_ = false;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            deviceId_ = other.deviceId_;
            own_memory_ = other.own_memory_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.own_memory_ = false;
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(int deviceId, size_t count) {
        release();
        size_ = count;
        deviceId_ = deviceId;
        own_memory_ = true;
        cudaSetDevice(deviceId_);
        cudaMalloc(&ptr_, count * sizeof(T));
    }

    void allocateAsync(size_t count, cudaStream_t stream) {
        release();
        size_ = count;
        own_memory_ = true;
        cudaMallocAsync(&ptr_, count * sizeof(T), stream);
    }

    void release() {
        if (ptr_ && own_memory_) {
            cudaSetDevice(deviceId_);
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
        own_memory_ = false;
    }

    void releaseAsync(cudaStream_t stream) {
        if (ptr_ && own_memory_) {
            cudaFreeAsync(ptr_, stream);
        }
        ptr_ = nullptr;
        size_ = 0;
        own_memory_ = false;
    }

    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    int deviceId() const { return deviceId_; }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

    void copyFrom(const T* host_src, size_t count, cudaStream_t stream = 0) {
        cudaMemcpyAsync(ptr_, host_src, count * sizeof(T),
                        cudaMemcpyHostToDevice, stream);
    }

    void copyTo(T* host_dst, size_t count, cudaStream_t stream = 0) const {
        cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T),
                        cudaMemcpyDeviceToHost, stream);
    }

private:
    T* ptr_;
    size_t size_;
    int deviceId_;
    bool own_memory_;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() : ptr_(nullptr), size_(0) {}

    explicit PinnedBuffer(size_t count) : ptr_(nullptr), size_(0) {
        allocate(count);
    }

    ~PinnedBuffer() { release(); }

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(size_t count) {
        release();
        size_ = count;
        cudaMallocHost(&ptr_, count * sizeof(T));
    }

    void release() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    size_t size() const { return size_; }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    operator T*() { return ptr_; }
    operator const T*() const { return ptr_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_;
    size_t size_;
};

} // namespace cudaprob3

#endif
