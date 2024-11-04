#pragma once
#include <stdexcept>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <functional>

#include <cuda.h>
#include <cuda_runtime.h>

namespace tcmp
{
#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

	/// Checks the result of a cuXXXXXX call and throws an error on failure
#define CU_CHECK_THROW(x)                                                                          \
	do {                                                                                           \
		CUresult result = x;                                                                       \
		if (result != CUDA_SUCCESS) {                                                              \
			const char *msg;                                                                       \
			cuGetErrorName(result, &msg);                                                          \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + msg);   \
		}                                                                                          \
	} while(0)


	/// Checks the result of a cudaXXXXXX call and throws an error on failure
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess)                                                                                        \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));  \
	} while(0)


#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define TCMP_HOST_DEVICE __host__ __device__
#define TCMP_DEVICE __device__
#define TCMP_HOST __host__
#else
#define TCMP_HOST_DEVICE
#define TCMP_DEVICE
#define TCMP_HOST
#endif

	int cuda_device_count();
	int cuda_device();
	void set_cuda_device(int device);

	bool cuda_supports_virtual_memory(int device);
	bool cuda_supports_virtual_memory();
	std::string cuda_device_name(int device);
	std::string cuda_device_name();
	uint32_t cuda_compute_capability(int device);
	uint32_t cuda_compute_capability();
	size_t cuda_memory_granularity(int device);
	size_t cuda_memory_granularity();

	struct MemoryInfo
	{
		size_t total;
		size_t free;
		size_t used;
	};

	MemoryInfo cuda_memory_info();

	class ScopeGuard
	{
	public:
		ScopeGuard() = default;
		ScopeGuard(const std::function<void()>& callback) : m_callback{ callback } {}
		ScopeGuard(std::function<void()>&& callback) : m_callback{ std::move(callback) } {}
		ScopeGuard& operator=(const ScopeGuard& other) = delete;
		ScopeGuard(const ScopeGuard& other) = delete;
		ScopeGuard& operator=(ScopeGuard&& other) { std::swap(m_callback, other.m_callback); return *this; }
		ScopeGuard(ScopeGuard&& other) { *this = std::move(other); }
		~ScopeGuard() { if (m_callback) { m_callback(); } }

		void disarm() { m_callback = {}; }
	private:
		std::function<void()> m_callback;
	};

	template <typename T>
	TCMP_HOST_DEVICE T div_round_up(T val, T divisor)
	{
		return (val + divisor - 1) / divisor;
	}

	template <typename T>
	TCMP_HOST_DEVICE T next_multiple(T val, T divisor)
	{
		return div_round_up(val, divisor) * divisor;
	}

	template <typename T>
	TCMP_HOST_DEVICE T previous_multiple(T val, T divisor)
	{
		return (val / divisor) * divisor;
	}

	constexpr uint32_t batch_size_granularity = 128;
	constexpr uint32_t n_threads_linear = 128;

	template <typename T>
	constexpr uint32_t n_blocks_linear(T n_elements)
	{
		return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
	}

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
	template <typename K, typename T, typename ... Types>
	inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args)
	{
		if (n_elements <= 0)
			return;
		kernel << <n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream >> > (n_elements, args...);
	}
#endif

}