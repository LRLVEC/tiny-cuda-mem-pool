#include <tiny-cuda-mem-pool/common.h>

namespace tcmp
{
	int cuda_device_count()
	{
		int device_count;
		CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
		return device_count;
	}

	int cuda_device()
	{
		int device;
		CUDA_CHECK_THROW(cudaGetDevice(&device));
		return device;
	}

	void set_cuda_device(int device)
	{
		CUDA_CHECK_THROW(cudaSetDevice(device));
	}

	bool cuda_supports_virtual_memory(int device)
	{
		int supports_vmm;
		CU_CHECK_THROW(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
		return supports_vmm != 0;
	}
	bool cuda_supports_virtual_memory()
	{
		return cuda_supports_virtual_memory(cuda_device());
	}

	std::string cuda_device_name(int device)
	{
		cudaDeviceProp props;
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
		return props.name;
	}
	std::string cuda_device_name()
	{
		return cuda_device_name(cuda_device());
	}

	uint32_t cuda_compute_capability(int device)
	{
		cudaDeviceProp props;
		CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
		return props.major * 10 + props.minor;
	}
	uint32_t cuda_compute_capability()
	{
		return cuda_compute_capability(cuda_device());
	}

	size_t cuda_memory_granularity(int device)
	{
		size_t granularity;
		CUmemAllocationProp prop = {};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = 0;
		CUresult granularity_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		if (granularity_result == CUDA_ERROR_NOT_SUPPORTED)
		{
			return 1;
		}
		CU_CHECK_THROW(granularity_result);
		return granularity;
	}
	size_t cuda_memory_granularity()
	{
		return cuda_memory_granularity(cuda_device());
	}

	MemoryInfo cuda_memory_info()
	{
		MemoryInfo info;
		CUDA_CHECK_THROW(cudaMemGetInfo(&info.free, &info.total));
		info.used = info.total - info.free;
		return info;
	}
}
