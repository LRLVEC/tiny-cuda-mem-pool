#include <tiny-cuda-mem-pool/gpu_memory.h>

namespace tcmp
{
	void free_gpu_memory_arena(cudaStream_t stream)
	{
		if (stream)
			stream_gpu_memory_arenas().erase(stream);
		else
			global_gpu_memory_arenas().erase(cuda_device());
	}

	void free_all_gpu_memory_arenas()
	{
		stream_gpu_memory_arenas().clear();
		global_gpu_memory_arenas().clear();
	}
}