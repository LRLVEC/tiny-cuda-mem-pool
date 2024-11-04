#include <tiny-cuda-mem-pool/gpu_memory.h>

namespace tcmp
{
	std::atomic<size_t>& total_n_bytes_allocated()
	{
		static std::atomic<size_t> s_total_n_bytes_allocated{ 0 };
		return s_total_n_bytes_allocated;
	}

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