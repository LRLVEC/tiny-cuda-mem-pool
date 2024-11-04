/** @file   bench-memory.cu
 *  @author Yuanxing Duan, PKU
 *  @brief  Compare GPUMemoryArena with GPUMemory.
 */

 /*
 * Bench content:
 * On RTX3090, use 16GB at maximum, continusly malloc and free small blocks with random
 * block size from 1MB to 64MB, the blocks are aligned to 128B and distributed exponentially.
 *
 * Results: GPUMemoryArena is good at efficiency but bad at utilization rate.
 */

#include <tiny-cuda-mem-pool/gpu_memory.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tcmp
{
	constexpr size_t max_mem_size = 1ll << 33;
	constexpr int times = 20000;

	struct Action
	{
		size_t size;
		uint32_t id;
		bool isMalloc;
	};

	std::vector<Action> bench_GPUMemory()
	{
		constexpr int min_log = 10;
		constexpr int max_log = 30;
		constexpr size_t min_size = 1ll << min_log;
		constexpr size_t max_size = 1ll << max_log;

		std::mt19937 mt(1337);
		std::uniform_int_distribution<int> rd_log(min_log, max_log - 1);
		auto get_random_size = [&rd_log, &mt]() {
			int lg = rd_log(mt);
			std::uniform_int_distribution<size_t> rd(1 << lg, 1 << (lg + 1));
			return next_multiple<size_t>(rd(mt), 128);};

		uint32_t current_id(0);
		size_t total_size(0);
		std::vector<Action> actions;
		std::unordered_map<uint32_t, GPUMemory<char>> blocks;

		auto start = std::chrono::high_resolution_clock::now();
		while (actions.size() < times)
		{
			if (total_size < max_mem_size)
			{
				size_t sz = get_random_size();
				blocks[current_id] = GPUMemory<char>(sz);
				blocks[current_id].memset(0);
				actions.push_back(Action{ sz, current_id, true });
				total_size += blocks[current_id].get_bytes();
				current_id += 1;
			}
			else
			{
				actions.push_back(Action{ blocks.begin()->second.get_bytes(),blocks.begin()->first,false });
				total_size -= blocks.begin()->second.get_bytes();
				// blocks.begin()->second.free_memory();
				blocks.erase(blocks.begin());
			}
			if (actions.size() % 1000 == 0)
			{
				printf("%zu: %zu B %zu B utilization %.2f%%\n",
					actions.size(),
					total_size,
					cuda_memory_info().used,
					100 * float(total_size) / cuda_memory_info().used
				);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Free and malloc " << actions.size() << " times:" << duration.count() << " ms\n";
		return actions;
	}

	void bench_GPUMemoryArena(std::vector<Action>const& actions)
	{
		std::unordered_map<uint32_t, GPUMemoryArena::Allocation> blocks;
		size_t total_size(0);

		auto start = std::chrono::high_resolution_clock::now();

		uint32_t c0(0);
		for (Action const& action : actions)
		{
			if (action.isMalloc)
			{
				blocks[action.id] = allocate_workspace(nullptr, action.size);
				total_size += action.size;
			}
			else
			{
				//blocks[action.id].~Allocation();
				blocks.erase(action.id);
				total_size -= action.size;
			}
			c0 += 1;
			if (c0 % 1000 == 0)
			{
				printf("%u: %zu B %zu B utilization %.2f%% Occupied %zu free %zu\n",
					c0,
					total_size,
					cuda_memory_info().used,
					100.0f * float(total_size) / cuda_memory_info().used,
					global_gpu_memory_arenas()[cuda_device()]->block_num(),
					global_gpu_memory_arenas()[cuda_device()]->free_block_num()
				);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Free and malloc " << actions.size() << " times:" << duration.count() << " ms\n";
	}
}

int main(int argc, char* argv[])
{
	using namespace tcmp;

	try
	{
		uint32_t compute_capability = cuda_compute_capability();

		// init
		GPUMemory<float> init(4096);
		init.free_memory();
		//bench
		bench_GPUMemoryArena(bench_GPUMemory());
	}
	catch (std::exception& e)
	{
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return 0;
}

