add_executable(bench_memory bench_memory.cu)
target_link_libraries(bench_memory PUBLIC ${TCMP_LIBRARIES} tiny-cuda-mem-pool)