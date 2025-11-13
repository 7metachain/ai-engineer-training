import numpy as np
import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import os

POOL = ThreadPoolExecutor()

def build_gpu_index(embeddings: np.ndarray):
    """构建GPU向量索引"""
    d = embeddings.shape[1]
    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    index_cpu = faiss.IndexFlatIP(d)
    gpu_funcs_available = hasattr(faiss, "get_num_gpus") and hasattr(faiss, "index_cpu_to_gpu")
    ngpus = 0
    if gpu_funcs_available:
        try:
            ngpus = faiss.get_num_gpus()
        except Exception:
            ngpus = 0
    if gpu_funcs_available and ngpus >= 1:
        try:
            co = faiss.GpuClonerOptions() if hasattr(faiss, "GpuClonerOptions") else None
            if ngpus == 1:
                res = faiss.StandardGpuResources()
                try:
                    res.setTempMemory(512 * 1024 * 1024)
                except Exception:
                    pass
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu, co) if co is not None else faiss.index_cpu_to_gpu(res, 0, index_cpu)
            else:
                index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co) if hasattr(faiss, "index_cpu_to_all_gpus") and co is not None else faiss.index_cpu_to_all_gpus(index_cpu)
            index_gpu.add(embeddings)
            print(f"使用GPU进行索引，GPU数={ngpus}")
            return index_gpu
        except Exception as e:
            print(f"GPU不可用，回退到CPU：{e}")
    index_cpu.add(embeddings)
    print("使用CPU进行索引")
    return index_cpu

async def async_search(index, query_vec: np.ndarray, k=3):
    """异步执行向量搜索（非阻塞）"""
    loop = asyncio.get_running_loop()
    q = np.ascontiguousarray(query_vec.astype('float32'))
    similarities, indices = await loop.run_in_executor(
        POOL,
        lambda: index.search(q, k)
    )
    return similarities, indices

async def main():
    print("=> 构建嵌入数据...")
    np.random.seed(42)
    embeddings = np.random.rand(10000, 128).astype('float32')
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)  # 归一化

    print("=> 创建FAISS-GPU索引...")
    start = time.time()
    index = build_gpu_index(embeddings)
    print(f"索引构建耗时: {time.time() - start:.3f}s")

    # 模拟多个并发查询（如多Agent并行检索）
    queries = np.random.rand(5, 128).astype('float32')
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

    print("=> 并发异步搜索...")
    tasks = [async_search(index, q.reshape(1, -1)) for q in queries]
    results = await asyncio.gather(*tasks)

    for i, (sim, idx) in enumerate(results):
        print(f"查询 {i}: 相似度={sim[0,0]:.3f}, 最近邻ID={idx[0,0]}")

if __name__ == "__main__":
    asyncio.run(main())
