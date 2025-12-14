# 作业一：基于 Milvus 的 FAQ 检索系统（LlamaIndex）

## 1. 目标与输入输出
- **输入**：用户自然语言问题（例如“如何退货？”）
- **输出**：最相关的 FAQ 条目（question/answer）及其检索得分（score）

本实现提供两种使用方式：
- **CLI**：`python -m milvus_faq.main query "如何退货？"`
- **REST API**（FastAPI）：`POST /query`，并提供 `POST /admin/reindex` 作为“热更新”入口

## 2. 工程结构
- `milvus_faq/main.py`：入口（CLI + API），包含索引构建、查询、热更新逻辑
- `milvus_faq/data/faqs.jsonl`：示例知识库（JSONL：每行一个 `{question, answer}`）

## 3. 索引与切片策略（语义切分 + 重叠）
索引构建使用 **LlamaIndex**，向量库使用 **Milvus**：
- 文档组织：将每条 FAQ 作为一个 `Document`，文本为 `Q/A` 拼接，答案保存在 metadata（便于直接返回）
- 切片策略：
  - 优先使用 `SemanticSplitterNodeParser`（若依赖可用）进行**语义切分**
  - 否则回退到 `SentenceSplitter(chunk_size=512, chunk_overlap=80)` 实现**重叠切片**

## 4. 热更新（自动 re-index）
本作业实现了两种“热更新”路径（满足 README 扩展项）：
- **手动 reindex**：`POST /admin/reindex` 或 CLI `reindex`
- **自动 reindex**：开启环境变量 `AUTO_REINDEX_ON_QUERY=true` 后，每次查询都会检查 KB 文件 mtime，发现变化则自动重建索引

说明：为了降低依赖与复杂度，这里用“文件 mtime 触发重建”替代文件监听器；在生产里可用 `watchfiles`/消息队列实现更稳健的增量更新。

## 5. REST API 设计
- `GET /health`：健康检查
- `POST /query`：输入 `{question, top_k}`，输出 `best_answer` + `matches`
- `POST /admin/reindex`：触发重建索引（可指定新的 KB 路径）

## 6. 运行方式（本地）
### 6.1 启动 Milvus（示例）
你可以用 Milvus 官方 docker 方式启动（或使用你已有的 Milvus 环境）。

### 6.2 安装依赖（示例）
建议使用 `uv` 或 `pip` 安装（依赖会在 `pyproject.toml` 中声明）。

### 6.3 启动 API
```bash
python -m milvus_faq.main api --host 0.0.0.0 --port 8000
```

### 6.4 调用查询
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"如何退货？","top_k":3}'
```

## 7. 实验现象与结果（可填写）
- **召回效果**：top1 是否命中？topk 是否覆盖？
- **切片影响**：语义切分 vs 固定 chunk + overlap 对相似问题是否更稳定？
- **热更新效果**：修改 `faqs.jsonl` 后是否能自动/手动生效？

（在此处补充你的实际截图/日志/对比表格即可）