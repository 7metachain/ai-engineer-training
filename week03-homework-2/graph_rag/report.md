# 作业二：融合文档检索（RAG）+ 图谱推理（Neo4j）的多跳问答系统

## 1. 场景与目标
**问题示例**：“A 公司的最大股东是谁？”

期望系统流程（与 README 对齐）：
1. **检索 A 公司相关信息（RAG）**：从文档中找出股权结构/相关描述作为证据
2. **图谱中查找控股关系（KG）**：在 Neo4j 里做股权穿透、多跳路径推理
3. **生成最终回答（LLM/模板）**：输出最终股东结果并提供可解释推理路径

## 2. 工程实现概览
- 入口：`python -m graph_rag.main ...`
- 数据：
  - `graph_rag/data/docs.jsonl`：示例文档库
  - `graph_rag/data/ownership_edges.jsonl`：示例股权边（from owns to）

## 3. 文档检索（RAG）
使用 **LlamaIndex** 构建本地向量索引，对公司名相关查询进行检索：
- 目标：召回与“股权/最大股东/结构”相关的文本片段，作为证据展示
- 切片：`SentenceSplitter(chunk_overlap=80)` 保证信息不被硬切断

输出：top-k 的命中文档片段（含相似度 score、标题、文本）。

## 4. Neo4j 多跳推理（Cypher）
### 4.1 图谱建模
- 节点：统一用 `(:Entity {name})` 表示公司/基金/个人等实体
- 边：`(:Entity)-[:OWNS {percent}]->(:Entity)`，表示“股东持有公司股份”
- `percent` 统一归一化到 \([0,1]\)

### 4.2 多跳穿透逻辑
“最大股东”按 **穿透后持股比例乘积最大** 的路径定义：
- 路径：`shareholder -> ... -> company`
- 路径权重：沿路径边权重相乘 \(w = \prod_i percent_i\)
- 在 Neo4j 中用 Cypher 的 `reduce` 对关系列表做乘积，并按 `weight desc` 排序取 top1。

### 4.3 可解释性输出
系统会输出：
- 使用的 Cypher（便于复现）
- 推理路径：`E集团 -[0.6]-> B控股有限公司 -[0.4]-> A公司 ; weight=0.24`

## 5. 联合评分机制（RAG + KG）
为了融合“文档证据”与“图谱推理置信度”，做了一个简单联合评分：
\[
score = \alpha \cdot sim_{rag} + (1-\alpha)\cdot w_{graph}
\]
其中：
- \(sim_{rag}\)：RAG top1 相似度（裁剪到 \([0,1]\)）
- \(w_{graph}\)：图谱路径权重（乘积，天然在 \([0,1]\)）
- \(\alpha\)：可配置（默认 0.6）

## 6. 运行说明
### 6.1 导入 Neo4j（推荐）
先确保 Neo4j 可用，并设置环境变量：
- `NEO4J_URI`（默认 `bolt://localhost:7687`）
- `NEO4J_USER`（默认 `neo4j`）
- `NEO4J_PASSWORD`

然后导入示例边：
```bash
python -m graph_rag.main ingest --edges ./graph_rag/data/ownership_edges.jsonl
```

### 6.2 提问
```bash
python -m graph_rag.main ask "A公司的最大股东是谁？" --top-k 3 --max-hops 3
```

### 6.3 无 Neo4j 的降级演示
如果没配 Neo4j，程序会自动回退到“内存图”用同一份 edges 数据做推理，仍会输出可解释路径（便于跑通作业流程）。

## 7. 实验结果与分析（可填写）
- **RAG 命中情况**：top-k 是否包含股权结构摘要？相似度如何变化？
- **多跳效果**：`max_hops=1` vs `max_hops=3` 是否能从 `B控股 -> A公司` 穿透到 `E集团`？
- **错误传播防御**：
  - 图谱边置信度/来源校验（可扩展）
  - 路径约束（限制 hops、过滤低 percent 边）
  - 与文档证据的一致性检查（例如要求 RAG 中出现股东实体名）

（在此处补充你的实际输出 JSON、截图或对比表格即可）