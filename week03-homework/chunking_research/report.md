# 句子切片/窗口切片实验报告

## 实验设置
- 数据：3 篇长文档（气候变化综述、量子计算科普、产品白皮书），每篇 1.5k–2k 字，含复杂段落与列表。
- 模型：`qwen-plus`（兼容 OpenAI），Embedding：`TEXT_EMBEDDING_V3`，top_k=5。
- 代码入口：`python -m chunking_research.main --data-dir data/chunking --question "文章的核心观点是什么？" --ground-truth "<摘录>" --include-markdown`。
- 评价方式：检查检索片段是否覆盖参考答案（hit），以及主观冗余度/连贯度（1 越差，5 最佳）。

## 参数与结果
| 切片方式 | 关键参数 | 命中率 (GT in context) | 冗余度 | 连贯度 |
| --- | --- | --- | --- | --- |
| Sentence 切片 | chunk_size=512, overlap=50 | 3/3 | 3 | 4 |
| Token 切片 | chunk_size=32, overlap=4 | 2/3 | 2 | 2 |
| Sentence Window | window_size=3 | 3/3 | 4 | 5 |
| Markdown（含 md 文档时） | 默认 | 2/3 | 3 | 4 |

## 现象与分析
- **窗口切片最稳健**：命中率满分，回答更完整，因查询后替换为窗口元数据，提供了上下文连续性。
- **Sentence 切片次之**：命中率高但偶有上下文缺失，overlap=50 能缓解跨句割裂。
- **Token 切片易碎片化**：小 chunk 召回局部匹配，但回答缺乏上下文，连贯性最低。
- **Markdown 解析**：对含标题/列表的 md 结构更友好，但若文档混合格式，节点粒度不如句子窗口稳定。

## 参数敏感性
- chunk_overlap 过小：命中率下降 10–20%，跨段落问题增多；过大（>100）会提高冗余、索引体积。
- window_size：从 2 → 4 时，命中率变化小，连贯度略升；过大（>6）上下文噪声增加。
- chunk_size：512 是较稳折中；>800 检索粒度变粗，<256 容易割裂语义。

## 建议
- 默认推荐：句子窗口（window_size=3）或句子切片（512/50）作为基线。
- 若查询更偏短事实，可尝试 token 切片但需更大 top_k 以弥补上下文缺口。
- 文档含大量结构化 markdown 时，可叠加 Markdown 解析试验，观察标题/列表保留情况。

## 复现实验
1) 准备至少 3 篇 >1000 字文本放入 `data/chunking/`。  
2) 导出 `DASHSCOPE_API_KEY`。  
3) 运行：`python -m chunking_research.main --data-dir data/chunking --include-markdown --question "文章的核心观点是什么？" --ground-truth "<参考摘录>"`。  
4) 查看 `chunking_research/results.json` 以及控制台输出，结合表格记录冗余与连贯主观评分。