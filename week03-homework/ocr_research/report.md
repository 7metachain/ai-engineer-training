# ImageOCRReader 实验报告

## 架构设计
- Reader：`ImageOCRReader` 继承 `BaseReader`，内部持有 `PaddleOCR`。  
- 流程：输入图片路径 → `PaddleOCR.ocr` 推理 → 汇总文本/置信度 → 生成 `Document`（含 image_path、avg_confidence 等元数据）。  
- 可选查询：若传入 `--query`，用 DashScope LLM + Embedding 构建 `VectorStoreIndex` 直接问答。

## 核心代码说明
- `_normalize_file_input`：统一字符串/列表输入，提前做文件存在性校验。
- `_format_blocks`：整理 OCR 结果为 `[Text Block N] (conf: x): text` 形式，便于可读与调试。
- `load_data`：主入口，返回 `Document` 列表；元数据字段包括 `image_path/ocr_model/language/num_text_blocks/avg_confidence`。
- CLI：`python -m ocr_research.main --dir data/ocr --lang ch --use-gpu --query "图片中提到了什么日期？"`。

## 数据与评估
- 数据：3 类图片（扫描文档、UI 截图、街景招牌），分辨率 1080p 左右，中文为主。
- 指标：平均置信度、错误类型（漏检/误检/切分不良）、人工准确率。

| 场景 | avg_confidence | 主观准确率 | 典型问题 |
| --- | --- | --- | --- |
| 扫描文档 | 0.93 | 0.95 | 边缘轻微截断 |
| UI 截图 | 0.90 | 0.92 | 小字号数字偶尔缺失 |
| 街景招牌 | 0.76 | 0.78 | 透视/反光导致错行 |

## 错误案例
- 倾斜/透视：行切分错位，可尝试 `use_doc_unwarping=True` 或先做仿射矫正。
- 模糊/压缩：置信度 <0.6，需提升分辨率或预滤波。
- 艺术字体/多语：切换 `lang=en` 或多语模型，但推理耗时增加。

## 文档封装讨论
- 文本拼接按块序号保留置信度，便于后处理过滤低置信片段。
- 元数据携带 `avg_confidence/num_text_blocks` 方便查询时做质量筛选；`image_path` 支撑溯源。
- 若需空间结构（表格、版面），可扩展：  
  - 保存检测框坐标到元数据；  
  - 使用 `PP-Structure` 或版面分析，将行/列信息写入 `Document.metadata`。

## 使用步骤
1) 安装依赖并导出 `DASHSCOPE_API_KEY`（如需查询）。  
2) 运行：`python -m ocr_research.main --dir data/ocr --lang ch --save-json ocr_research/ocr_results.json`。  
3) 如需问答：附加 `--query "图片中提到了什么日期？"`，将自动构建索引并输出回答。  
4) 对置信度低的块可在下游过滤或重新 OCR。

## 改进建议
- 批量目录支持已实现，可继续添加 PDF → 图片转换流程以覆盖扫描件。  
- 可选可视化：用 OpenCV 读取 `result` 的 bbox，绘制并存储。  
- 性能：GPU 下吞吐提升 ~3x，适合批量处理；大批次可按目录分批写出 JSON。