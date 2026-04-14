# AI CAD Agent 产品需求文档（技术版）

## 1. 文档信息
- 产品名称：AI CAD Agent
- 文档版本：v1.0（当前实现基线）
- 文档定位：技术架构与实现规范文档
- 读者：研发、架构师、DevOps、测试

---

## 2. 需求抽象与技术目标

### 2.1 需求抽象
输入为自然语言需求，输出为可执行 CAD 几何文件，并提供可视化反馈与失败自愈。

### 2.2 技术目标
- 构建可扩展的多 Agent 状态机；
- 支持多模型 Provider 与统一调用工厂；
- 通过自愈机制提高执行成功率；
- 支持云端部署与安全配置；
- 全链路具备可观测性（状态、历史、错误）。

---

## 3. 总体架构

### 3.1 模块划分
- `app.py`：Streamlit 交互层
- `agent_graph.py`：LangGraph 编排层
- `config.py`：LLM 配置与工厂
- `cad_tools.py`：CadQuery 执行与导出
- `rag_context.py`：轻量 RAG 检索
- `docs/rag/*.md`：知识片段

### 3.2 逻辑分层
1. **UI 层**：采集输入、展示状态与结果
2. **编排层**：控制节点执行与条件路由
3. **智能层**：Parser/Planner/Coder LLM 推理
4. **执行层**：CAD 代码执行、文件导出
5. **知识层**：标准约束与错误修复知识注入

---

## 4. 状态模型设计

核心状态对象：`AgentState`（TypedDict，partial 更新）

关键字段：
- 输入域：`input`, `chat_history`
- 解析域：`metadata`, `status`
- 规划域：`plan`
- 代码域：`code`
- 执行域：`errors`
- 控制域：`iterations`, `history`

设计原则：
- 节点幂等：同样输入可重复执行
- 状态可追踪：`history` 记录关键节点事件
- 错误可传递：`errors` 作为修复输入进入后续节点

---

## 5. 编排与路由（LangGraph）

### 5.1 主链路
`parser -> planner -> coder -> executor`

### 5.2 条件路由
- `decide_after_parser`
  - `need_clarification`：中断或等待补充
  - `parsed`：继续至 planner
- `decide_after_planner`
  - 有 `plan` -> coder
  - 无 `plan` 且未超限 -> retry planner
- `decide_after_coder`
  - 有 `code` 且无错 -> executor
  - 出错且未超限 -> retry coder
- `decide_to_retry`（executor后）
  - 错误且 `iterations < 3` -> retry coder
  - 否则 fail/complete

---

## 6. 节点设计细节

## 6.1 Parser Node
- 输入：`input + chat_history`
- 输出：结构化 metadata（JSON schema）
- 容错：
  - JSON 提取器（兼容 markdown code block 包裹）
  - LLM 失败回退规则解析
  - 人性化追问兜底生成
- 放行策略：
  - 计算 `blocking_missing_params`
  - 非关键参数（如 material/tolerance）可不阻塞

## 6.2 Planner Node
- 输入：原始需求 + metadata + RAG 标准
- 输出：步骤化 plan
- 机制：
  - create_llm 统一调用
  - 内部重试 + backoff + 超时预算

## 6.3 Coder Node
- 输入：需求、plan、RAG、上轮错误
- 输出：CadQuery 代码
- 机制：
  - 代码提取与静态校验
  - 修复 prompt 二次生成
  - 参数强制注入：
    - `extracted_params + suggested_defaults`
    - 渲染为代码顶部变量赋值块

## 6.4 Executor Node
- 执行 `code`，捕获 traceback
- 成功导出：
  - `part.step`
  - `part.stl`
- 失败时写回 `errors` 触发自愈循环

---

## 7. 错误治理与自愈机制

### 7.1 错误映射（Error -> Fix Strategy）
已覆盖：
- `StringSelector` 禁用替代
- `arcTo` 替代策略
- 选择器空结果
- solid stack 为空
- `BRep_API: command not done`
- `BRepAdaptor_Curve::No geometry`
- `No pending wires present`
- `NameError: xxx is not defined`

### 7.2 重试策略
- 节点内重试（planner/coder/parser）
- 图级重试（executor -> coder）
- 最大迭代保护（避免死循环）

### 7.3 降级策略
- LLM 不可用时 parser 可规则回退；
- mock fallback 由配置控制（调试/生产可切换）。

---

## 8. LLM 与配置架构

### 8.1 Provider 抽象
`config.py` 使用 `create_llm()` 统一管理：
- Gemini（`langchain-google-genai`）
- OpenAI/DeepSeek/Qwen（OpenAI-compatible）

### 8.2 环境变量设计
- `LLM_PROVIDER`, `LLM_MODEL`
- `USE_REAL_LLM`, `ALLOW_MOCK_FALLBACK`
- `*_API_KEY`, `*_BASE_URL`

### 8.3 启动校验
- `validate_llm_startup()` 检查 key 与占位符
- `.env` 通过 dotenv 加载（override=True）

---

## 9. 轻量 RAG 设计

### 9.1 数据源
- `docs/rag/cadquery_compat.md`
- `docs/rag/modeling_standards.md`

### 9.2 检索机制
- markdown chunk + 关键词评分（in-memory）
- 结果注入 coder prompt 作为强约束

### 9.3 价值
- 缓解模型“幻觉 API”问题
- 增强本地 CadQuery 版本兼容性

---

## 10. 前端架构（Streamlit）

### 10.1 会话状态
- `messages`：聊天线程
- `agent_history`：链路轨迹
- `last_result`：右侧看板统一渲染数据

### 10.2 交互区域
- 左侧：对话线程 + 输入
- 右侧：metadata + 状态机进度 + 错误 + 下载

### 10.3 3D 预览技术方案
- 后端：STL -> glTF
- 前端：three.js 渲染
- 网络容错：多 CDN 源 fallback + 页面内错误提示

---

## 11. 依赖与部署方案

### 11.1 Python 依赖
- `requirements.txt`（streamlit/langgraph/langchain/cadquery/pyvista/vtk 等）

### 11.2 系统依赖
- `packages.txt`：`libgl1`

### 11.3 云部署（Streamlit Cloud）
- repo: `Lyueyulia/cad-agent-demo`
- main file: `app.py`
- 必须在部署高级设置选择 Python 3.11
- Secrets 在 Cloud 控制台配置

---

## 12. 安全与合规

- `.env` 不入库（gitignore）
- `.env.example` 仅保留占位符
- 云端 Secrets 托管 API Key
- Deploy key（GitHub）用于仓库拉取，建议保留 read-only

---

## 13. 测试策略

### 13.1 当前测试
- 单测：`tests/test_retry_loop.py`
  - 错误回注验证
  - retry gate 达上限行为验证

### 13.2 建议补齐
- Parser schema 回归测试
- Coder 代码静态校验集
- Executor 文件产物断言（step/stl存在且非空）
- E2E smoke test（典型 prompt 集）

---

## 14. 性能与可观测性

### 14.1 性能关键点
- LLM 调用时延
- 自愈循环次数
- 3D 文件大小对前端渲染影响

### 14.2 可观测性建议
- 记录节点耗时与错误类型统计
- 打点成功率（first-pass / retry-pass）
- 建立失败样本库驱动 prompt 与策略迭代

---

## 15. 技术债与优化清单

### 高优先级
1. 预设零件模板化（连杆、法兰）减少几何漂移  
2. 默认值应用前确认机制  
3. 3D 前端依赖本地化（彻底去 CDN 依赖）  

### 中优先级
1. 结构化 plan schema 与约束验证  
2. RAG 升级到向量检索  
3. 更细粒度错误策略与自动诊断

### 低优先级
1. 多零件装配链路  
2. 团队协作与权限体系  
3. 企业审计日志

---

## 16. 里程碑建议

- M1（已完成）：单零件对话建模闭环 + 云部署
- M2（下一阶段）：模板化与稳定性提升（目标成功率 > 85%）
- M3（后续）：多零件与标准工程化能力

---

## 17. 附录

### 17.1 关键文件
- `app.py`
- `agent_graph.py`
- `config.py`
- `cad_tools.py`
- `rag_context.py`
- `requirements.txt`
- `packages.txt`

### 17.2 典型输入示例
- 连杆：`做一个连杆，中心距150mm，大头50mm，小头20mm`
- 法兰：`做一个带8孔的法兰盘，外径200，内径50，厚20，孔径12，PCD150`

