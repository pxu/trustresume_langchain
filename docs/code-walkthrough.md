# TrustResume 代码导读（学习笔记）

> 面向第一次读这份代码的人：帮助你快速建立整体心智模型，理解每个模块**为什么这样写**，
> 以及跟着数据"走一遍"一次完整的简历生成过程。
>
> 这是原始项目 [`trustresume`](https://github.com/pxu/trustresume)（pydantic-ai + Qdrant）
> 在 **LangChain + LangGraph + ChromaDB** 上的重写。如果你想看更深的设计论证（需求、
> 更多 ADR），去读原始仓库的 `architecture/`；本仓库的 `architecture/high-level-design.md`
> 和 `architecture/decisions/` 讲的是"哪里一样、哪里变了、为什么变"。

---

## 1. 这个项目在做什么

一句话：**基于证据的、对 ATS（简历筛选系统）友好的简历自动生成**，核心卖点是
**可信度验证（Trust Verification）**——生成的每一条简历内容都必须能在候选人自己上传的
材料里找到依据，杜绝"AI 编造经历"。

它用到三样东西：

1. **RAG（检索增强生成）**：候选人上传的简历、项目报告、STAR 故事、证书被切块、
   向量化，存进向量库；生成时按岗位描述去语义检索最相关的证据。
2. **多智能体（multi-agent）**：把"读岗位 → 检索证据 → 写简历 → 查真伪 → 评 ATS"
   拆成一串各司其职的 agent，由一个编排器（orchestrator）串起来。
3. **质量闭环（quality loop）**：写完先自动打两个分（Trust 分、ATS 分），不达标就
   带着"具体哪里要改"的反馈重写，最多重写 3 次。

> 核心理念（贯穿全代码）：**一个为了刷 ATS 关键词而撒的谎，比一个诚实的能力空缺更糟。**
> 所以 Trust 分永远优先于 ATS 分（见 `feedback.py`）。

---

## 2. 三十秒看懂目录结构

代码在 `src/trustresume/`，按"数据从下往上流"的层次组织：

```
models/            纯 Pydantic 数据契约，不依赖任何框架 —— 所有其它模块都 import 它
  ├─ 各层之间传递的就是这些对象（JobDescription / EvidenceSet / ResumeDraft / TrustReport …）

storage/           SQLite 仓库（用户、文档、chunk、简历、评估、profile 缓存）—— 从原项目逐字照搬
retrieval/         FastEmbed 向量化 + Chroma 向量库，永远按 user_id 隔离
ingestion/         解析 → 清洗 → 切块 → 同时写两个库（写路径，独立于生成流程）

agents/            六个"纯输入→输出"的 agent（下面详解）
orchestration/     Orchestrator（LangGraph 状态图）+ profile 缓存服务 + 反馈生成
trust_verification/  Trust 校验的 prompt / 格式化 / 报告组装（纯函数，不碰 LLM）
evaluation/        ATS 关键词覆盖打分（纯函数）

api/               TrustResumeApp 门面 + LLM provider 工厂 + 离线 test provider + FastAPI
poc/               独立的 LLM 连通性冒烟测试，不属于主应用
```

一个很有用的记忆法：**箭头方向 = 依赖方向**。`models/` 在最底层谁都不依赖，`api/`
在最顶层把所有东西"接线"成一个可运行的应用。想理解任何一个对象，先去 `models/`
看它的字段定义。

---

## 3. 六个 agent：各是什么、为什么这么分

所有 agent 都遵守同一条铁律（`agents/base.py` 的注释）：
**每个 agent 是一个纯粹的"输入 → 输出"步骤**——它需要的一切都从参数传入，返回一个
`models/` 里的对象，**绝不调用另一个 agent、绝不读编排器的状态**。谁来串它们？编排器。

| Agent | 类型 | 输入 → 输出 | 关键点 |
|---|---|---|---|
| `JobDescriptionAgent` | LLM | 岗位原文 → `JobDescription` | 抽取标题/技能/关键词/职责。返回时**强行把 `raw_text` 塞回原文**，防止模型改写后丢了原始信息。 |
| `CandidateProfileAgent` | LLM | 候选人材料文本 → `CandidateProfile` | 抽取候选人技能/证书。**与岗位无关**，所以被缓存、不是每次都跑（见 `CandidateProfileService`）。 |
| `EvidenceRetrievalAgent` | **非 LLM** | user_id + job → `EvidenceSet` | 故意**不用 LLM**——检索是确定性的语义搜索，套个 LLM 只会增加不确定性和成本。 |
| `ResumeWriterAgent` | LLM | job + evidence (+ feedback) → `ResumeDraft` | 唯一的"生成"步骤。**只能用给定的证据**，绝不自己编。写完**从不自评**。返回时强行盖上本轮 `iteration`。 |
| `TrustHarnessAgent` | LLM | draft + evidence → `TrustReport` | 项目的**核心研究贡献**。逐条抽取简历里的声明，对照证据判定 SUPPORTED / PARTIALLY / UNSUPPORTED。 |
| `ATSEvaluationAgent` | **非 LLM** | draft + job → `ATSReport` | 纯代码算关键词覆盖率，确定、便宜、可复现。 |

### 三个反复出现、值得记住的设计决策

1. **"生成"和"验证"是分开的两次 LLM 调用（ADR-0004）。**
   写简历的 agent 永远不被信任去自评准确性——Trust Harness 是独立一步、独立一次模型调用。
   让写手自己说"我写得很实在"是没有意义的。

2. **分数由代码算，不由 LLM 说。**
   Trust Harness 里，LLM 的职责只是**分类**每条声明（支持/部分/不支持）；那个 0–100 的
   Trust 分是 `TrustReport.compute_score` 这段确定性代码算的（见 §5）。ATS 分同理。
   好处：可复现、可解释、不会因为模型"心情好"就给高分。

3. **LLM agent 用 `chat_model.with_structured_output(Schema)`。**
   这是 LangChain 的写法：把一个 Pydantic 模型绑给聊天模型，模型就会以工具调用的形式
   返回一个符合该 schema 的结构化对象。原项目用的是 pydantic-ai 的 `Agent`——这是本次
   移植最主要的技术替换点。注意 `base.py` 说：**每个 agent 构造时必须注入 model**，
   因为 LangChain 没有 pydantic-ai 那种"构造后再换模型"的钩子，测试只能在构造时直接塞假模型。

---

## 4. 编排器：LangGraph 状态图，与那个"反直觉但很重要"的计数

`orchestration/orchestrator.py` 是整个控制流的唯一拥有者。原项目手写了一个 `while` 循环
（为了写论文时代码好读），本项目按原 ADR-0003 早就预言的方向，把它换成了 **LangGraph
`StateGraph`**（ADR-0003）。**但公开契约完全没变**：构造函数还是那六个 agent，入口还是
`async def run(*, user_id, job_posting, gate=None) -> WorkflowState`，所以上层
(`api/app_service.py`) 一行都不用改。

### 图长这样

```
START → analyze_job → load_candidate_profile → retrieve_evidence
      → write_resume → score_trust → score_ats → [条件边 _route]
                                                     ├─ "end"     → END
                                                     └─ "rewrite" → prepare_rewrite → 回到 write_resume
```

- 前三步（读岗位、取 profile、检索证据）**每次生成只跑一次**——岗位和候选人的证据在重写
  之间不会变，变的只有草稿。
- `write_resume → score_trust → score_ats` 这三步构成可重复的**质量闭环**。

### 状态是怎么在图里流动的

图内部用一个私有的 `_GraphState`（TypedDict），跑完再转回公开的 `WorkflowState`。
其中三个字段被标成 `Annotated[list[X], operator.add]`：

```python
drafts: Annotated[list[ResumeDraft], operator.add]
trust_reports: Annotated[list[TrustReport], operator.add]
ats_reports: Annotated[list[ATSReport], operator.add]
```

`operator.add` 是 LangGraph 的"追加式 reducer"：每个节点返回 `{"drafts": [draft]}`，
LangGraph 会**把它拼到已有列表后面**而不是覆盖。于是每一轮的草稿和分数都被完整留档
（供论文/UI 回看）。其它字段是默认的"后写覆盖"。

### ⚠️ 最容易踩坑的地方：`max_iterations=3` 产生的是 **4** 份草稿

这是全代码里最反直觉、也最"承重"的细节，一定要记住：

```python
def _route(self, state):
    passed = gate.passes(...)
    is_exhausted = state["iteration"] >= gate.max_iterations   # 用的是"增量前"的值
    return "end" if (passed or is_exhausted) else "rewrite"
```

`_route` 在 `score_ats` 之后、`prepare_rewrite`（负责 `iteration += 1`）**之前**被调用。
所以判断用的 `iteration` 还是刚打完分那份草稿的编号。默认 `max_iterations=3` 时：

- iteration 0（初稿）→ 不达标 → rewrite（iteration 变 1）
- iteration 1 → 不达标 → rewrite（变 2）
- iteration 2 → 不达标 → rewrite（变 3）
- iteration 3 → 判断 `3 >= 3` 为真 → end

一共 **4 份草稿（0、1、2、3）**，不是 3 份。这个语义有专门的测试守着：
`tests/unit/test_orchestrator.py::test_orchestrator_failsToCap_stopsAndExportsRealScores`。
改任何跟循环有关的东西前，先跑这个测试。

> 顺带一提：`run()` 里把 LangGraph 的递归上限从默认 25 提到了
> `6 + 4 * max_iterations + 10`——否则调用方传一个更大的 `max_iterations` 时会先撞上
> 递归上限报错，而不是正常"到顶停下"。

---

## 5. Trust 分是怎么算出来的（研究核心）

看 `models/trust.py`，两个模型：

- `VerifiedClaim`：一条声明 + 它的判定。有个属性 `is_hallucination`——
  **只有"事实类"声明（技能/经历/证书/成就）在 UNSUPPORTED 时才算幻觉**；
  一句 `OTHER` 类的文风性表述即使无依据也不算幻觉。
- `TrustReport.compute_score`：透明的默认评分规则——

  ```
  SUPPORTED 记 1 分，PARTIALLY_SUPPORTED 记 0.5，UNSUPPORTED 记 0
  求平均 × 100
  声明列表为空 → 0 分（没有任何可核实内容 = 不可信）
  ```

数据流分工很清晰：
- **LLM**（`TrustHarnessAgent` 里的私有 schema `_ClaimExtraction`）只负责把声明分类。
- **`trust_verification/verifier.py`**（纯函数）负责 prompt、把 draft/evidence 格式化给模型、
  以及把分类结果组装成报告。
- **`models/trust.py`** 负责那个数据级的评分规则。

三层各管一段，谁都能被单独测试。

> 💡 **离线模式的一个"特性不是 bug"**：用 `TRUSTRESUME_LLM_PROVIDER=test` 时，
> 假模型合成出来的 `_ClaimExtraction.claims` 永远是空的，所以 Trust 分恒为 0，
> 于是离线跑总是"撑到迭代上限"而非"通过"。这是确定性的预期行为。

---

## 6. 存储与检索：两个库，靠 user_id + chunk_id 缝合（ADR-0001），外加混合检索（ADR-0010）

系统存两类数据，用两个库：

| 库 | 存什么 | 说明 |
|---|---|---|
| **SQLite**（`storage/`）| 结构化记录：用户、文档元数据（含 `content_hash`,见 §7.2）、chunk 全文、生成的简历、Trust/ATS 分、profile 缓存 | 从原项目**逐字照搬**——它不属于本次要换的技术栈。 |
| **Chroma**（`retrieval/`）| 切块后、向量化的文档内容，用于语义检索 | 本次从 Qdrant 换来的。 |

两库靠 `user_id` + `chunk_id` 关联。几个要点：

- **每次检索都带 `user_id` 元数据过滤器**（`ChromaVectorStore.search` 里的 `filter=`），
  这是**用户隔离的唯一防线**（ADR-0001）。绝不能去掉它，也不能改成客户端过滤。代码里
  甚至没有"不带过滤"的检索路径。
- **`chunk_id` 直接当 Chroma 的文档 id**：Chroma 接受任意字符串 id，所以省掉了 Qdrant
  时代那层 uuid5 转换。重复写同一个 chunk 会原地覆盖（幂等），不会产生重复。
- **距离 vs 相似度**：Chroma 默认返回的是**距离**（越小越相似），而应用其余部分期望
  `EvidenceChunk.score` 是"越大越相关"。所以 `search` 里做了 `score = 1 - distance`
  转换（collection 配了 `hnsw:space: cosine`）。
- **嵌入模型懒加载**：`FastEmbedEmbeddings` 在第一次真正 embed 时才下载/加载底层模型，
  只是构造一个向量库并不会触发下载——对测试和启动速度很友好。

### 6.1 混合检索：向量 + 关键词，用 RRF 融合（ADR-0010）

**动机**：`FastEmbedEmbeddings` 用的是一个通用的小模型（`BAAI/bge-small-en-v1.5`，384 维），
不是针对简历/岗位描述这个领域调过的——恰恰在"精确技术名词"上最弱：候选人写"AWS Lambda"，
向量检索可能因为语义相近，把"AWS EC2"排得跟它差不多高，但岗位描述点名要 Lambda 的话，
应该精确命中写了 Lambda 的那条,而不是被语义相近的别的服务稀释掉。纯关键词检索又有反过来的
问题（找不到没有共同词汇的同义改写）。所以业界标准做法是**两个都用,融合排名**。

好消息是 SQLite 里本来就存了每个 chunk 的全文（`chunks.text`），加关键词检索不需要引入
第三个存储,只需要给已有数据加个索引：

- **`chunks_fts`**（`storage/schema.py`）——一张 FTS5 **external-content** 虚拟表
  （`content='chunks', content_rowid='rowid'`）：索引 `chunks.text` 但不复制它,靠
  `chunks` 表上的 `AFTER INSERT`/`AFTER DELETE` 触发器自动同步。`ChunkRepository.add`/
  `delete_for_document` 完全不用知道这个索引存在。
- **`ChunkRepository.search_keywords`**——跑 FTS5 的 `MATCH` 查询,按 SQLite 内置的
  `bm25()` 排名。⚠️ **原始查询字符串不能直接传给 `MATCH`**——FTS5 的查询语法对
  `-`/`"`/`*`/`:`/`/`/括号有特殊含义,而岗位描述几乎肯定会带上其中几个（比如
  "AI/ML"），直接传会报语法错误。所以 `_to_fts5_query` 先把查询**分词成纯字母数字的
  token,每个 token 用双引号包成短语,再用 OR 连接**——绕开了所有特殊字符问题。
- **`HybridRetriever`**（`retrieval/hybrid.py`）——同时查 `ChromaVectorStore.search`
  （向量）和 `ChunkRepository.search_keywords`（关键词,BM25），再用
  **RRF（Reciprocal Rank Fusion）** 融合：每个 chunk 的分数是 `1/(k+rank)`（`k=60`，
  RRF 论文的默认值），在它出现的每个列表里都累加一次。

  **为什么融合排名、不融合原始分数**：Chroma 的余弦相似度在 `[0,1]`（越大越好），
  SQLite 的 `bm25()` 是无界、符号反转的对数几率分数（越小越好）——这两个数字根本不在同
  一个量纲上,直接加权平均等于拿苹果比橘子。RRF 只看**排名位置**，两边都能给出可比较的
  排名，这是唯一能公平融合的东西。
- `HybridRetriever.search(user_id, query, limit) -> EvidenceSet` 的签名跟
  `ChromaVectorStore.search` 完全一样,所以 `EvidenceRetrievalAgent`（现在类型标注是一个
  `Protocol`，不再是具体的 `ChromaVectorStore`）和 `TrustResumeApp.search_evidence`
  **一行调用代码都不用改**，只是构造时换成传 `HybridRetriever`。

一个直观例子（`tests/unit/test_hybrid_retrieval.py` 里就是这么测的）：往库里塞 6 条
chunk，其中只有一条真的含有"Lambda"这个词，其余 5 条是不相关的填充内容。用假的
hash 嵌入模型（没有真实语义信号）单独做向量检索，那条含 Lambda 的 chunk 很可能排不到
前面；但混合检索里，关键词那一路会精确命中它，RRF 融合后它稳定进入结果——这就是混合
检索存在的意义。

---

## 7. 摄取（ingestion）：一条独立的写路径

`ingestion/service.py` 的 `IngestionService` **不是生成流程里的一步**。它由文档
上传/删除单独触发，负责把一份文档落进两个库并**保持一致**：

```
parse（解析文件，unstructured）→ clean_text（清洗，手写）→ 判重（content_hash，见 §7.2）
→ chunk_text（切块，LangChain）→ 先写 SQLite 的 chunk 元数据行 → 再写 Chroma 向量
```

**顺序很重要**：先写 SQLite，再写 Chroma。如果向量 upsert 失败，就**回滚刚写的 SQLite
chunk 行**，避免两库漂移（`try/except` 里那段）。

它还负责**让缓存失效**：任何一次摄取或删除后，都会调用
`candidate_profiles.mark_stale(user_id)`，这样下次生成时 `CandidateProfileService`
会重新计算 profile，而不是拿旧的技能/证书糊弄。

**解析（parsing）和切块（chunking）是两个独立的步骤，各用各的库**：
- **解析**（`parser.py`）：把 `.docx`/`.pdf` 的原始字节变成纯文本——见 §7.1，现在统一走
  `unstructured` 库。
- **切块**（`chunker.py`）：把纯文本切成一个个 chunk——见 §7.3，用 LangChain 的
  `RecursiveCharacterTextSplitter`，按"段落 → 空格 → 字符"逐级递归切，尽量保住一条
  bullet 或一个完整 STAR 故事——那正是 Trust Harness 之后要检索的语义单元。相邻块之间带
  overlap，让跨边界的句子在邻块里仍完整出现。

### 7.1 解析 `.docx`/`.pdf`：统一走 `unstructured`

`parser.py` 现在**不再各写一个格式一个函数**（原来 `.docx` 用 `python-docx`、`.pdf` 用
`pypdf`，两套代码）。改成一个共享入口：

```python
from unstructured.partition.auto import partition

def _parse_rich_bytes(filename: str, data: bytes) -> str:
    elements = partition(file=io.BytesIO(data), metadata_filename=filename, strategy="fast")
    return "\n".join(str(element) for element in elements)
```

`unstructured` 是 RAG 场景解析文档的事实标准库,`partition()` 靠文件名后缀自动识别格式,
`.docx`/`.pdf`/以后想加的格式都走同一段代码——真正做到了"加新格式只加一行 `_RICH_SUFFIXES`
后缀,不用写新函数"。

⚠️ **`strategy="fast"` 是故意选的,不是默认值**：`unstructured` 默认策略是 `"hi_res"`，
会加载一个基于 torch 的版面分析模型做 OCR/表格识别——实测解析一份简历 PDF 要 **49 秒**。
这个项目只需要文本内容去切块，不需要版面感知的阅读器，`"fast"`（纯文本抽取，不跑模型）把
同一份 PDF 的解析时间降到 **2.8 秒**，内容质量一致。这是"选对策略参数"比"用了什么库"更
重要的一个例子。

> 💡 **为什么不是 `langchain-community` 的 `PyPDFLoader`/`Docx2txtLoader`？**
> 那两个本质上只是把 `pypdf`/`docx2txt` 包了一层，而且 `langchain-community` 这个包
> 本身已经被标记为"sunset,不再积极维护"——用它不是真正的 best practice，只是换了个
> 已经过时的包装皮。`unstructured` 才是真正解决"多格式统一解析"这个问题的标准库。

### 7.2 摄取判重：同一份文档传两次，不应该在两个库里各存两份

`IngestionService.ingest_text` 在写任何东西之前,先算**清洗后文本**的 SHA-256 哈希
（`_content_hash`），去 SQLite 查这个用户名下有没有已经存过同样哈希的文档
（`DocumentRepository.find_by_content_hash`）。有就直接返回那份文档的 id,**什么都不写**
——不切块、不生成向量、不碰任何一个库。没有这层判重,上传同一份简历两次会让 SQLite 和
Chroma 里的 chunk 数量都翻倍,以后每次检索都会重复算这份内容两遍。

几个设计细节：

- **哈希算的是"清洗后"的文本,不是原始字节**——`clean_text` 已经把空白/换行差异抹平了，
  所以同一份简历从 Word 和 Google Docs 分别导出（字节级不同,内容其实一样）也能被识别成
  重复。
- **判重的粒度是 `(user_id, content_hash)`**——两个不同用户上传内容完全相同的简历模板
  不算重复（ADR-0001 的用户隔离边界,判重不能打破它）。
- **数据库层也有一道防线**：`documents` 表上有 `UNIQUE(user_id, content_hash)` 索引，
  不只是应用层的"先查再写"检查。这是为了兜住一个真实存在的竞态窗口：如果两个请求同时
  摄取同一份内容,应用层的检查和插入之间不是一个原子事务,可能都查到"不存在"然后都去插入
  ——数据库唯一索引会让第二个插入失败,`ingest_text` 捕获这个 `IntegrityError` 后重新查一次,
  返回第一个请求真正写入的那个 `document_id`,而不是把这个竞态原样抛给调用方。

### 7.3 切块：为什么用 LangChain 而不是手写（如何 chunking）

先看现状：RAG 这条链路**整条都在用 LangChain 的抽象**——
`FastEmbedEmbeddings` 实现 `langchain_core.embeddings.Embeddings` 接口、
`ChromaVectorStore` 包 `langchain_chroma.Chroma`、切块用
`langchain_text_splitters.RecursiveCharacterTextSplitter`。**这是学 LangChain 的项目该有的样子。**

`chunk_text` 现在的实现就是薄薄一层包装（公开签名 `chunk_text(text, *, max_chars, overlap)`
没变，所以 `IngestionService` 等上层一行都不用改）：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_chars,
    chunk_overlap=overlap,
    separators=["\n", " ", ""],   # 见下方注意点
    keep_separator=False,
)
return splitter.split_text(text)
```

它的机制：按 `separators` 从"大"到"小"**递归**尝试——先按段落切，段落还太大就退到按空格、
再不行按字符——尽量让每块 ≤ `chunk_size`，相邻块之间保留 `chunk_overlap` 的重叠。

**为什么这是 best practice（而不是自己写循环切）：**
文本切块是个"看着简单、其实全是边界情况"的问题（overlap 怎么带、超长段落怎么兜底、
分隔符优先级……）。LangChain 已经把它做成经过大量项目验证的标准件——**用框架的，就不用自己
维护和测试这些边界，也更符合生态习惯**。从零写一个文本切块器等于重造轮子，那才是坏味道。
（本项目当初为"忠实移植原始 pydantic-ai 版本"手写过一版；作为**学习项目**，换成框架标准件
更有价值，所以换了。）

**两个落地时容易踩、这次已经处理好的点：**

1. **分隔符对齐了 `clean_text`。** `clean_text` 已把空行 `\n\s*\n+` 折叠成**单个 `\n`**，
   清洗后的文本里根本没有 `\n\n`。所以 `separators` 用的是 `["\n", " ", ""]` 而**不是**默认的
   `["\n\n", "\n", " ", ""]`——否则默认的 `\n\n` 永远匹配不上，白白多一层。`clean_text` 保持
   手写是对的：它是**空白归一化**（领域预处理），不是切块，LangChain 也不做这件事。

2. **overlap 语义变了（这是换框架带来的真实行为变化）。**
   - 之前手写版：只有硬切超长段落时才加 overlap；正常累积多个段落成一块时，块间**无** overlap。
   - `RecursiveCharacterTextSplitter`：**所有相邻块之间都带 overlap**——这其实是 RAG 里更标准、
     更推荐的行为（跨块上下文不丢）。

   换完 `pytest`/`ruff`/`mypy` 全绿（现有断言恰好仍成立），但要意识到：切块结果变了 →
   理论上检索命中、Trust/ATS 分都可能变。在**学习项目**里这是可接受、甚至更好的默认；如果这是
   生产系统，配套应该补一个 ADR + 一轮"换前换后对比分数"的检索质量回归。

> 同样的判断后来也用在了 `parser.py`（解析文件）上：§7.1 已经把它换成了 `unstructured`。

---

## 8. Profile 缓存服务：为什么它是 "Service" 而不是 "Agent"

`orchestration/candidate_profile_service.py` 包着 `CandidateProfileAgent`，在它外面加了
**缓存判断 + 文档拼装**：

- `get_or_refresh`：缓存新鲜就直接返回；否则把该用户所有 chunk 拼成一大段文本，
  跑一次 agent，存缓存。
- 缓存判定**只看 `stale` 标志**（由 ingestion 设置），那个 `_hash` 只是记录下来给调试
  用的，不参与判定。

为什么叫 Service 不叫 Agent？因为它本身是**确定性的编排逻辑**（缓存/拼装），真正的
LLM 推理完全发生在它包着的那个 agent 里面。同理 `IngestionService` 也是 Service。
这就是本项目"Service vs Agent"的区分线。

---

## 9. API 层与"接线"

- **`TrustResumeApp`（`api/app_service.py`）** 是应用门面——所有真实逻辑都在这里被
  接成一个对象（两个库、ingestion、六个 agent、编排器）。FastAPI 只是它上面很薄的一层，
  这样不用起 HTTP 就能测全链路。注意 `generate()` 跑完会把最终草稿+分数**持久化**到 SQLite，
  哪怕是"撑到上限"的草稿也照存（导出真实分数，ADR-0005）。
- **`model_factory.py`** 决定用哪个 LLM。配置优先级：
  **环境变量 > `config/llm.local.json` > `config/llm.json` > 默认值**。
  支持 `bedrock`（默认）/ `openai` / `google` / `test`。每个 provider 的 SDK 都在各自
  分支里**懒 import**，所以只装你要用的 provider 就行。
- **`test_provider.py` 的 `AutoStructuredFakeChatModel`** 是本项目一个巧妙的补丁：
  LangChain 自带的假模型不支持 `with_structured_output`，而本项目每个 LLM agent 都靠它。
  这个假模型会**读取被绑定的任意 schema，合成一个最小合法实例**（空串/空列表/枚举第一个值/
  数字取 0），于是 `TRUSTRESUME_LLM_PROVIDER=test` 能**无凭证跑通整条生成流水线**。
  pydantic-ai 的 `TestModel` 原生有这能力，LangChain 没有，所以这里重造了一个。
- **`server.py`** 是 FastAPI。`create_app(facade)` 注入门面（测试用内存+假模型的门面），
  `build_served_app` 是给 uvicorn 的 `--factory`（延迟到启动才建真库/真模型，保证 import
  无副作用）。端点：`/api/documents`（JSON 增/查）、`/api/documents/upload`（multipart
  文件上传，服务端解析）、`/api/search`（独立的语义检索，见下）、`/api/generate`（生成）、
  `/api/ping`（真连 LLM 探活）、`/api/health`。

### 9.1 `/api/search`：把检索单独拎出来

`TrustResumeApp.search_evidence(user_id, query, limit)` 直接调用
`ChromaVectorStore.search`——和 `EvidenceRetrievalAgent` 内部用的是**同一个检索**，
只是不需要走完整的 `generate()` 流程（读岗位→检索→写→查真伪→评分）就能单独看"这个查询会
检索到什么"。这对**调试检索质量**特别有用：想知道"如果我搜 Kubernetes,会不会命中我上传的
那份简历",不用真的跑一次昂贵的生成流程,直接 `POST /api/search` 就行。`user_id` 隔离
（ADR-0001）照样生效。

### 9.2 结构化日志：`logging_config.py`

`JsonFormatter` 把每条日志渲染成一行 JSON（容器日志采集的标准格式）。`configure_logging()`
只在 `server.py` 的 `build_served_app`（真实进程入口）调一次；库代码永远只用
`logging.getLogger(__name__)`，不自己配置 handler——这样 import 任何模块都没有副作用。

编排器在每个节点转换、每次质量闭环的路由决策都打日志（`orchestration/orchestrator.py`）;
ingestion 在成功摄取/回滚时打日志。**一个真实踩过的坑**：`extra={"filename": ...}` 会跟
Python `logging` 内部保留的 `LogRecord` 属性（`filename`/`module`/`name`/`lineno` 等）
**撞名**，触发 `KeyError`——这个 bug 只有在**真的起一个服务器调用 `logger.info(extra=...)`**
时才会暴露,单元测试里手工构造 `LogRecord`（绕过了这个检查）是测不出来的。教训被记在
`test_logging_config.py` 的 `test_loggerInfo_withExtra_doesNotRaiseOnReservedLookingKeys`
里,以后加 `extra` 字段前先检查名字别撞车。

---

## 10. 跟着数据走一遍：一次 `POST /api/generate`

假设文档已经摄取好了。整个流程（对应 high-level-design 的时序）：

```
1. JobDescriptionAgent.run(posting)          → JobDescription   （一次）
2. CandidateProfileService.get_or_refresh()   → CandidateProfile （一次，通常命中缓存）
3. EvidenceRetrievalAgent.run(user_id, job)   → EvidenceSet      （一次，混合检索+按用户过滤，见§6.1）
   ┌── 质量闭环（≤ 4 轮：初稿 + 最多 3 次重写）───────────────────────┐
4. ResumeWriterAgent.run(job, evidence, feedback?) → ResumeDraft
5. TrustHarnessAgent.run(draft, evidence)     → TrustReport  （LLM 分类，代码打分）
6. ATSEvaluationAgent.run(draft, job)         → ATSReport    （确定性覆盖率）
   ── 若 Trust ≥ 90 且 ATS ≥ 85 → 通过，停
   ── 否则若 iteration == 3 → 到顶，停（照样导出）
   ── 否则 build_feedback(trust, ats) → 重写（回到第 4 步）
   └────────────────────────────────────────────────────────────────┘
7. 最终草稿 + 评估落库 SQLite
8. WorkflowState → GenerateResponse（真实分数 + 被标记的声明）
```

第 6 步的反馈（`orchestration/feedback.py`）是**确定性生成**的，不是又一次 LLM 调用：
它从 Trust/ATS 报告里拼出具体指令——"删掉这条没依据的 Kubernetes 声明"、"补上 AWS 这个
关键词"。而且**幻觉问题排在关键词缺失前面**，因为准确性优先于关键词覆盖。

---

## 10.5 前端：Streamlit + 部署（CI/Docker）

### Streamlit UI（`src/trustresume/ui/`）

**只是 REST 客户端**，不直接 import 任何 `trustresume.api`/`orchestration` 内部对象——
依赖方向永远是"UI → HTTP → 后端"，所以后端可以脱离 UI 独立跑（无头模式），`streamlit`
也就只是个可选的 `ui` extra，不污染核心依赖。

- **`api_client.py` 的 `TrustResumeClient`**——薄薄一层 `requests.Session` 封装，一个方法
  对应一个后端路由（`health`/`list_documents`/`upload_document`/`search`/`generate`）。
  单独拆出来是为了**不依赖 streamlit 就能测**（`tests/unit/test_ui_api_client.py` 直接 mock
  `requests.Session`）。
- **`streamlit_app.py`**——三个 tab,对应候选人跟系统交互的三件事：
  - 📁 **Documents**——上传证据文档（调 `/api/documents/upload`）+ 列出已上传的文档。
  - ✨ **Generate**——粘贴岗位描述，跑完整的 RAG + 多智能体流水线，展示 Trust/ATS 分数、
    草稿、被标记的幻觉声明、缺失关键词。
  - 🔍 **Search**——直接调 `/api/search`，单独看 RAG 检索结果（不用跑一次昂贵的生成）。

  运行：`TRUSTRESUME_API_URL=http://localhost:8000 streamlit run src/trustresume/ui/streamlit_app.py`
  （后端要先起）。

> ⚠️ **一个真实踩过的坑**：`streamlit run some_file.py` 是把这个文件当**脚本**执行，
> 没有包上下文——所以脚本里不能用相对导入（`from .api_client import ...`），会报
> `ImportError: attempted relative import with no known parent package`。这个 bug
> 只有**真的用 `streamlit run` 起一次、在浏览器里打开**才会暴露；写完代码后 `ruff`/`mypy`
> 都不会报错（它们不管一个模块是被 import 还是被当脚本跑）。修法：改成绝对导入
> `from trustresume.ui.api_client import TrustResumeClient`。**这就是为什么"写完代码"和
> "在真实环境里跑一遍"是两件不同的事**——尤其是 CLI 入口脚本这种，静态检查覆盖不到的场景。

### CI / Docker

这两块是本次会话新加的,原始项目没有：

- **`.github/workflows/ci.yml`**——push/PR 时跑 `ruff check` → `ruff format --check` →
  `mypy src` → `pytest`（Python 3.11 和 3.13 两个矩阵），依赖从 `uv.lock` 精确安装
  （`uv sync --locked`），保证"我本地能跑"和"CI 能跑"用的是完全一样的依赖版本。
- **`uv.lock`**——锁定每个依赖的精确版本（不只是 `pyproject.toml` 里的版本范围）。
  没有它，"能复现"就只是一句空话——今天装的和明天装的可能是不同小版本，出问题不好排查。
- **`Dockerfile`**——多阶段构建：`builder` 阶段用 `uv sync --locked` 装依赖，
  `runtime` 阶段是精简的最终镜像（跑 FastAPI），`ui` 阶段（`FROM runtime AS ui`）复用
  同一个 venv 只是换个启动命令（跑 Streamlit）——两个镜像共享依赖层，不用装两遍。
- **`docker-compose.yml`**——把 `api` + `ui` 两个服务连起来，默认
  `TRUSTRESUME_LLM_PROVIDER=test`,所以 `docker compose up --build` 不需要任何凭证就能跑
  起整个系统（含前端）。

---

## 11. 测试模型：整栈离线可跑（NFR-5）

不需要网络、不需要凭证就能跑完整个栈。替身对照表：

| 真实依赖 | 测试替身 |
|---|---|
| SQLite 文件 | `connect(":memory:")` |
| Chroma | `chromadb.EphemeralClient()`——⚠️ 每个测试/store 用**唯一的 collection 名**，否则 Chroma 会按 settings 哈希缓存底层存储，同名 collection 会在同一进程内"串味"（跨"全新"客户端泄漏状态）。`TrustResumeApp.__init__` 专门开了个 `chroma_collection_name` 参数给测试用（生产代码用默认值） |
| SQLite FTS5（关键词检索） | 不需要假的——`connect(":memory:")` 本身就带 FTS5，`ChunkRepository.search_keywords`/`HybridRetriever` 直接对着真实 FTS5 索引测（`tests/unit/test_storage.py`、`test_hybrid_retrieval.py`） |
| 嵌入模型 | `tests/fakes.py` 的 `FakeEmbeddings`（和原项目一样用 SHA-256 哈希造向量）；真实 `FastEmbedEmbeddings` 的"懒加载"逻辑用 mock 掉 `fastembed.TextEmbedding` 来测,它的真实输出交给一个 `live` 标记的测试（真模型，首次跑可能触发下载） |
| 文档解析（.docx/.pdf） | mock 掉 `unstructured.partition.auto.partition` 测拼接逻辑；真实解析交给两个 `live` 标记的测试，各对着一份真实样例文件跑（`test_parseDocx_realSampleFile`/`test_parsePdf_realSampleFile`） |
| LLM（单元测试）| `tests/fakes.py` 的 `scripted_tool_call(name, args)`——造一个只回一个脚本化工具调用的假模型；`name` 必须等于目标 Pydantic 模型的类名 |
| LLM（`provider="test"`，集成/live 测试）| `AutoStructuredFakeChatModel`（§9） |
| Streamlit 前端 | `streamlit.testing.v1.AppTest`（§10.5、`test_streamlit_app.py`）——真的跑一遍脚本（渲染、点按钮、填表单），只 mock 掉 `requests.Session`，不需要浏览器。⚠️ 测试之间要清 `st.cache_resource.clear()`（autouse fixture），否则后面的测试会拿到前一个测试 mock 过的客户端缓存 |

`tests/unit/` 一对一镜像 `src/trustresume/` 的包边界；`tests/integration/` 端到端跑
`TrustResumeApp` 和 FastAPI。`test_api_live.py` 会真起一个 uvicorn 子进程，标了 `live`，
默认不跑（`pytest -m live` 才跑）。

**覆盖率门槛**：`pytest` 的 `addopts` 里内置了 `--cov-fail-under=95`（实际跑出来大概
99%）——覆盖率不够，`pytest` 直接跑失败，不是"看报告发现的"。`src/trustresume/poc/*`
被排除在覆盖率统计外（`[tool.coverage.run]` 的 `omit`）：它是留给人手动跑的连通性冒烟
测试，真要测出意义得真连一个 LLM，跟 `live` 标记是同一套道理。

常用命令：

```bash
pytest                          # 离线单元 + 集成测试（含覆盖率门槛，95%）
pytest -m live                  # 额外跑真起服务器/真嵌入模型的 live 测试
pytest tests/unit/test_foo.py::test_bar   # 单个测试
ruff check .                    # lint
ruff format --check src tests   # 格式检查（CI 用的范围）
mypy src                        # 严格类型检查
```

---

## 12. 建议的阅读顺序

如果你想动手读源码，按这个顺序心智负担最小（和依赖方向一致，从下往上）：

1. `models/`——先看数据长什么样，尤其 `trust.py`、`workflow.py`、`evidence.py`、`job.py`。
2. `retrieval/vector_store.py` + `embedder.py`——理解 user_id 隔离和 distance→score 转换；
   再看 `retrieval/hybrid.py`——理解 RRF 怎么融合向量和关键词排名（§6.1）。
3. `ingestion/parser.py`（统一解析入口，§7.1）+ `service.py`（判重，§7.2）+ `chunker.py`
   （切块，§7.3）——理解"写路径"和两库一致性。
4. `agents/`——一个个看，都很短；先 `base.py`，再 `job_description_agent.py`（最典型），
   `resume_agent.py`（注意 `_DraftExtraction` 那层清洗逻辑），最后 `trust_agent.py`
   （研究核心）。
5. `orchestration/orchestrator.py`——重点理解那个"4 份草稿"的计数（§4）。
6. `api/app_service.py`——看所有东西怎么被接成一个应用；再看 `server.py` 的路由。
7. `ui/streamlit_app.py` + `api_client.py`——前端怎么薄薄地包在后端 HTTP 接口上面。
8. 想扩展时，再回头读 `architecture/decisions/` 里的 ADR（含 ADR-0010 混合检索）；
   想跑/部署，看 `.github/workflows/ci.yml` 和 `Dockerfile`/`docker-compose.yml`。

**一条贯穿始终的原则**：这个仓库是原项目的**忠实移植**，不是重新设计——除了"用
LangChain/框架生态标准件替代手写代码"这一类改动（本身就是这个移植项目的学习目的），
改别的模块前先去 `/Users/joe.xu/repo/trustresume` 对照原实现确认意图（见 `CLAUDE.md`
和 `SYNC.md`）。
