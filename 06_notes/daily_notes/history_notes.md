### 1.研究点：

可以投：SIGKDD,ICML，IJCAI，SIGIR,KR

**从实验现象发现动机**。**检索内容和模型偏好不对齐**，**增强检索内容跟问题之间的上下文推理能力**

可以优化：1.数据构建、2.训练的模块(文档精炼中间件，检索器emb和rerank，生成器)、3.训练的方法（Reward的设计)，在这些之间进行排列组合

生成子query不够好->可以通过正确答案在检索文档里面的出现字数，设计reward，也可以通过检索文档和gt文档的重叠率来设计reward，或者设计多种reward，包括监督最终生成结果（奖励设计用LLM_as_Judge而不是特定的计算公式），RL训练LLM或者检索器的embedding和Rerank模型，能检索到的结果更好

设计reward：正确答案在检索结果里面的出现率，检索文档和gt文档的重叠率（如果有gt文档），最终生成结果（用LLM_as_Judge）

训练的模型：LLM本身，检索器的embedding和Rerank模型

奖励设计用LLM_as_Judge而不是特定的计算公式

模型的function可以不只是search

相似的研究方向：DeepSearch，Agentic RAG+RL

### 2.目前要做：

（1）整理推理能力增强的论文（2）选模型，看哪些大模型对这几个数据集本身就能测很高，目前还需要在NQ和多跳数据集上测（3）目前进展：按照MA-RAG的思路说一下如何构建数据集，做了一些实验来验证通过什么样的prompt来构建数据**（4）确认人大对齐是否有问题分解、使用open-rag提到的检索器、分步骤里面的note和重排验证、nq数据集验证**（5）串行能否改并行（6）测CoRAG的分解好的子问题数据集，检索器用同一个，比较和我写的测出来结果如何（CoRAG好）（7）测一下我写的框架是否隐含了思维链：没（8）看一些问题分解的文章，看人大的讲解的最后一段(9)CoRAG用检索+子问题和答案不如只用子问题和答案

#### 目前要做细节

1.多阶段奖励：检索质量奖励、推理连贯性奖励、证据相关性奖励

2.难度自适应的数据生成：根据问题难度动态调整MCTS配置

3.对每个节点多次评估，取中位数

4.多工具：代码解释器，search

5.针对多种问题：多步计算，多轮检索

6.换训练方法：GRPO+DPO，参考Tool-Former



#### 发现

**2025.11.3**

RoleRAG是基于人大FlashRAG库里的数据集和索引开发的,FlashRAG提供了很多数据集，其他方法的复现，以及预处理的索引

**2025.11.7**

RoleRAG标数据的代码可以并行跑，不用跑那么久

如果子问题问的不好，对应的检索也检索到无用的信息

**2025.11.10**

FlashRAG提供的检索器每次检索内容一样，他的语料库整理得更好

我最开始跑的result.70B.0应该是用错了index，导致最开始的问题检索到的结果很差

在vscode右上角在此文件禁用/启用折行，可以把jsonl文件转换为一行显示或者多行显示

我的框架很多答案输出为空，怀疑是没有按<final_answer>…</final_answer> 输出的问题,prompt里面加入example会提高模型的指令遵循度

self-rag的match评估方式存在缺陷

**2025.11.12**

llama3.1-8B-Instruct和llama3-8B-Instruct不是同一个模型

**2025.11.13**

读取NAS上的文件会拖慢速度，最好都先复制到home本地

RoleRAG标数据生成查询图的prompt有问题

**RoleRAG标数据生成的摘要如果没有转义符可能会导致json读取失败，导致最后回答为unknown**

**2025.11.14**

trl,Llama-factory都包含了直接训强化学习的库

**2025.11.17**

reasonrag建索引会爆内存

修改了reasonrag虚拟环境的retriever.index_builder.py函数的_process_batch方法

**2025.11.19**

max_model_len控制 **输入prompt + 生成文本** 的总token数量，越大需要的GPU内存越多，常见值是1024,2048,4096

max_num_seqs控制**调度器在同一时间周期内考虑的所有活跃序列的数量上限**。

max_num_batched_tokens控制**单个调度周期中处理的所有序列的 token 总数上限**

**2025.11.21**

RoleRAG里面给的来自于hotpotqa的训练数据在hotpotqa数据集里面找不到？**找得到**

**2025.11.25**

ReasonRAG标完output为空的原因是模型没有按照指定的输出格式输出<query>..<query>或者<answer>..<answer>导致document_analysis阶段提取不到query报错

**2025.11.26**

修改ReasonRAG的prompt试图解决模型不按格式输出的问题，加了3-shot example，进一步优化了begin_reasoning_prompt和document_analysis_prompt

**2025.11.27**

看目前标出来的偏好数据集发现数据量太少了，主要原因有两个:(1)模型的输出太固定了,**修改了flashrag库里面的默认seed**尝试看是不是seed的原因,发现确实跟seed有关(2)模型不按格式输出导致提前报错了，不是max_token太小的问题，不是top_p的问题，不是top_k的问题，与temperature有关，temperature对指令遵循度的影响大，

目前看ReasonRAG可以改中间推理过程的一些prompt组织，以及改训练的强化学习方法

RoleRAG用了10089条hotpotqa数据，6489条musique数据，9076条2wikimultihopqa数据,总共25654条数据；

ReasonRAG用了2843条hotpotqa数据，1056条2wiki数据，704条popqa数据，总共13300条过程监督数据

那如果直接用ReaonRAG标好的数据在RoleRAG的模型上训，效果就好，说明是有效的

**2025.11.28**

目前的问题：

1.标数据慢、标的数据生成不了偏好对，**如何在约束模型输出特定格式的情况下，使得模型的输出具有多样性**，约束temperature加prompt

2.训练后期会爆显存

解决方法：

加载量化模型是不是需要额外的配置，使得权重可以更少。**答案是没有，因为会先检查模型配置里面的quantization_config**

==采用StructuredOutputsParams约束模型回答，空率从20%降为2%==，修改了flashrag库里面的默认seed，防止每次回答一样

用KTransformers？

用DeepSpeed技术?

**2025.12.2**

vllm框架里面已经实现了量化，不需要额外安装量化awq库

**2025.12.4**

pipeline.run是为每个问题生成一条链，pipeline.search是为每个问题生成多条链

能否通过ReasonRAG给出的过程链倒推原始用的哪些数据：能，但是数据量跟论文对不上，给出的数据倒推出的原问题只有3482个，论文中说有4603个，并且貌似存在popqa测试集泄露的问题

**2025.12.6**

reasonrag标的还是有空，因为在评估路径打分的时候也没有按格式输出分数。

或者是没有按So the next query is <query>[\s\S]*?</query>来回答，省去了so，加了标点等等

**2025.12.12**

dpo训练后期显存爆炸调小cutoff_len可解决

**2025.12.13**

**关于rolerag的prompt tuning做法**

推理时模型能“识别”这些 [qgg1]… 的前提只有两种：

模型权重里已经合并了它们的词表和 embedding

训练或保存 checkpoint 时，把虚拟 token 和对应 embedding 一起存进了模型（和 tokenizer）。
这时推理只要用同一个模型目录加载（tokenizer 和模型对应），它就天然认识这些 token。
推理前显式加载虚拟 token embedding

通过 load_tokens() 把 [qgg…] 等 token 加进 tokenizer 词表，并把你训练好的 embedding 填进 embedding 层 / lm_head。
然后再进行推理。
如果两者都没做，推理时看到的只是普通字符串，tokenizer 会把它们拆成若干子词（如 [, q, gg, 1, ]），模型不会有任何 prompt tuning 效果，只当普通字符处理。

**2025.12.16**

写了一个将RoleRAG标好的链的数据拆分成不同模块的prompt tuning的输入数据的函数split_result.py

**2025.12.18**

某些版本的 Accelerate 在解析嵌套的 `zero_optimization` 时可能有 bug。

用以下方法写会退回到zero2

deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  zero_optimization:  
    stage: 3  # ← 如果缩进不对，这里可能被忽略
    zero3_init_flag: false

**2025.12.26**

调DeepSeek API时发现他们的API提到了多轮对话，按指定Json格式输出，按指定前缀接着输出等模式，可能会对后续研究有用

**2026.1.3**

目前效果差，感觉是标的数据集处理和训的超参数问题，用GPT5.1分析两个训练数据集的差异如下

- **Prompt 划分错误导致“训练分布和推理分布不一致”**
  - RAG DPO：模型学到的是“在 system 里有一套规范说明，user 只给问题；然后我根据规范决定答还是生成 query”；
  - RoleRAG DPO：模型学到的是“user 每次都丢给我一堆操作指南 + 示例 + 问题”，这跟你真实 RoleRAG 流水线里的 Prompt 很可能不一样。
  - 结果：DPO 把模型往“只在这种超长、模板式 user 输入场景下工作”上推，而你实际推理时用的是另一套 prompt 模板，自然容易退化。
- **负例质量差、难度低，DPO 信号不够“细”**
  - RAG 数据里，正负都是“看起来像人写的解答”，差别在是否完整/准确/遵守格式，属于 fine-grained preference。
  - RoleRAG 里很多负例是明显坏输出（空、占位、把系统提示当回答），DPO 训练很快就“学会”了这些简单模式，但这对真实难例的区分能力帮助有限。
  - 你的模型更容易过拟合这些“垃圾负例模式”，而不是真正提高检索-推理-回答质量。
- **监督目标和你真正想要的评价指标有偏差**
  - RAG_ProGuide 中，很多 pair 是在“最终回答质量”或“证据抽取质量”上打分；
  - RoleRAG_ProGuide 中：
    - 浅层（layer 0）有大量 `Query Generation` 样本（4373 条 0 层 vs RAG 的 3482），偏向于“如何分解 / 生成下一步 query”；
    - 深层（layer 3、4 以后）样本非常少，几乎没有对“最后一步 answer generation”做细致偏好对比。
  - 如果你评测的是“最终答案 EM/F1”，而训练 DPO 更偏向前几步 query 生成和粗糙过滤，自然就会出现：RoleRAG 学会了“怎么演”，但最终答题并没比 RAG 更好。
- **数据量更小 + 噪声比例更高**
  - 样本数：RAG =13289，max(prompt+response)=2390,RoleRAG =9573,max(prompt+response)=28387,只有32条样本的cutoff_len大于2048，本身监督信号就少 28% 左右；
  - 再加上负例里各种结构性错误、无意义输出，等价于有效样本数更少；
  - 用同样的学习率 / epoch 数训练，很容易要么学不到东西，要么对坏模式过拟合。

**2026.1.4**

把RoleRAG_ProGuide数据集中Instruction和input重新划分，训练llama时cutoff_len改为2500,结果保存在dpo_RoleRAG_ProGuide_v2中，重写了prompt_v2和pipeline_v2，约束输出为json格式

**2026.1.5**

分析RoleRAG_ProGuide训练的模型效果差的原因：（1）数据量比RAG_ProGuide少（9573 VS 13289)，数据分布不均衡：RAG_ProGuide的pair 偏向于在“最终回答质量”或“证据抽取质量”上打分；而RoleRAG_ProGuide偏向于在“子问题分解上”打分（2）模型输出了太多思考的过程，没有遵循指定格式输出（3）reasoning_prompt让模型误以为回答子问题就输出answer就结束了，实际上还没回答原问题

**2026.1.28**

vllm部署的不同模式，可以采用tokenizer.apply_chat_template将message转换为特定模板的字符串输入，也可以直接使用llm.chat传入message

search-R1中找到了几种典型的错误类型

（1）子问题分解错误

![image-20260128192315130](https://s2.loli.net/2026/01/28/kNRGyPSouXmAtLw.png)

![badcase1](https://s2.loli.net/2026/01/28/PrqOmIWcG76SsYl.png)

（2）检索不到结果的错误

![image-20260128192241637](https://s2.loli.net/2026/01/28/JK5tguo8ZjEbBqH.png)

![img](https://workspace-zb-cdn.quark.cn/a6d90ca436934edf9153a14d115f3a51%2Fo%2F1769597961048.png?auth_key=1772191020-0-0-601eb2b76d0b9fda6a0b36ff5b771d07#except=auth_key)

（3）检索噪声导致的错误

![image-20260128192728947](https://s2.loli.net/2026/01/28/vOEQ6wSTpyV3RkC.png)

![img](https://workspace-zb-cdn.quark.cn/5e1f5b51b8d84fd597bece843383a065%2Fo%2F1769597206874.png?auth_key=1772191920-0-0-b68c2fdb5e00ce411bf90e0302545fc2#except=auth_key)

**2026.2.22**

采用了claude的方案来解决模型不遵循指令输出格式的问题，方案如下，需要修改FlashRAG.generator.generator.py源码：

```
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        guided_json = generation_params.pop("guided_json", None)
        json_schema = generation_params.pop("json_schema", None)
        guided_decoding = generation_params.pop("guided_decoding", None)
        structured_outputs = generation_params.pop("structured_outputs", None)

        sampling_params = SamplingParams(**generation_params)

        if guided_decoding is None:
            if guided_json is None and json_schema is not None:
                guided_json = json_schema

            if guided_json is None and structured_outputs is not None:
                if isinstance(structured_outputs, dict):
                    guided_json = structured_outputs.get("json")
                else:
                    guided_json = getattr(structured_outputs, "json", None)

            if guided_json is not None:
                guided_params_cls = None
                try:
                    from vllm import GuidedDecodingParams

                    guided_params_cls = GuidedDecodingParams
                except Exception:
                    try:
                        from vllm.sampling_params import GuidedDecodingParams

                        guided_params_cls = GuidedDecodingParams
                    except Exception:
                        guided_params_cls = None

                if guided_params_cls is not None:
                    guided_decoding = guided_params_cls(json=guided_json)

        if guided_decoding is not None:
            try:
                sampling_params.guided_decoding = guided_decoding
            except Exception:
                pass

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params)
```



修复 guided decoding，使用 JSON Schema 而非regex（最直接）当前的 regex 约束太松散，换成 vLLM 的guided_json，用严格 JSON Schema 强制输出格式：

**2026.2.26**

gpt-5.3-codex分析两个训练集的差异：

- **样本量差异**：RAG_ProGuide 有 13289 条，RoleRAG_ProGuide 有 9573 条，前者多约 39%，DPO 更容易学稳。
- **长度/截断差异（最关键）**：在你常用 `cutoff_len=2048` 下，RAG 的截断率约 1.44%，RoleRAG 约 27.17%；`cutoff_len=2500` 下仍是 0.2% vs 7.3%。RoleRAG 被截断太多，偏好信号被破坏。
- **标签噪声**：RoleRAG 存在 57 条空 rejected，DPO 有效对比对减少且带噪。
- **任务分布更混杂**：RoleRAG 里 retrieval/reference/query 模板占比更高，很多 pair 学到的是“格式偏好/步骤偏好”，不一定直接转化为你关心的最终 QA 指标。
- **输出更长更难**：RoleRAG 的 chosen/rejected 平均长度显著更长，单卡小 batch 下梯度方差更大，训练曲线更抖、效果更不稳定。

**2026.3.2**

可以用vscode的vim插件格式化具有\n等的文本

#### 讨论

##### 2025.10.30

（1）确立baseline，PRAG，self-rag

（2）找到了子问题分解的好的corag数据集，对子问题进行剪裁，即有的子问题可能对于回答原问题是没有帮助的。主要目标是找到一个数据集能训模型。使模型能更好地分解子问题

（3）回顾人大的视频

**2025.11.7**

（1）确立了baseline为RoleRAG

（2）首先看数据集回答错的是为什么，找是否存在子问题分的不好，会影响最终回答的例子；在RoleRAG框架标的开发集和我们的框架标的开发集里面找，再在RoleRAG没有标的训练集里面找

**2025.11.9**

（1）我原本写的迭代框架好像不存在子问题分解与原问题不相干的情况

（2）确认rolerag是否存在子问题分解的不好或者直接分的与原问题无关的情况

**2025.11.13**

（1）rolerag好像不存在子问题分解与原问题不相干的情况

（2）rolerag的错点主要是子问题理解上，需要**对子问题进行矫正**

（3）调研一下rag里面用强化学习咋训起来的，强化学习有一类是中间过程奖励,一类是只用最终的结果，调研reasonrank,reasonrag，RAG-gym代码

**2025.11.20**

（1）我们做的是检索增强的多跳QA问答任务，总目标就是让模型回答得更准确。我们首先复现了一个benchmark，分析目前的方法为什么回答错的，把回答错因归类成了两类：一类是检索相关的错误，跟检索器的好坏，检索结果的处理有关；另一类是模型分解子问题的错误，原问题太复杂了导致模型分解子问题分的不好，最开始做的实验是从人大重排的论文里面得到启发，想试试把检索文档随机排序或者做个摘要再输入给模型，看能不能得到改善。然后发现模型回答的准确率提升不高，然后根据一些复现结果去分析检索器的好坏占比很大，但这块其实跟我检索知识库怎么处理的占比很大，不好优化。然后针对另一个错因，我们首先想的是对子问题进行裁剪，因为可能利用模型分解出来的子问题和原问题不相干，这样裁剪掉不相干的子问题后模型可能会回答对，但我们去复现了RoleRAG这篇论文，发现他的方法模型分解的子问题是跟原问题相关的，问题答错的错因主要是子问题理解上，需要**对子问题进行矫正**。比如问The football manager who recruited David Beckham managed Manchester United during what timeframe?因此需要专门对模型拆分子问题做一些处理，针对这块的优化目前主要是使用微调，强化学习等方法，统一的框架就是标数据，训练。然后我们就调研了一些强化学习的论文，目前结果监督强化学习方法比较多，但是过程监督很少，一篇nips一篇emnlp。然后复现跑通了，目前想的是以rolerag为baseline，在rolerag上改进，因为他的方法是没用强化学习的，就想用目前复现跑通的强化学习的框架在rolerag的基础上标过程监督的数据，再进行强化学习训练

**2025.11.23**

检查RoleRAG其他数据集是否也有未知数据->标点错误

**2025.11.24**

调研子问题矫正相关论文

调研子问题困惑度相关

**2025.12.4**

能否通过ReasonRAG给出的过程链倒推原始用的哪些数据：能，但是数据量跟论文对不上，给出的数据倒推出的原问题只有3482个，论文中说有4603个，并且貌似存在popqa测试集泄露的问题

**2025.12.12**

做ppt,首先讲pipeline，然后讲怎么标数据，最后讲怎么训练

**2025.12.19**

数据和模型对齐了，测一下效果

**2026.3.19**

用DPA-RAG标好的正负文档对数据/命令openclaw标数据，去DPO一个打分模型，在reasonrag/tree-grpo的rollout过程中，用打分模型输出的文档分数作为reward，train一个embedding模型，再用policy model在训完的embedding模型上rollout，再grpo

### 3.实验流程

（1）实验1：用基本的RAG流程，让模型回答popqa,trivaqa的问题，找出回答错的，看为什么回答错，baseline采用self-rag，没用检索器，直接用的self-rag现成的检索结果。尝试对检索结果进行rerank和note

（2）实验2：用自己写的迭代流程，让模型回答2wikimultihop的问题，看回答错的错因，检索器用的flexrag。发现回答错的问题两类错点：子问题本身分的就不好，检索内容差，都没检索到结果

（3）实验3：用8B的模型来测Rolerag标好的数据集，然后看看对这些问题数据能不能用70B的打个分，然后在对最后打分的数据训练一个模块

目前在hotpotqa_dev发现了例子,还需要修改删除的QA对顺序，选不同的数据集实验



### 4.评估指标

- Exact Match（EM）

  预测字符串与标准答案在“归一化”后是否逐字符完全一致。归一化 = 小写 + 去标点 + 去冠词 + 合并多余空格。值为0或1 

- F1分数

​	把预测和标准答案都当成词袋（bag-of-words），计算 precision、recall，再求 F1。值为0-1

**第一步：理解 TP, FP, FN**

为了计算精确率和召回率，我们首先需要定义三个基本概念（以抽取词语为例）：

- **True Positive (TP, 真阳性)**: 模型预测为某个词，**并且**标准答案里也有这个词。（预测对了）
- **False Positive (FP, 假阳性)**: 模型预测为某个词，**但**标准答案里**没有**这个词。（预测错了，多预测了）
- **False Negative (FN, 假阴性)**: 标准答案里有某个词，**但**模型**没有**预测出来。（预测错了，漏预测了）

**第二步：计算精确率和召回率**

- **精确率** = `TP / (TP + FP)`
  - **含义**: 在你所有预测为“正”的结果中，有多少是**真正正确**的？
  - **通俗理解**: 查得有多准？宁缺毋滥。
- **召回率** = `TP / (TP + FN)`
  - **含义**: 在所有**真正为“正”**的结果中，你成功找出了多少？
  - **通俗理解**: 查得有多全？宁可错杀，不可放过。

**第三步：计算 F1 分数**

F1 分数是精确率和召回率的**调和平均数**。

- **公式**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**为什么是调和平均数？**
调和平均数的一个重要特性是，它会更“偏向”两个数中较小的那个。这意味着，只有当精确率和召回率**都比较高**时，F1 分数才会高。如果其中一个很低，F1 分数也会被拉得很低。这促使模型在“查准”和“查全”之间取得平衡。

- Accuracy

​	只要 **任意一个** 标准答案字符串 **出现在** 预测字符串里（大小写敏感，原文不做归一化），就算 1，否则 0。==一般用于选择分类题，用于问答时和Match是一个意思==

- Match

​	遍历标准答案列表，只要 **有一个** 答案在预测里出现即 1，否则 0。

### 5.数据集、知识库和检索器

- NQ：307,373 个训练示例、7,830 个开发示例和 7,842 个测试示例
- TriviaQA
- HotpotQA：Json数组，list[dict]
- 2wikimultihopQA

### 知识库

DPR项目(Wikipedia 2018)：DPA-RAG,Auto-RAG,airrag，RoleRAG

Atlas2021年语料

Atlas2020年语料：self-rag

MS MARCO:

# 召回

「**检索器**」= 编码器 + 索引 + 搜索逻辑

- **编码器**：把文本编码成向量

- **索引** =「怎么存向量」——把高维向量组织成某种数据结构（Flat、HNSW、IVF、PQ 等），以便后续快速缩小搜索范围。
- **搜索逻辑** =「怎么找近邻」——在这个数据结构里执行近似或精确的距离计算，返回与查询向量最相似的 top-k 结果。

**Elasticsearch**是搜索引擎，不是搜索算法

## 稀疏检索

文本表示成的向量大多位置都是0

### TF-IDF

### BM25

### SPLADE

## 稠密检索

### embedding模型

- E5系列：E5-base-v2，E5-large，multilingual-e5-base

- BGE系列：BGE-base-en-v1.5（ReasonRAG）
- 双塔系列：DPR（DPA-RAG），
- **Contriever-MS MARCO** → 编码器（在 MS MARCO 上微调过的 Contriever）

### 索引结构

FAISS不是某个具体的检索算法，而是一个包含了很多检索算法的框架

- IVF（Inverted File Index)：kmeans聚类，待查询query先与nprobe个簇中心算，再簇内算
- HNSW（Hierarchical Navigable Small World）：多层图结构，从高层逐步往低层走![img](https://pic2.zhimg.com/v2-a321ae222dd9c5d14952fb2f85cd91ef_r.jpg)
- PQ：把单个向量分成若干段，每段单独做一个聚类，然后每段的向量以他最近的聚类中心的类别数字代表，query来了以后只需要计算跟每段的所有聚类中心的距离
- Flat：朴素的每个向量都要算一次相似度

### 向量数据库系统

- Milvus

- Contriever-MS MARCO:稠密索引，self-rag
- E5-base-v2(E5-base)：Text embeddings by weakly-supervised contrastive pre-training，Auto-RAG,CoRAG，RoleRAG
- BM25：稀疏索引，PRAG，CoRAG
- multilingual-e5-base:Multilingual e5 text embeddings: A technical report，airRAG，
- E5-large:Text embeddings by weakly-supervised contrastive pre-training

# 精排

DPA-RAG对bge-reranker进行了微调

# 生成器

超参：

- max_model_len控制 **输入prompt + 生成文本** 的总token数量，越大需要的GPU内存越多，常见值是1024,2048,4096

- max_num_seqs控制**调度器在同一时间周期内考虑的所有活跃序列的数量上限**。

- max_num_batched_tokens控制**单个调度周期中处理的所有序列的 token 总数上限**
- temperature:重新缩放模型输出的原始概率,0~2的浮点数，`0.1` 到 `1.5`。对于事实性问答，常用 `0.1~0.5`；对于创意写作，常用 `0.7~1.0`。
- top-p:会从概率最高的词开始累加其概率，直到累加和刚好超过 `top_p` 这个阈值。然后，只从这个“候选池”中抽样，并重新归一化概率。池子外的词概率直接归零。
- top-k:只保留概率最高的 `k` 个词作为候选，然后在这 `k` 个词中重新归一化概率并进行抽样。
- do-sample:这是一个**总开关**。如果设置为 `False`，那么无论 `temperature`、`top_p`、`top_k` 怎么设置，模型都会使用贪心搜索（即总是选概率最高的词）。必须将其设置为 `True`，上述所有随机性控制参数才会生效。
- repetition_penalty:通过惩罚已出现过的词来**间接增加多样性**，避免模型陷入重复循环。
- max_tokens:指的是 **生成的新 token 的最大数量**，即 **不包括输入 prompt 的 token**
- cutoff_len就是模型能够处理的最大序列长度（以token为单位）。如果输入的文本超过这个长度，模型会自动截取前cutoff_len个token，后面的token会被忽略

### 6.基线方法

论文清单：

The-power-of-noise:有代码，已复现，DPR检索器，有检索代码，NQ数据集

**self-rag**：有代码，已复现，Contriever-MS MARCO检索器，PopQA、TriviaQA数据集，没有链式推理的能力

Ret_robust：有代码

Backtracking Correction：有代码，从后往前优化的理论可借鉴到多跳问题上

deeprag：无代码

deepnote：无代码

**prag:**有代码，已复现

MCTS-RAG：仅推理，未微调

AirRAG：暂无代码

AutoRAG：有代码，只有推理

Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation

**CoRAG：生成多条检索链，有代码**

RAG-gym：有代码,强化学习，过程奖励

CR-planner：无代码

**DPA-RAG：有代码**

**open-RAG**：有代码。self-rag改进版

ARise：有代码

RATT:有代码

**ReasonRank:强化学习，有代码**,

ReasonRAG：强化学习，有代码，nips

5.后期要做：（1）利用选的模型设计prompt标数据，讲故事（2）标完的数据结合RL（3）改进检索器，可以当成第二个工作

索引排序那个会影响对齐，摘要也会影响对齐，然后在统一的设置下做个实验对比，看看更精细的是咋影响的

参考：understand,airrag

https://github.com/RAG-Gym/RAG-Gym树形结构，有推理路径搜集的代码，流程比较详细

https://github.com/microsoft/LMOps/tree/main/corag 有直接在多跳上测试的，可以直接换模型，也有多跳数据集链接

https://openragmoe.github.io  self-rag扩展到多跳上

https://github.com/OpenCausaLab/ARise 树形结构，代码有中文注释

### 7.开题报告问题

（1）实验做的较少，偏好对齐研究方案需要有初步的实验结论来进行分析

（2）推理能力优化的研究方案也需要有一些实验结果来验证方案是否可行

（3）偏好对齐研究方案的数据增强部分，增强方法的设计如问题重写需要设计prompt，这类prompt需要参考别的文献是如何设计的

（4）存在中英文逗号混用、逗号和顿号误用、图标题字号不对、缺少句号、参考文献里期刊和会议误用等格式问题

比如一个问题：A和B是不是有相同的国籍，我逐步分解为A的国籍是什么，B的国籍是什么，他们两国籍是否相同，但是我的代码处理逻辑在处理子问题3的时候丢失了前面的回答，导致回答错误，并且在处理子问题3的时候其实不再需要检索文档。除了这种问题还有其他类型的问题，比如A的出生地所在城市最著名的桥梁是什么，就要分解成A的出生地所在城市在哪里，（假如回答是B城市），下一步就要问B城市最著名的桥梁是什么，这又是需要检索的

### 8.环境配置问题汇总

ModuleNotFoundError: No module named ‘pyairports‘

解决方法：

https://blog.csdn.net/weixin_48389642/article/details/153334402

### 实验分析

#### 单跳



#### 多跳

- 文档分块问题导致段落零碎
- 子问题之间不能太过于相似，不是换个说法问子问题，导致有的子问题没问上
- 子问题都没检索到相关信息（很多）
- 最终答案为空
- 多义词问题
- 多定语限定，逐个定语分解导致范围扩大
- 检索的时候怎么限定实体范围？是直接用问题去查询相关文档吗？
- 子问题太过于间接，没有直击要害![image-20251015200937563](https://s2.loli.net/2025/10/15/2Ia9LyDPFqbun1e.png)
- 拆分步数不够

##### hotpot非检索造成的错误特例

- 回答7但标准答案是seven
- 子问题分解步数不够，还没有问到最后问题就结束了：2
- 子问题分的不好，第一个子问题太宽泛了：1
- 检索结果里面有，但是模型没提取出来信息或提取错误：2
- 子问题都回答对了，但最终答案错了，不知道是不是因为最终回答时也加了检索文档的缘故：1
- 子问题分解正确，子问题的检索结果里就没有，或者有相关的但是不是符合答案要求的知识段，需要模型根据常识判断：5
- 原问题答案本身奇怪，一个搞笑的问题：What criteria subjectively qualifies the Houston Astrodome to be called the Eighth Wonder of the World? 
  answer：comparable to the seven Wonders of the World
- 检索内容跟数据集提供的标准答案矛盾：How many novels are there in the series of novels of which Shadows in Flight is the tenth novel ?
- 其他都对，但最后回答为空？**不知道为什么**

##### RoleRAG在hotpot dev集上的错误

- 原问题范围太宽，有多种可能答案，比如问一个人的职业：1

- 检索结果中没有标准答案：2

- 子问题分的不好（指代不明确，限定条件不够）：1

  The football manager who recruited David Beckham managed Manchester United during what timeframe?

  Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?

- 输出格式、同义词

- 检索结果中有噪声干扰了正确答案：1

- 根据检索文档回答内容回答得不好

- 标准答案不够好：1 
