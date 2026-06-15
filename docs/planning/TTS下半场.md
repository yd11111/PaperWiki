TTS下半场，2026下半年。

未来：现在我们目标是什么，目前有什么？能够做什么？

要回答这个问题，首先回顾一下历史。

历史：24年末到26年现在，这个过程中，我们做了什么？留下了什么？还能做什么？

### 1. 历史&现状

TTS大模型应用的历史：小高老师精品导航音色，小徳对话。 -> 大模型（阶段一VALLE -> 阶段二semaTTS -> 阶段三PilotTTS）

**第一个阶段：**

做了什么？自有数据的VALLE复现，第一次大模型的链路跑通、上线小高老师、小徳小徳。

留下了什么？

数据处理：基础数据处理Pipeline。Podcast&Emilia 20w小时基础数据。

模型训练：一套基础的数据dataloader和训练pipeline。

工程链路：后续更新模型其实工程链路就主要是替换模型了，针对模型的推理优化。

还能做什么？表现力&稳定性待优化，推理效率待优化。数据积累+数据Pipeline。

总结：使用大模型替换小模型，打通链路，实现大模型落地。

**第二个阶段：**

做了什么？自研Tokenizer +（重训LLM+FlowMatching）。落地：小高、小徳、VoiceAgent、客服。尝试小高的方言&情感落地。

留下了什么？

精品音色SFT流程：打底+目标人全量微调策略。后训练数据增广。

一套目标人的情感&方言合成能力拓展链路。零目标人风格录制数据，底模零instruct能力基础。

还能做什么？数据少，质量差。数据pipeline急需优化->数据质量优化+分层+扩源。

总结：大模型升级（效果、效率），尝试拓展能力+使用场景。

**第三个阶段：**

做了什么？模型集各家之所长。数据Pipeline优化+数据扩源。

留下了什么？新数据pipeline。产出基础模型、方言&情感&副语言指令模型。

还能做什么？模型架构升级，提升效果上限。数据更多维度的标注，说话人，副语言，情感，风格。

总结：模型快速跟进业界先进设计，补足数据处理基础能力。

---

在整理了我们在做什么的同时，我想先不要直接就回答我们往下要做什么，看看同行在这个阶段做了什么？

### 2. 业界动态

> 信息来源：PaperWiki知识库529篇论文 + 行业全景报告（30+团队/实验室调研）+ 公开论文和产品信息

#### 微软

2023-2024年TTS领域产出最密集的学术机构，两条主线定义了行业范式：

**VALL-E系列**（codec language model路线）：VALL-E（2023，开创codec LM做zero-shot TTS）→ RALL-E（2024，CoT prompting提升鲁棒性）→ MELLE（2024，去掉VQ直接用连续表征做AR，与CUHK Helen Meng合作）→ FELLE（2025，ACM MM 2025）。这条线的核心思路——把语音当token用LM建模——成为整个行业的出发点。

**NaturalSpeech系列**（扩散/表征解耦路线）：NaturalSpeech2（2023，latent diffusion做zero-shot语音和歌声）→ NaturalSpeech3（2024，ICML，FACodec属性分解+扩散生成）→ E2 TTS（2024，Interspeech，非自回归zero-shot TTS）。

**VibeVoice**（2025-2026）：微软最新的语音对话系统，连续VAE tokenizer + next-token diffusion实现端到端，支持4说话人/90分钟长对话。表明微软并未完全退出，而是从独立TTS论文转向多模态对话系统。

但2024年后核心团队确实发生了重大人才流失：Xu Tan → 月之暗面（带走Kai Shen/Zeqian Ju/Yichong Leng等NaturalSpeech核心人员，产出Kimi-Audio）；Zhuo Chen/Jian Wu → 字节Seed团队。Speech组更名为"Multi-Modal Interaction Group"，NaturalSpeech/VALL-E系列停更。

**信号：微软定义了TTS-LLM的两大范式（codec LM + 扩散表征解耦），但核心团队解体后这些方向被中国团队接手推进。VibeVoice说明微软仍在做，但重心已从独立TTS学术转向多模态产品整合。**

#### 谷歌

TTS-LLM范式的另一个源头。SoundStream（2021，端到端neural audio codec，定义了RVQ离散化框架）→ AudioLM（2023，首次证明语音可以用language model建模）→ SoundStorm（2023，非自回归并行生成codec token）。这三篇与微软的VALL-E系列共同奠定了整个codec LM时代的基础。核心人物Neil Zeghidour后来离开去了Kyutai/Moshi。

近期转向LLM-native路线：Gemini原生语音能力（Gemini 3.1 Flash TTS，72+语言，Audio Tags精细控制）+ 收购整合（2026.01收购Hume AI，获得情感语音能力）+ 安全侧SynthID语音水印。产品层通过Gemini Live进入实时语音对话。

**信号：谷歌和微软共同定义了codec LM范式（SoundStream/AudioLM + VALL-E），但两家的核心人才都在外流。谷歌现在走"LLM-native语音"路线，语音是Gemini的原生模态，不再做独立TTS研究。**

#### 阿里通义实验室/FunAudioLLM

CosyVoice系列是国内开源TTS标杆（GitHub 9K stars）。

演进线：CosyVoice（2024.07，监督语义token + Flow Matching）→ CosyVoice2（2024.12，chunk-aware FM，流式里程碑）→ CosyVoice3（2025.05，1.5B参数，100万小时数据，DiffRO可微分reward优化）→ Fun-Audio-Chat（2025.12，双分辨率全双工对话）。

核心赌注：监督语义token + 可微分reward做post-training。DiffRO是业内唯一绕过FM+vocoder直接在token级做reward优化的方案，但后续RRPO论文发现DiffRO会导致自然度下降（N-MOS 3.61 < SFT 3.72）。35+篇论文产出，全栈开源。

**信号：稳定为先的路线，开源生态最完整。但表现力提升空间受限于监督语义token的信息瓶颈。**

#### 阿里Qwen语音团队

Qwen3-TTS（2025.05正式发布），走LLM-native路线——语音是Qwen大模型的原生输出模态。5M+小时数据训练，97ms TTFB，低延迟能力业界领先。与通义实验室是两个独立团队，技术路线不同。

**信号：超大数据规模 + LLM原生输出 = 资源密集型路线。**

#### 小红书语音团队

国内布局TTS最系统的互联网公司之一，论文产出覆盖了从基础模型到对话场景的完整链路。

演进线：FireRedTTS v1（2024.09，400M AR，HuBERT自监督token，13种显式情感标签，CER 2.09% 胜 CosyVoice 5.68%）→ FireRedTTS-2（2025.09，Qwen2.5 Dual-Transformer，Whisper监督token，12.5Hz帧率，1.4M小时数据，3分钟/4说话人对话）→ **dots.tts**（2026.06，2B参数连续AR基础模型，Seed-TTS-Eval所有已报告系统最优，Apache 2.0开源）。

dots.tts是一个重要节点——小红书从离散token路线（FireRedTTS系列）跳到连续AR路线，且直接做到了公开系统SOTA：
- 架构：AudioVAE（48kHz→128维@25Hz）+ Qwen2.5-1.5B LLM（6.25Hz语义规划）+ 18L DiT AR-FM Head（full-history conditioning生成连续latent patch）
- 核心创新：SOAR自修正后训练（不需要reward model，用模型自己的off-trajectory状态做自监督修正）+ CFG-aware MeanFlow蒸馏（推理只需1次forward，NFE=4即可保持质量）
- 指标：Seed-TTS-Eval avg WER 2.95% / SIM 79.2，超越CosyVoice3（3.06%/75.3）、VoxCPM2（3.65%/76.7）、Seed-TTS（3.65%/77.8）
- 数据：1.5M小时（1.2M内部+300K开源），开源版基于Emilia子集
- 流式：MeanFlow NFE=4首包延迟54ms（interleaved模式），RTF 0.245

几个值得注意的技术选择（横跨FireRedTTS和dots.tts）：
- **路线跳跃**：从离散token（FireRedTTS用HuBERT→Whisper）直接跳到连续AR（dots.tts彻底不做量化），说明小红书判断连续表征是更有前景的方向
- **LLM只看语义摘要不看raw latent**：dots.tts的semantic encoder做4x下采样信息瓶颈，防止声学细节干扰LLM的全局语义规划——这是与DiTAR/VoxCPM的关键区别
- 12.5Hz超低帧率（FireRedTTS-2）：3分钟对话token从9000缩到2250，使长对话在transformer attention范围内可行
- 情感从显式标签（v1，97-100%准确率）转向上下文隐式推断（v2，83-93%但更自然）——足够的上下文建模可以替代显式控制
- 五步工业级数据管线，624K小时清洗到248K（保留率仅40%）

**信号：小红书是国内TTS投入最完整的互联网公司——基础模型（dots.tts SOTA）、对话场景（FireRedTTS-2 3分钟/4说话人）、数据管线（工业级五步清洗）三条线都有系统布局。dots.tts的SOAR后训练（不需要reward model的自修正）和连续AR路线值得重点关注。**

#### B站语音团队

IndexTTS系列是B站在TTS方向的核心产出，定位工业级视频创作场景。

演进线：IndexTTS（2025.02，GPT-style AR + 单码本VQ/FSQ 25Hz + BigVGAN2 + 字符-拼音混合BPE解决中文多音字，avg WER 3.7% / SIM 0.776）→ IndexTTS2（2025，AAAI 2026，在AR框架内实现精确duration control + GRL对抗训练解耦情感与音色，WER 1.008% / SIM 0.865 on Seed-TTS test-zh，情感迁移ES 0.887）。

几个值得注意的点：
- IndexTTS2的W_sem=W_num位置编码共享是零开销实现AR duration control的巧妙设计，直接解决视频配音的音画同步需求
- GRL（梯度反转层）做情感-音色解耦，用仅135小时情感数据+三阶段训练（全量预训→135h精调→全量回炉）实现零样本情感迁移
- 面向视频创作场景的产品导向——多音字可控、时长精确、情感迁移，都是视频配音的刚需

**信号：B站的TTS投入围绕视频创作场景展开，duration control和情感迁移是差异化方向。IndexTTS2的Seed-TTS-Eval指标（WER 1.008%, SIM 0.865）已进入第一梯队。**

#### 字节语音团队

论文产出最密集的TTS团队之一（40+篇），**同时押注多条路线，每个新工作都明确指出前一个的不足**。

演进线：Seed-TTS（2024.06，定义SEED-TTS-Eval行业评估标准）→ DiTAR（2025.02，ICML，批评两阶段级联"误差累积限制scaling"，转向连续向量预测）→ MagiCodec（2025.06，新一代codec基建，与X-LANCE合作）→ DiSTAR（2025.10，与X-LANCE合作，批评自家DiTAR"连续latent对偏移敏感"，回到离散RVQ + masked diffusion，0.3B即WER 1.66% < 人类1.80%）→ SpeechJudge（2025.11，与CUHK-SZ合作，TTS Reward Model，发现所有自动指标在自然度判断上接近随机）→ WavTTS（2026.06，与X-LANCE合作，直接波形DiT，跳过所有codec）。

同时在做推理加速（SplitMeanFlow 20x加速，部署豆包生产）。

**信号：字节的策略是多路并行快速迭代，不固定在任何一条路线。说明他们判断token/codec路线尚未收敛。SpeechJudge发现自动指标对自然度判断接近随机（WER准确率57.9%, SIM 44.5%, UTMOS 53.7%）——整个行业的TTS评估基础设施是不够的。**

#### MiniMax

零论文、纯产品路线做到TTS Arena #1。在GLM-TTS论文横评中，MiniMax-Speech CER 0.83%（中文）、WER 1.65%（英文），两项均为所有系统最优。完全闭源，无法判断技术路线。

**信号：工程优化和数据积累的价值可能被学术界低估。**

#### 阶跃

最积极拥抱RLHF的TTS团队。

演进线：Step-Audio v1（2025.02，130B统一模型）→ Step-Audio-EditX（2025.11，3B语音编辑，合成对比数据+PPO，情感准确率53.5%→70.7%）→ Step-Audio 2.5（2026.05，统一MoE基座，三分支特化：ASR可验证解码 / TTS preference RLHF / Realtime generative reward modeling，Arena整体胜率69.1%）。

v1到v2.5的关键转折：从130B理解+3B生成分离架构 → 完全统一为一个MoE backbone——"任务差异不再是架构差异，而是操作模式差异"。另外Step-Audio-R1系列开创了音频推理这个全新赛道。

**信号：EditX证明属性编辑是低成本提升可控性的路径——3B模型就能改善多家系统的情感表达。RLHF全覆盖+统一基座是阶跃的核心策略。**

#### Kimi/月之暗面

前微软TTS核心团队组建，Xu Tan带队。Kimi-Audio（2025.04），7B全栈音频基座（Qwen2.5初始化），13M+小时数据，完全开源（4.6K stars）。

技术选择独特：12.5Hz离散语义token + 连续Whisper特征双输入，并行text/audio head + chunk-wise FM。走全栈音频基座路线而非独立TTS。

**信号：前微软核心团队选择不做独立TTS，做全栈音频基座。**

#### 智谱

GLM-4-Voice（2024.12）是端到端语音聊天早期代表作（243引用）。核心创新是超低码率tokenizer（175bps，业界最低之一，12.5Hz单codebook）。后续GLM-TTS做了GRPO四维优化，但此后无新作。

**信号：超低码率路线有独特价值（序列极短→速度快+稳定），但智谱可能已将语音重心转向多模态大模型。**

#### 小米

出乎意料地活跃，形成**生成 + 评估 + 基座**三条线并进：

- 生成：ZipVoice（123M参数NAR，RTF 0.0125，比F5-TTS快24-33x）→ OmniVoice（600+语言，WER 0.84%中文，CMOS +0.44超过ground truth）
- 评估：TTS-PRISM（与清华吴志勇合作，12维诊断框架，发现去掉对抗负样本后评估LCC从0.717暴跌至0.150）
- 基座：MiMo-Audio 7B（被10+篇论文引用为baseline）

**信号：一致押注NAR路线，OmniVoice证明NAR质量已不输AR（WER 0.84% vs CosyVoice3 0.71%）且效率更高。ZipVoice 123M超越1B+模型提醒我们效率也是竞争力。**

#### 国际创业公司概览

| 公司 | 估值/融资 | 核心赌注 |
|------|----------|---------|
| **ElevenLabs** | $110亿（Series D $5亿） | 从TTS演进为"Audio OS"平台：TTS+Agent+配音+音效 |
| **Cartesia** | $1.6亿（NVIDIA参投） | SSM/Mamba替代Transformer，40ms延迟 |
| **Sesame AI** | ~$1亿+（a16z/Sequoia） | 对话韵律（CSM-1B开源，14.7K stars） |
| **Fish Audio** | 未公开 | Dual-AR + GRPO + 激进开源（31K stars） |
| **Voxtral/Mistral** | 估值$60亿+ | 4B开源TTS，胜率68.4% vs ElevenLabs |

Voice AI融资2023→2024暴增8倍达$21亿；Voice Agent方向占YC最新一期22%；ElevenLabs从$0到$500M ARR不到3年。

---

#### 总结：行业的热点、应用场景、待解决问题

**已饱和（不再是差异化）：** 零样本克隆、多语言（30+）、基本流式。

**正在拉开差距的方向：**

| 方向 | 2024→2026变化 | 关键事件 |
|------|:---:|---|
| Post-training对齐 | 1家→5+家 | DiffRO/SpeechJudge/GRPO各自创新 |
| 全双工/实时对话 | 2家→7+家 | HumDial首个竞赛（ICASSP 2026） |
| Codec设计创新 | 1-2种→10+种 | 共识未形成，分化极快 |
| TTS可控性 | 少量→多方 | Audio Tags/EditX/Voice Design |
| 上下文韵律/长语音 | 几乎无→3-4家 | FireRedTTS-2/Sesame CSM |

**整体TTS演进趋势：**
1. 独立TTS → LLM-native语音（Qwen-Omni, Gemini Flash TTS）——趋势明确但半年内不是主矛盾
2. 手工特征工程 → Post-training对齐（已从1家到5+家采用）
3. Transformer独占 → 架构多元化（SSM/DiT/Masked Generative/Dual-AR并行）

**技术路线选择：** Tokenizer/Codec层面10+种并行方案，共识完全没有形成——这既是风险也是机会。

**应用场景的选择：** VoiceAgent/实时对话是融资最热方向（ElevenLabs $110亿验证）；播客/长语音是产品形态已验证的增量场景；精准可控（instruct-following）是差异化空间最大的方向。

**待解决的核心问题：**
- 评估缺失：SpeechJudge发现所有自动指标对自然度判断接近随机
- 表现力 vs 稳定性的跷跷板：RL对"稳"有效，对"好"效果有限甚至有害
- 数据断层：通用朗读数据充分，但情感/方言/对话场景数据严重不足

---

### 2.5 表现力 vs 稳定性：技术层面的结构性纠缠

上面的行业梳理揭示了一个贯穿所有团队的共同主题：**表现力和稳定性在当前技术栈中是结构性矛盾，不是调参能解决的**。这个矛盾的根源在 Tokenizer/Codec 层——TTS 系统的表现力上限，不是由模型大小决定的，而是由语音被编码成什么样的 token 决定的。

理解这个矛盾，需要看三个阶段的 Tokenizer 演进，以及每次转向背后的因果链。

#### 阶段一：RVQ 全量建模（2022-2023）

起点很自然：文本 LLM 的 next-token prediction 已经证明强大，语音也能离散化成 token → 复用 GPT 范式。SoundStream/EnCodec 提供离散表征，AudioLM/VALL-E 证明走得通。

这个阶段没人纠结"信息量"问题——所有信息都编进 RVQ，全量交给 AR 建模。

但实际部署后问题集中爆发：幻觉、跳词、不稳定、prompt 敏感。根因是：RVQ 码本层级多导致序列太长，帧率偏高导致冗余信息多，全量信息噪声大导致 AR 自由度太高——管不住。

CosyVoice 给出了数据：EnCodec token 做 TTS 的 WER 18.70%，而监督语义 token WER 仅 3.93%。MaskGCT 也指出："AR 系统（VALL-E, VoiceCraft）逐 token 生成导致鲁棒性差、推理速度慢"。

#### 阶段二：语义 token 分流，稳定优先（2024-2025）

诊断清楚后，行业收敛到一个方案：AR 只管语义 → Flow/DiT 渲染声学。CosyVoice、Seed-TTS、GLM-4-Voice、MaskGCT 都走了这条路。稳定性大幅提升，大规模上线成为可能。

但代价同样清晰——**表现力被压缩了**。情感、副语言、音量这些信息在语义/声学拆分时丢失，下游模块补不回来。

NaturalSpeech3 诊断了根因："语音信号内在地纠缠了多种属性（内容、韵律、音色、声学细节），RVQ 多层仍混合所有属性信息，无法独立控制"。SAC 进一步验证了干扰的严重程度：将语义和声学融合在一起的 codec（如 X-Codec、SpeechTokenizer）中，两个优化目标互相干扰——SemantiCodec 的语义流重建 WER 高达 30.67，而 SAC 的冻结语义流仅 3.99。MagiCodec 从另一个角度指出："neural audio codec 的重建质量和下游可建模性存在优化冲突"。

这就是"稳了，但不够活"的技术根因——不是模型不够大，而是 token 层就把信息丢了。

#### 阶段三：重新拥抱全量信息，但换了建模方式（2025-2026）

上一阶段暴露了两个结构性缺陷：分阶段训练复杂（误差累积）、codec 表征有信息上限。加上 VoiceAgent、播客、Omni 等新场景对表现力的要求，推动了第三波探索。

DiTAR 明确批评两阶段级联："离散 token AR + diffusion 精修导致误差累积，限制了 LM 的 scaling 潜力"，转向 patch 级连续向量预测。DiSTAR 则批评了 DiTAR 的连续路线："连续 latent 对分布偏移敏感，高维优化脆弱"，回到离散 RVQ 但用 masked diffusion 替代 AR——0.3B 模型 WER 1.66% 超越人类 1.80%。WavTTS 最激进，批评所有中间表征都有信息损失，直接在波形空间用 DiT 生成，但发现了信号-噪声方差失配问题。dots.tts 走连续 AR 路线，用 semantic encoder 做 4x 下采样信息瓶颈，让 LLM 只看语义摘要不看 raw latent，避免声学细节干扰全局语义规划。

路线还没收敛，各家在为不同问题选不同解法。

#### 三次转向背后的逻辑

| 转向 | 面临的问题 | 解决思路 | 牺牲了什么 |
|------|-----------|----------|-----------|
| RVQ 全量 → 语义 token 分流 | AR 建模 RVQ 多层太难，幻觉严重 | 只让 AR 建模"容易"的语义层 | 表现力天花板 |
| 语义 token → 重新拥抱全量 | 语义 token 丢了太多声学细节 | 新范式建模全量信息 | 方案复杂度、训练难度 |

**核心洞察：不是"codec 选什么"在变，而是"谁来承担建模难度"在变。** 第一阶段让 AR 扛全部，扛不住；第二阶段让 codec 做预处理减轻 AR 负担，但压缩太多；第三阶段让新的生成范式（diffusion/mask/patch）替代 AR 来扛全量信息。

#### 当前 Codec 设计的快稳好权衡

不同 Codec 在"快、稳、好"三角中选了不同位置：

| Codec | 权衡选择 | 为 TTS 解决了什么 |
|-------|---------|-----------------|
| **CosyVoice FSQ** | 稳 > 好 > 快 | 单码本+监督，LM 容易学。FSQ 利用率 100% vs VQ 23% |
| **GLM 175bps** | 快 ≈ 稳 > 好 | 12.5Hz 极低帧率，序列超短 |
| **MagiCodec** | 稳 ≈ 好 > 快 | Gaussian 注入使 token 分布接近自然语言。mask 0%→30% TTS WER 从 5.51 降到 3.30 |
| **SAC（X-LANCE/上交）** | 好 > 稳 > 快 | 双流分离，语义流 SIM 仅 0.17（几乎零说话人泄漏） |
| **VARSTok（USTC+阿里通义）** | 快 ≈ 好 | 变长 token，自适应信息密度 |
| **WavTokenizer** | 好 > 稳 | 单码本极致重建，40 token/s |
| **无 codec（WavTTS）** | 好 >> 快/稳 | 零信息损失，但方差失配问题未彻底解决 |

**结论：表现力和稳定性的矛盾不是某个模型的 bug，而是 Tokenizer 设计层面的结构性 trade-off。** 突破这个天花板，需要的不是更大的模型，而是更好的语音表示方案——这也是为什么 Codec/Tokenizer 方向10+种路线并行、共识完全没有形成的根本原因。

---

### 3. 结合行业和我们自己，未来初步思考可以做的

回到我们自身。三个阶段走下来，我们拿到了：稳定性优势（SIM 0.862, CER 0.87%）、数据效率（200K小时超百万级对手）、多维控制基础（情感88.1%、副语言85.1%、14种方言）、完整工程链路。

但应用侧暴露的问题也很清晰：表现力天花板受codec限制、上下文韵律能力为零、评价体系缺失只能挑demo。

**结合行业趋势和我们的基础：**

**1. 将目前TTS的instruct能力在VoiceAgent上尝试。（T0）（选取表现力高，稳定的 情感&副语言&方言）**

行业参照：ElevenLabs Conversational AI、阶跃Realtime模式、通义Fun-Audio-Chat都在做。趋势是从显式instruct转向隐式context-aware——小红书FireRedTTS-2证明上下文隐式推断虽准确率略低（83-93% vs 97-100%），但更自然。

做法：先用显式instruct验证能力天花板，同步探索context-aware。同时建对话级评测集（情感准确度+副语言时机+方言切换+多轮一致性）。

**2. TTS instruct能力的拓展。（T0）**

   **2.1 控制标签、控制方式的拓展。（标签拓展，自然语言，context）**

   行业实际控制能力分层：
   - L1 全局标签控制 ← 大多数系统当前水平，包括我们
   - L2 自然语言指令 / Voice Design ← 已成为多家标配能力：Qwen3-TTS（thinking pattern提升instruction following，APS 85.2）、CosyVoice3（5000小时instruct数据，100+种风格）、dots.tts（caption-style描述）、OmniVoice（属性指令控制音色）、天工MoE-TTS等
   - L3 上下文隐式控制 ← ElevenLabs、Sesame CSM、FireRedTTS-2（interleaved format + 对话微调实现从上下文推断情感）
   - L4 词级时变控制 ← 学术前沿，几乎无人系统做

   判断：L2自然语言指令已不是差异化——多家系统都已支持Voice Design和基础instruct。真正的差异化在L3（上下文隐式）和L4（词级时变），以及**多层级联动**（L1全局 → L3段落级context → L4词级fine-grained 三层联动），这是目前没有团队系统做的。

   **2.2 强化学习在Instruct的应用。**

   行业现状：Post-training已成标配（5+家），但每家都发现不同问题——字节发现reward hacking，阿里发现DiffRO导致自然度下降，GRPO-TTS发现CER降83%但MOS不变。**核心判断：RL对"稳"有效，对"好"效果有限甚至有害。**

   我们的差异化：做"属性对齐RL"——奖励信号不是通用质量偏好，而是instruct-following（TTS输出是否精确遵循了控制指令）。NLP的IF-eval → RLHF在语音领域没人做过。需要：Instruct-Following Reward Model + 属性级偏好数据 + 防reward hacking机制。

**3. 下一代TTS架构升级。（T0）**

   **3.1 效果优先。**

   核心认识：表现力天花板由codec/token决定，不是由模型大小决定（CosyVoice3发现升到1.5B瓶颈在post-training数据不在模型大小；Seed-TTS称tokenizer为"整个系统的瓶颈"）。

   **3.1.1 声学&语义的权衡。tokenizer方向。**

   | 方案 | 代表 | 适合我们 |
   |------|------|:---:|
   | 双流（语义+声学分离） | SAC（X-LANCE/上交，ACL 2026） | ★★★★★ |
   | 多码本+非AR | DiSTAR（字节，0.3B WER < 人类） | ★★★★ |
   | 变长token | VARSTok（USTC+阿里通义，AAAI Oral） | ★★★★ |
   | 连续表征 | DiTAR（字节，ICML） | ★★★ |
   | 无codec | WavTTS（X-LANCE/字节） | ★★★（高风险） |

   路线：短期保持FSQ + 模型升2B → 中期探索双流codec → 长期关注变长token/无codec。

   **3.1.2 更大的模型尺寸+更多的数据。** 2B，7B等常规尺寸。多阶段训练：低质量数据Pretrain + 精选数据ContinueTrain + Post-training（RL/DPO）。利用LLM权重初始化（CosyVoice2验证Qwen初始化CER降18.5%）。

   **3.2 效率。**

   **3.2.1 更小的模型、更低的帧率。** 参考：ZipVoice 123M超越1B+模型；GLM-4-Voice 12.5Hz极低帧率。

   **3.2.2 原生文本流式模型。** 区别于当前主流方案（输入完整文本 → 流式输出wav），这里指的是**文本输入也是流式的**——边收到LLM生成的文本token，边合成语音输出。这是VoiceAgent场景的刚需：LLM逐token输出回复，TTS不需要等完整句子，实现真正的端到端流式。目前CosyVoice2 chunk-aware FM、Qwen3-TTS等解决的主要是输出端流式（输入仍需完整或分句文本），真正的文本流式输入+语音流式输出的原生方案仍在探索中。

   **3.2.3 更新的模型架构。** MoE（天工MoE-TTS首个验证可行）；SSM/Mamba（Cartesia验证替代Transformer可行，线性复杂度适合端侧）。

**4. 探索方向：**

   **4.1 音效、声音事件、语音联合统一建模生成。** 参考蚂蚁Ming-UniAudio、通义ThinkSound。差异化角度：不做泛音频生成，做"语音+音效联合控制"——播客/有声书/视频配音的刚需。

   **4.2 TTS播客（多人对话合成）。** 参考：小红书FireRedTTS-2做到3分钟/4说话人；京东JoyVoice长上下文；NotebookLM已验证产品形态。与Instruct方向高度协同。

   **4.3 TTS edit（合成数据pipeline，VC音色编辑，内容编辑，情感编辑）。** 最大价值是**作为数据引擎**——用edit能力低成本构造标注数据反哺训练。阶跃EditX证明3B模型即可做属性编辑。TTS Edit → 数据 → 训练 → 更好的Edit → 更多数据——数据飞轮。

---

### 需要资源：

**1. 数据积累，更丰富的标注维度。**

- 基础层：说话人ID、时间戳、文本对齐
- 属性层：情感（连续值）、语速、音高轮廓、能量、音频属性
- 细粒度：word-level情感、副语言事件位置和类型
- 偏好层：成对偏好（A>B）、多维度打分（自然度/表现力/一致性）

优先数据类型：多维度标注数据（P0）、人类偏好对（P0）、对话数据（P1）、大规模预训练数据（P1）。

**2. 工具积累，更多的标注工具。**

| 工具 | 优先级 |
|------|--------|
| 自动MOS评估（SpeechJudge/UTMOS类） | P0 |
| Instruct遵循度评估（自建） | P0 |
| 强制对齐+韵律提取管线 | P0 |
| 偏好标注平台（A/B对比） | P1 |
| TTS Edit工具链 | P1 |

**3. 训练资源，卡。**

建议分配：架构训练（Pretrain/CT/流式）60% + RL实验（Reward/GRPO/DPO）25% + 探索（Edit/播客）15%。

---

> **一句话总结：我们已经拿到了"稳"的先发优势，行业已过"稳"是核心瓶颈的阶段。下一步竞争焦点是"可控的好"——不是随机地好听，而是精确地好听。策略是：升级token提升表现力天花板，建立"属性评价→标注→强化"闭环，做到可控的表现力。**
