# jev-mini

**Jev 的三个核心宣称，在你自己的 GPU 上跑一遍。**

```bash
python quickstart.py
```

一条命令，自动下载一个 0.5B 模型（~1GB，走 ModelScope，国内直连），
然后把「System One Model」的三个卖点逐条放到台面上验证。

在一块 **RTX 3060 Laptop（6GB，还开着浏览器）** 上跑完全程约 2 分钟。

---

## 这个仓库为什么存在

Jev 发布后被讨论得很热，但社区的质疑集中在三点：并行结构化输出不是新东西，
"0% 幻觉"偷换了 hallucination 的定义，而唯一真正重要的 calibration
**至今没有公开证据，也没有独立测试**。

那就自己测。本仓库不试图复刻 Jev，而是提供一把**你能跑、能读、能改的尺子**：
用同一个本地模型、同一批问题，把"受限打分"和"自回归生成 JSON"放在一起比，
并且把大家都在谈、却没人量化的那个指标——概率校准——真正算出来。

**结论先放这里：前两个宣称是真的但被过度包装，第三个宣称没那么容易兑现。**

---

## 机制：一句话

> 不要让模型「写」出答案，让它给你允许的答案「打分」。

生成式模型回答选择题，要一个 token 一个 token 地把词拼出来。
这里一个 token 都不生成：把每个候选项接到 prompt 后面，读出模型赋予它的
对数概率，最高的就是决策，归一化后就是置信度。

三个结果随之而来，正好对应 Jev 宣传的三件事：

| Jev 的说法 | 这里的实现 | 实测结论 |
|---|---|---|
| Typed probabilistic decisions | 输出是 list 下标 | ✅ 成立 |
| Zero hallucinations | 结构上不可能越界 | ⚠️ 成立，但只是类型保证 |
| Calibrated confidence (RLCD) | 概率即返回值，可直接测 | ❌ 小模型上很差，且**模型越大不一定越好** |

---

## 实测数据（全部来自本机真实运行）

硬件：RTX 3060 Laptop 6GB · torch 2.9.1+cu126 · Python 3.14 · Windows
数据：18 条客服工单 + 12 条评论，人工标注，**标注先于任何模型运行**，
并按 easy / medium / hard 分层——刻意放入歧义样本，因为
**只有在该犹豫的地方，校准才有意义**。

原始输出见 [`results_0.5b.json`](results_0.5b.json) 和 [`results_1.5b.json`](results_1.5b.json)。

### 1. 受限打分 vs 自回归生成 JSON（同模型、同问题）

以 0.5B 为例：

| | 受限打分 | 生成 JSON |
|---|---|---|
| 中位延迟 | **135 ms** | 6206 ms |
| 生成 token | **0** | 96 |
| 类型错误 | **0 / 18** | 1 / 18 |
| 分类准确率 | 0.611 | 0.611 |

**约 46x 加速，准确率完全相同**（1.5B 上为 24x：模型越大，生成侧的相对劣势
被自身前向开销摊薄了一些）。提速全部来自砍掉解码，不是模型变聪明了。

有一个附带结果值得注意：**1.5B 上生成式 baseline 的准确率反而掉到 0.444，
低于受限打分的 0.611**。同样的权重、同样的问题，仅仅因为答案要先被"写"出来
再解析，就损失了准确率——模型把 token 花在了复述格式而不是判断上。

### 2. 并行字段：K 个问题真的约等于 1 次前向吗

| 字段数 | 中位延迟 | 打分候选数 | 前向次数 |
|---|---|---|---|
| 1 | 129.0 ms | 5 | 2 |
| 3 | 123.6 ms | 12 | 2 |
| 8 | **139.4 ms** | 26 | 2 |

**8 个字段只比 1 个字段慢 1.08x**（1.5B 上为 1.13x）。这条宣称站得住：
共享前缀只算一次，KV cache 复用，所有候选项在一个 batch 里打分。

### 3. 校准：那个没人给证据的指标

| 任务 | 模型 | 准确率 | 平均置信度 | ECE | 判断 |
|---|---|---|---|---|---|
| 工单分类 | 0.5B | 0.611 | 0.645 | 0.100 | 轻微过度自信 |
| 是否紧急 | 0.5B | 0.722 | 0.593 | 0.129 | **欠**自信 |
| 情感分析 | 0.5B | 0.333 | 0.836 | **0.624** | 灾难性过度自信 |
| 工单分类 | 1.5B | 0.611 | 0.810 | 0.199 | 过度自信 |
| 是否紧急 | 1.5B | 0.667 | 0.878 | **0.212** | 过度自信 |
| 情感分析 | 1.5B | 0.333 | 0.779 | 0.446 | 过度自信 |

**最值得注意的一行**：情感分析里，0.5B 有 10/12 个样本置信度落在 0.8 以上，
而这批样本的实际正确率只有 **0.2**。模型在反讽和微妙语气上错得非常自信——
`"Well, it certainly is a product that exists."` 这类句子正是它翻车的地方。

**反直觉发现：模型变大，准确率没涨，校准反而更糟。**
`ticket_urgent` 上 0.5B 是欠自信（-0.129），1.5B 翻转成过度自信（+0.212）。
体现在运维上：阈值设 0.50 时，1.5B 放过了 **38.9%** 的错误，0.5B 只有 22.2%。

> 「模型更大，置信度更可信」——实测不成立。

### 4. 温度标定：一个标量能补救多少

| 模型 | 5 折 CV 拟合 T | 基线 ECE | 标定后 ECE | 结果 |
|---|---|---|---|---|
| 0.5B | 1.443 ± 0.194 | 0.158 | 0.174 | **更差** |
| 1.5B | 1.880 ± 0.152 | 0.238 | **0.122** | 改善 49% |

这里有两个诚实的结论：

1. **1.5B 上，一个拟合出来的标量就让 ECE 降了近一半。** 如果 RLCD 的收益
   能被温度标定追平，那它就不值一个新训练范式的名头——当然，这需要
   TypeSafe 公开数据才能真正比较。
2. **0.5B 上温度标定让结果变差了，我把它原样留在这里。** n=18 实在太小，
   拟合出的 T 主要是切分噪声（单一切分甚至给出 T=0.714，方向都是反的）。
   所以代码提供 `fit_temperature_cv()`，并把折间标准差一起打印出来——
   **方差大本身就是"数据不足以下结论"的信号**。这是负面结果，不是 bug。

### 5. 路由表：校准到底买到了什么（0.5B，工单分类）

| 阈值 | 自动处理占比 | 自动处理准确率 | 放过的错误 |
|---|---|---|---|
| ≥0.50 | 77.8% | 0.714 | 22.2% |
| ≥0.70 | 33.3% | 0.833 | 5.6% |
| ≥0.80 | 22.2% | **1.000** | **0%** |

这张表才是校准的实际用途：它是模型置信度和你的自动化策略之间的合同。

---

## 一个副产品发现：标签措辞会显著改变概率

同一个问题，只换 yes/no 的字面写法，极端样本上的区分度差了近一倍：

| 标签 | 分离度 |
|---|---|
| `yes` / `no` | **+0.594** |
| `Yes` / `No` | +0.566 |
| `true` / `false` | +0.515 |
| `urgent` / `not urgent` | +0.316 |

如果你要把这类方法用到生产里，**标签措辞是一个需要调的超参**，不是随手写的。

---

## 用法

```python
from jevmini import JevMini, Schema, Choice, Noul, Score

schema = Schema(fields=[
    Choice(name="category", options=["billing", "technical", "account"],
           question="Which department should handle this?"),
    Noul(name="urgent",   question="Is this urgent?"),
    Score(name="severity", low=1, high=5, question="How severe?"),
])

d = engine.decide("I was charged twice for order #4471.", schema)

d.value("category")       # 'billing'  —— 一定是三个选项之一
d.confidence("category")  # 0.718
d["category"]["distribution"]
# {'billing': 0.718, 'account': 0.28, 'technical': 0.002, ...}

if d.confidence("category") > 0.8:
    auto_route(d.value("category"))
else:
    escalate_to_human()
```

`Score` 除了 argmax 还返回**期望值**：一条在 2 星和 4 星之间摇摆的评论
应该读作 3，而不是险胜的那个桶。

---

## 安装

```bash
git clone https://github.com/RichardoMrMu/jev-mini
cd jev-mini
python -m venv .venv && .venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers modelscope

python quickstart.py
```

**国内网络**：模型走 ModelScope，不需要 HuggingFace 访问。
pip 可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

**完整报告**：

```bash
python scripts/benchmark.py --model <本地模型路径> --out results.json
```

### 显存很小或内存吃紧

本仓库就是在这种环境下开发的（6GB 显卡 + 已占用 2GB + 16GB 内存）。
`quickstart.py` 已默认设置 `TRANSFORMERS_NO_CACHING_ALLOCATOR_WARMUP=1`，
避免 transformers 按整模型大小预留显存而在共享显卡上直接失败。

实测峰值显存：**0.5B → 1579 MB，1.5B → 3598 MB**。

若遇到 `OSError 1455 页面文件太小`，那是系统提交内存耗尽而非显存问题。
关掉 WSL（`wsl --shutdown`）或其他大内存进程即可。

---

## 这个仓库**没有**证明什么

必须说清楚，否则就是在用另一种方式夸大：

- **这不是 Jev。** Jev 的权重、架构、参数量都没公开，这里用的是通用小模型 +
  受限打分。速度和校准数字只代表本方法在本机的表现。
- **不能用来反驳 TypeSafe 的数字。** 他们的 193x / 444x 是特定工作流下的
  最大差距；这里的 24–46x 是另一组任务、另一个模型、另一块卡。
- **数据集很小**（18 + 12 条）。ECE 在这个量级上误差不小，所以本仓库
  同时输出折间方差，而不是只给一个好看的数。扩大数据集是首要的 TODO。
- **没有实现 RLCD。** RLCD 的训练方法和 reward function 都未公开，无法复现。
  这里提供的是**测量它的工具**，不是它的实现。

能证明的是：受限打分带来的类型安全和并行加速是真实且易得的；
而校准——真正难的那部分——在一个没有专门优化过的模型上确实很差，
并且**不会因为把模型换大就自动变好**。

---

## 目录

```
jevmini/
  core.py          引擎：受限打分、KV cache 复用、Choice/Score/Noul
  calibration.py   ECE / MCE / Brier / NLL、温度标定（含交叉验证）、路由表
  datasets.py      人工标注数据，按难度分层
scripts/
  benchmark.py     完整五节报告
quickstart.py      一键体验
```

代码有详细注释，尤其是几个容易踩的坑：
**为什么只对标签 token 打分**（把问题 stem 也算进去会把分布压平，
这是开发过程中真实踩过的 bug——修复前 billing 只有 0.298，修复后 0.718），
以及**为什么必须做长度归一化**。

## License

MIT
