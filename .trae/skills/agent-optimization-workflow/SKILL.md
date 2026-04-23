---
name: "agent-optimization-workflow"
description: "GUI Agent优化工作流规范。当需要测试、调试、优化agent时调用，包含测试运行、prompt评估、错误分析、软性改进等完整流程。"
---

# GUI Agent 优化工作流规范

## 一、项目学习路径

### 1.1 核心文件结构
```
code-for-student/
├── agent.py              # 主Agent：act循环、workflow规划、状态管理、区域缓存
├── agent_base.py         # 基类：常量定义(ACTION_CLICK等)、数据类、API配置
├── test_runner.py        # 测试入口：离线/在线测试执行
├── utils/
│   ├── agent_prompt.py   # 主模型Prompt构建（USER prompt核心）
│   ├── agent_click_prompt.py  # click-localizer子模型Prompt
│   ├── agent_actions.py  # 动作后处理（约束检查、坐标修正）
│   ├── agent_regions.py  # 区域定义、坐标 snapping
│   ├── agent_rules.py    # 指令分解正则、子目标提取
│   └── task_playbook.py  # 任务流程先验记忆（外卖/地图/视频/购物）
└── test_data/offline/
    └── step_xxx_0001/
        ├── ref.json      # 测试用例定义（步骤+期望动作+bbox）
        └── 0.png~N.png   # 每步截图
```

### 1.2 数据流
```
指令 → agent.act() → 主模型(判断动作) → CLICK? → click-localizer(精确定位)
                                    → 后处理(agent_actions约束)
                                    → 输出 AgentOutput(action, parameters)
```

## 二、线上API配置

### 2.1 环境变量配置

**调试阶段（本地开发）必须设置：**

```bash
# 必须设置：你的 API 密钥
export VLM_API_KEY="ark-32f7c908-cdef-4486-8b15-63a41ff6daaf-fd0eb"

# 可选设置：自定义 API 地址（默认使用下方 DEFAULT_API_URL）
# export DEBUG_API_URL="https://your-custom-api.com/v3"

# 可选设置：自定义模型ID（默认使用下方 DEFAULT_MODEL_ID）
# export DEBUG_MODEL_ID="your-model-id"
```

### 2.2 固定配置（不可修改）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `DEFAULT_API_URL` | `https://ark.cn-beijing.volces.com/api/v3` | API地址，提交时强制使用 |
| `DEFAULT_MODEL_ID` | `doubao-seed-1-6-vision-250815` | 模型ID，提交时强制使用 |

### 2.3 提交阶段 vs 调试阶段

| 阶段 | 触发条件 | API URL | Model ID | API Key |
|------|----------|---------|----------|---------|
| **调试阶段** | 默认（不设EVAL_MODE） | DEBUG_API_URL 或 DEFAULT | DEBUG_MODEL_ID 或 DEFAULT | **VLM_API_KEY** |
| **提交阶段** | `EVAL_MODE=production` | EVAL_API_URL（主办方设置） | EVAL_MODEL_ID（主办方设置） | EVAL_API_KEY（主办方设置） |

> ⚠️ 提交时所有 DEBUG_* 环境变量被忽略，任何篡改配置的行为将被检测并终止评测

### 2.4 配置验证

```bash
# 启动前检查环境变量是否设置
echo "VLM_API_KEY: ${VLM_API_KEY:0:10}..."  # 只显示前10位保护密钥

# 运行时会自动打印使用的配置
python test_runner.py --data_dir test_data/offline/step_meituan_onekey_0001 2>&1 | grep -i "model\|api\|debug"
```

## 三、测试运行方式

### 3.1 所有可用测试用例

| 用例目录 | 应用类型 | 指令关键词 |
|----------|----------|-----------|
| `step_meituan_onekey_0001` | 🍔 美团外卖 | 购买窑村干锅猪蹄店铺的干锅排骨 |
| `step_baidumap_onekey_0008` | 🗺️ 百度地图 | 导航相关 |
| `step_baidumap_onekey_0010` | 🗺️ 百度地图 | 导航相关 |
| `step_aiqiyi_onekey_0011` | 🎬 爱奇艺 | 视频播放/搜索 |
| `step_bilibili_onekey_0008` | 📺 B站 | 视频播放/搜索 |
| `step_douyin_onekey_0008` | 🎵 抖音 | 视频搜索/播放 |
| `step_kuaishou_onekey_0003` | 📸 快手 | 视频搜索/播放 |
| `step_mangguo_onekey_0008` | 🥭 芒果TV | 视频播放 |
| `step_tengxunshipin_onekey_0005` | 🎬 腾讯视频 | 视频播放/搜索 |
| `step_ximalaya_onekey_0001` | 🎧 喜马拉雅 | 音频搜索/播放 |
| `step_quonekey_0030` | 🛒 趣购 | 购物下单 |

### 3.2 单个用例测试（最常用）

```bash
# 基本命令
python test_runner.py --data_dir test_data/offline/step_meituan_onekey_0001

# 过滤关键输出（推荐日常使用）
python test_runner.py -d test_data/offline/step_xxx | grep -E "(Agent Output|Score|准确|测试完成|WORKFLOW)"

# 保存完整日志用于深度分析
python test_runner.py -d test_data/offline/step_xxx 2>&1 | tee /tmp/test_log.txt
```

### 3.3 全量测试

```bash
# 运行所有测试用例
python test_runner.py --data_dir ./test_data/offline

# 打分模式（验证失败时终止执行，用于正式评估）
python test_runner.py -d ./test_data/offline --no_debug_test
```

### 3.4 命令参数说明

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | `-d` | `./test_data/offline` | 测试数据目录（单个用例或目录） |
| `--output_dir` | `-o` | `./output` | 结果输出目录 |
| `--no_debug_test` | — | False | 打分模式：失败时终止 |

### 3.5 输出解读

```
2026-04-23 05:19:20 - INFO - Agent Output: action=OPEN, params={'app_name': '美团'}     ← 每步输出
2026-04-23 05:19:34 - INFO - Agent Output: action=CLICK, params={'point': [106, 186]}
...
[No. 1: step_meituan_onekey_0001] Score: 9/14 = 0.64                              ← 步骤准确率
[WORKFLOW] 规划完成: ['打开美团APP', '点击底部导航栏...', ...]                     ← 工作流
测试完成!
总用例数: 11
通过用例数: 0                                                                      ← 全部正确才算通过
用例准确率: 0.00%                                                                   ← 用例级别准确率
步骤准确率: 64.29%                                                                  ← ⭐ 关注这个！
```

**关键指标：步骤准确率** = 正确步数 / 总步数

## 四、线上测试前必须本地验证

### 4.1 Prompt 验证脚本（dry-run）

在修改 prompt 相关代码后，**必须先跑本地验证确认格式正确**：

```python
from utils.agent_prompt import GroundedActionPrompt
from utils.task_playbook import classify_task

instruction = '去美团外卖购买xxx'
pb = classify_task(instruction, '美团')

tests = [
    ('Step 4 (激活后应TYPE)', [
        {'action':'OPEN','parameters':{'app_name':'美团'}},
        {'action':'CLICK','parameters':{'point':[106,188]}},
        {'action':'CLICK','parameters':{'point':[387,115]}},
    ]),
    ('Step 5 (TYPE后应确认)', [
        {'action':'OPEN','parameters':{'app_name':'美团'}},
        {'action':'CLICK','parameters':{'point':[106,188]}},
        {'action':'CLICK','parameters':{'point':[387,115]}},
        {'action':'TYPE','parameters':{'text':'窑村干锅猪蹄（科技大学店）'}},
    ]),
]

for name, history in tests:
    prompt = GroundedActionPrompt.get_user_prompt(
        instruction=instruction, state={'app_name':'美团'},
        history=history, playbook_info=pb
    )
    print(f'=== {name} ({len(prompt)}字) ===')
    print(prompt)
    print()
```

### 4.2 语法检查

```bash
# 快速检查代码语法
python3 -c "from agent import Agent; print('Agent OK')"
python3 -c "from utils.agent_actions import ActionProcessor; print('Actions OK')"
python3 -c "from utils.agent_prompt import GroundedActionPrompt; print('Prompt OK')"
```

### 4.3 验证 Checklist

每步 prompt 检查：
- [ ] 字数是否合理（目标：200~350字/步）
- [ ] "已完成"是否显示具体内容（TYPE要有文本，激活要有✓标记）
- [ ] "当前"是否只显示一步（不要显示下一步）
- [ ] 状态转换提示是否准确（"已输入→需确认"）
- [ ] 无冗余信息（去掉无用的坐标、重复的规则）

## 五、错误分析方法论

### 5.1 逐步对比表格（每次测试必做）

| 步 | 模型输出 | 期望动作 | 判定 | 偏差分析 | 改进方向 |
|---|---|---|---|---|---|
| 1 | OPEN 美团 | OPEN | ✅ | — | — |
| 4 | TYPE 店名 | CLICK或TYPE | ✅ | — | — |
| 5 | TYPE 店名 *(重复)* | CLICK搜索 | ❌ | 动作重复 | 显示已输入文本 |

### 5.2 错误分类速查表

| 类型 | 表现 | 常见原因 | 解决方向 |
|------|------|----------|----------|
| **动作错误** | TYPE vs CLICK 搞混 | 输入协议理解不清 | 强化状态转换提示 |
| **重复操作** | 连续相同动作 | 状态传递不足 | 显示具体内容+✓标记 |
| **多余步骤** | 不该有的操作 | 缺少"完成"判断 | COMPLETE规则引导 |
| **坐标偏离** | 点错位置 | region hint干扰 | 移除硬编码+降snapping |

### 5.3 坐标偏差计算

```python
model_point = [x, y]                    # 模型输出坐标
expect_bbox = [xmin, ymin, xmax, ymax]  # 期望bbox（从ref.json获取）
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2
dx = model_point[0] - center_x           # X轴偏差（正值偏右）
dy = model_point[1] - center_y           # Y轴偏差（正值偏下）
print(f"X偏差: {dx:+.0f}, Y偏差: {dy:+.0f}")
# 判定标准: |dx|<50 且 |dy|<50 通常可接受
```

## 六、改进原则（核心！）

### 6.1 软措施优先级

```
① Prompt 引导（告诉模型状态和期望）       ← 最优，首选
② 重复检测 + 让模型重判(top2)            ← 兜底机制
③ 硬编码拦截                             ← 最后手段，尽量避免
```

### 6.2 ❌ 不要硬编码的东西

| 禁止项 | 示例 | 替代方案 |
|--------|------|----------|
| 固定坐标 | `return {"point": [900, 70]}` | 让click-localizer决定 |
| 强制动作转换 | `TYPE重复→改为CLICK` | 构造新prompt让模型重判 |
| 硬编码region匹配 | `if region == "TOP": ...` | 语义描述引导 |

### 6.3 ✅ 应该做的软措施

- 在 prompt 中明确告知"上一步做了什么"（带具体内容）
- 在 prompt 中明确告知"当前应该做什么"（只说当前步骤）
- 状态标记：`✓已就绪` `✓完成` `(具体文本内容)`
- 检测到可疑重复时，构造新 prompt 让模型重新判断（top2选择）

## 七、Prompt 设计规范

### 7.1 信息密度原则

```
每步 prompt 目标字数: 200~350字

高价值信息（必须有）:
  ✓ 任务指令              (~30字)  核心目标
  ✓ 应用名                (~6字)
  ✓ 已完成历史(语义化)     (~60~100字)  带状态标记+具体内容
  ✓ 当前步骤              (~15字)  只显示当前
  ✓ 状态转换提示          (~20字)  上一步→当前的关系
  ✓ 核心提示(tips)        (~80字)  高价值规则

低价值/冗余信息（避免）:
  ✗ 完整历史坐标          → 只在区域缓存中按需加载(CLICK时才加载)
  ✗ 多步流程预览          → 只显示当前步骤，不展示后续
  ✗ 重复的硬性规则        → 合并到tips，避免啰嗦
  ✗ 无用的占位符          → "-" 也尽量省略
```

### 7.2 历史记录格式

```python
# ✅ 好：语义化 + 状态标记 + 具体内容
"已完成: 打开目标应用, 进入外卖频道, 【激活】点击搜索框(✓已就绪), 【输入】TYPE 店铺名称(窑村干锅猪蹄..)"

# ❌ 差：只有动作名称，没有状态信息
"已完成: 打开应用, 点击, 点击, TYPE"
```

### 7.3 流程显示格式

```python
# ✅ 好：只显示当前步骤 + 状态转换提示
"当前: 【输入】TYPE 店铺名称"
"输入框已激活，直接TYPE输入内容"

# ❌ 差：显示多步流程（分散注意力）
"流程: 3.激活搜索框 | → 4.TYPE店名 | 5.确认搜索 | 6.进店"
```

### 7.4 区域缓存（按需加载）

```
只在以下情况加载"已识别"区域记忆:
  ✓ 上一步是 CLICK（需要参考之前点过的位置）
  ✗ 上一步是 TYPE/OPEN/COMPLETE（不需要参考位置）

格式: "已识别: TOP_SEARCH_BOX≈[387,115], BOTTOM_BAR≈[106,188]"
```

## 八、常见问题与解决方案

### 8.1 重复 TYPE
**现象**: 连续两步输出相同的 TYPE 文本
**根因**: 模型被指令关键词绑架，忽略截图变化和历史记录
**解决**:
1. prompt 中显示具体输入文本：`【输入】TYPE 店铺名称(窑村干锅猪蹄..)`
2. 当前步骤改为确认动作：`当前: 【确认】点击搜索按钮`
3. 状态转换提示：`已输入文本，需点击搜索确认`

### 8.2 重复 CLICK 同一区域
**现象**: 激活输入框后又多点一次（如点[370,70]两次）
**根因**: 不信任输入框已激活状态
**解决**:
1. 历史记录加 `✓已就绪` 标记
2. 状态转换提示："输入框已激活，直接TYPE输入内容"
3. 区域缓存记录已识别的UI元素

### 8.3 坐标偏离
**现象**: 点击位置偏离目标 bbox 较远
**根因**: REGION_HINTS 硬编码坐标 / snapping 权重过高(40%)
**解决**:
1. 移除硬编码 region 坐标提示
2. 降低 snapping 权重（40% → 10%）
3. 使用语义描述代替精确坐标

### 8.4 多余步骤
**现象**: 任务完成后还继续操作（如选规格后继续点击）
**根因**: 缺少"完成推进"意识
**解决**:
1. 硬性规则8：每一步只做一件事，不要执行多余操作
2. 状态转换提示引导下一步

## 九、代码修改 Checklist

每次修改后**依次检查**：

```bash
# 1. 语法检查
python3 -c "from agent import Agent; print('OK')"

# 2. Prompt 格式检查
python3 -c "
from utils.agent_prompt import GroundedActionPrompt
from utils.task_playbook import classify_task
pb = classify_task('测试', '美团')
p = GroundedActionPrompt.get_user_prompt('测试', {'app_name':'美团'}, [], pb)
print(f'Prompt OK, {len(p)}字')
"

# 3. Dry-run 验证各步骤 prompt 内容（见第四节）

# 4. 跑离线测试看分数变化
python test_runner.py -d test_data/offline/step_meituan_onekey_0001 2>&1 | grep -E "(Score|准确)"

# 5. 对比修改前后逐步输出，定位变化点
```

## 十、关键经验总结

1. **Prompt > 硬编码**: 能用 prompt 解决的就不要写死逻辑
2. **状态 > 动作**: 告诉模型"当前是什么状态"比"做了什么动作"更有用
3. **具体 > 抽象**: 显示实际输入文本比显示"已输入"更有效
4. **当前 > 未来**: 只告诉模型当前该干嘛，不要提前展示下一步
5. **密度 > 数量**: 300字的高密度信息 > 800字的低密度信息
6. **软措施 > 硬拦截**: 先改 prompt，再考虑重判机制，最后才硬拦截
7. **先本地后线上**: 修改后先 dry-run 验证 prompt，再跑离线测试，最后考虑全量
