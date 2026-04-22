# 第9轮：基于当前远程仓库的下一版改进说明

## 基线
本轮修改基于当前远程仓库 `submission/src/agent.py` 最新版本，不回退到旧实现。

当前版本已经具备：
- phase / milestone 状态机
- 候选区域与局部裁剪图
- app-level playbook memory
- 中断恢复与更多生活类 App 先验

但线上失败仍集中在三类：
1. 跳步：未点搜索框直接 TYPE
2. 误点：顶部搜索入口 / 右上小控件点不准
3. 不停：应 COMPLETE 时继续点击

## 本轮改动重点
### 1. TYPE 前置条件硬约束
新增 `_allow_type_now()`：
- 只有 phase 在 `search_input / submit_search`
- 或已经点过顶部搜索框
- 或上一动作就是顶部点击
才允许 TYPE

否则把 TYPE 强制改写为：
- `CLICK` 顶部搜索框 `TOP_SEARCH_BOX`

这条改动直接针对：
- 美团 / 腾讯视频 这类“该先点后输”的失败

### 2. 顶部与右上小控件细粒度定位
新增更细粒度候选区域：
- `TOP_SEARCH_BOX`
- `TOP_LEFT_ICON`
- `TOP_RIGHT_ICON`
- `TOP_RIGHT_SMALL`

同时补充对应局部裁剪图与候选点集合：
- 搜索框使用多个候选点
- 右上角小图标使用更靠边、更靠上的候选点
- `CLICK` 在这些高频小控件区域时，不再完全信任模型原始点，而是做候选点重排

这条改动直接针对：
- 抖音顶部搜索入口误点
- 快手右上小控件误点
- 去哪儿顶部输入栏误点

### 3. COMPLETE 二段式放行
新增 `_verify_complete()`：
- step 数过少时禁止 COMPLETE
- phase 仍在 `launch/home/search_entry/search_input/submit_search` 时禁止 COMPLETE
- page_type 仍像搜索页时禁止 COMPLETE
- 没有输入过关键词、也没有进入 detail/confirm/results 深阶段时禁止 COMPLETE

只有通过代码侧校验，才真正放行 COMPLETE。

这条改动直接针对：
- 芒果 TV “该停不停”
- 喜马拉雅 “该停不停”

### 4. 状态补充
新增状态字段：
- `search_box_clicked`
- `complete_ready`
- `submit_ready`

使 phase 之外还保留更强的动作约束信号。

## 产出文件
- `agent_submission_agent_v3_upgrade.patch`
- `agent_submission_agent_v3_upgraded.py`

## 验证
已完成两步验证：
1. 对新文件做 Python 语法校验，已通过。
2. 在临时目录对当前基线文件真实应用 patch，补丁可正常打上；打补丁后的文件与目标文件逐字节一致。

