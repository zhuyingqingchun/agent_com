# Prompt Source Map

这个文件说明 `prompt_trace.jsonl` 里不同阶段的 prompt 来自哪里。

## 1. 主动作决策（main_action）
- `utils/agent_prompt.py::{具体模板类}.get_system_prompt`
- `utils/agent_prompt.py::{具体模板类}.get_user_prompt`
- `agent.py::Agent._build_messages`

当前仓库 `Agent.__init__()` 里默认 `prompt_template="grounded_action"`。

## 2. 点击二阶段定位（click_localizer）
- `utils/agent_click_prompt.py::ClickPromptHelper.get_click_prompt_for_action`
- `agent.py::Agent._build_click_localizer_messages`

## 3. 工作流规划（workflow_planner）
- `agent.py::Agent._plan_workflow`

## 4. 顶部误点纠偏（phase_corrector）
- `utils/agent_actions.py::ActionProcessor._call_phase_corrector`

## 5. 左侧栏误点纠偏（left_panel_corrector）
- `utils/agent_actions.py::ActionProcessor._call_left_panel_corrector`

## 6. 过早 COMPLETE 纠偏（completion_checker）
- `utils/agent_actions.py::ActionProcessor._call_completion_checker`
