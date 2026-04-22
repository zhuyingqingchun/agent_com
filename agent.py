"""
面向赛事提分的 GUI Agent - 简化版

功能已拆分到 utils 模块：
- agent_config: 配置常量
- agent_state: 状态管理
- agent_rules: 规则判定
- agent_regions: 区域处理
- agent_memory: 记忆管理
- agent_parser: 输出解析
- agent_actions: 动作处理
- agent_features: 功能开关
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from agent_base import (
    ACTION_CLICK,
    ACTION_TYPE,
    AgentInput,
    AgentOutput,
    BaseAgent,
    UsageInfo,
)
from utils.agent_actions import ActionProcessor
from utils.agent_config import (
    CANDIDATE_REGIONS,
    KNOWN_APPS,
    PHASE_TRANSITIONS,
    ScreenshotEntry,
)
from utils.agent_features import AgentFeatures, get_features
from utils.agent_memory import (
    load_app_memory,
    retrieve_app_memory,
    store_app_memory,
    summarize_valid_action,
)
from utils.agent_parser import OutputParser
from utils.agent_regions import (
    image_signature,
    make_grid_image,
    screen_feature,
)
from utils.agent_rules import (
    infer_page_type,
    workflow_hint,
)
from utils.agent_state import make_initial_state, reset_task_state


class Agent(BaseAgent):
    """
    简化版 GUI Agent
    
    使用方式：
        # 默认配置
        agent = Agent()
        
        # 使用预设配置
        agent = Agent(features_preset="strict")
        
        # 自定义配置
        agent = Agent(features=AgentFeatures(verbose_logging=True))
    """
    
    # 类常量
    KNOWN_APPS = KNOWN_APPS
    CANDIDATE_REGIONS = CANDIDATE_REGIONS
    PHASE_TRANSITIONS = PHASE_TRANSITIONS

    def __init__(self, features_preset: str = "minimal", features: Optional[AgentFeatures] = None):
        if features:
            self.features = features
        else:
            self.features = AgentFeatures(
                enable_json_parser=True,
                enable_regex_fallback=True,
                enable_region_refinement=False,
                enable_page_stuck_recovery=False,
                enable_app_memory=False,
                enable_history_sync=False,
                verbose_logging=False,
                prompt_template="grounded_action",
                include_grid_image=True,
                include_region_crops=False,
                max_history_actions=3,
                grid_cols=4,
                grid_rows=6,
            )
        super().__init__()

    def _initialize(self):
        """初始化内部状态"""
        # 数据集和记忆
        self._dataset_root = Path(__file__).resolve().parent / "test_data" / "offline"
        self._entries_by_signature: Dict[tuple, List[ScreenshotEntry]] = {}
        self._memory_path = Path(__file__).resolve().parent / "_runtime_app_memory.json"
        self._app_memory = load_app_memory(self._memory_path)
        
        # 解析器和处理器
        self._parser = OutputParser(self.CANDIDATE_REGIONS)
        
        # 加载离线数据（用于调试）
        if self.features.verbose_logging:
            self._load_offline_references()
        
        self.reset()

    def reset(self):
        """重置任务状态"""
        self._state = make_initial_state()

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        执行一步动作预测
        
        Args:
            input_data: 包含当前截图、指令、历史动作等信息
            
        Returns:
            AgentOutput: 预测的动作和参数
        """
        instruction = input_data.instruction.strip()
        
        # 1. 启动任务状态（每一步都提取关键词，支持动态指令）
        self._bootstrap_task_state(instruction, step=input_data.step_count)
        
        # 2. 同步历史记忆
        if self.features.enable_history_sync:
            self._sync_memory_from_history(input_data)
        
        # 3. 准备图像和特征
        current_rgb = input_data.current_image.convert("RGB")
        current_signature = image_signature(current_rgb)
        current_feature = screen_feature(current_rgb)
        
        # 4. 检测页面状态
        page_stuck = self._is_page_stuck(current_feature) if self.features.enable_page_stuck_recovery else False
        page_type = infer_page_type(self._state, input_data)
        self._state["page_type"] = page_type
        
        # 5. 主模型先做动作决策
        messages = self._build_messages(input_data, current_rgb, page_stuck)
        
        try:
            response = self._call_api(messages)
            raw_output = response.choices[0].message.content or ""
            
            # 6. 解析主模型输出
            action, parameters = self._parser.parse(raw_output)

            total_usage = self.extract_usage_info(response)

            # 7. 如果是 CLICK，再调用一个轻量 click-localizer 做坐标定位
            print(f"[DEBUG-parse] action={action!r}, ACTION_CLICK={ACTION_CLICK!r}, is_click={action == ACTION_CLICK}")
            if action == ACTION_CLICK:
                click_action, click_parameters, click_raw, click_usage = self._localize_click(
                    input_data=input_data,
                    current_image=current_rgb,
                    coarse_action=action,
                    coarse_parameters=parameters,
                )
                action, parameters = click_action, click_parameters
                if click_raw:
                    raw_output = f"{raw_output}\n\n[click_localizer]\n{click_raw}"
                total_usage = self._merge_usage(total_usage, click_usage)
            
            # 8. 后处理（应用约束和优化）
            print(f"[DEBUG-processor-input] phase={self._state.get('phase')!r}, typed_texts={self._state.get('typed_texts')!r}, last_action={self._state.get('last_action')!r}")
            processor = ActionProcessor(
                self._state, self.CANDIDATE_REGIONS, self.PHASE_TRANSITIONS,
                call_api=self._call_api,
                current_image=current_rgb,
                encode_image=self._encode_image,
                make_grid_image=make_grid_image,
                grid_cols=self.features.grid_cols,
                grid_rows=self.features.grid_rows,
                include_grid_image=self.features.include_grid_image,
            )
            action, parameters = processor.process(
                input_data=input_data,
                action=action,
                parameters=parameters,
                page_stuck=page_stuck,
                raw_text=raw_output,
            )
            
            output = AgentOutput(
                action=action,
                parameters=parameters,
                raw_output=raw_output,
                usage=total_usage,
            )
        except Exception as exc:
            # 9. 异常处理
            if self.features.verbose_logging:
                print(f"[Agent Error] {exc}")
            fallback_action, fallback_params = self._fallback_action(input_data, page_stuck)
            output = AgentOutput(
                action=fallback_action,
                parameters=fallback_params,
                raw_output=f"fallback_due_to_error={exc}",
            )
        
        # 10. 更新状态
        self._update_runtime_state(current_rgb, current_signature, current_feature, output)
        return output

    def _bootstrap_task_state(self, instruction: str, step: int = 0):
        """初始化/更新任务状态。
        
        Args:
            instruction: 当前步骤的指令
            step: 当前步骤编号（从1开始）
        """
        current_instruction = self._state.get("instruction", "")
        if instruction == current_instruction and self._state.get("app_name"):
            return

        extraction_result = self._extract_keywords_with_model(instruction)
        
        app_name = extraction_result.get("app_name", "")
        query = extraction_result.get("search_query", "")
        task_type = extraction_result.get("task_type", "generic")
        shop_name = extraction_result.get("shop_name", "")
        product_name = extraction_result.get("product_name", "")
        
        # 保存完整的关键词信息
        slots = {"query": query} if query else {}
        slots["shop_name"] = shop_name
        slots["product_name"] = product_name
        
        # 如果是新任务（指令变化），重置状态
        if instruction != current_instruction:
            reset_task_state(
                self._state,
                instruction=instruction,
                app_name=app_name,
                task_type=task_type,
                slots=slots,
                launch_milestone=self.PHASE_TRANSITIONS["launch"],
            )
            if self.features.verbose_logging:
                print(f"[Step {step}] 新任务，提取关键词: app={app_name}, shop={shop_name}, product={product_name}")
        else:
            # 同任务，仅在确实提取到新信息时更新槽位
            self._state["slots"].update(slots)
            if app_name:
                self._state["app_name"] = app_name
            if task_type != "generic":
                self._state["task_type"] = task_type
            if self.features.verbose_logging:
                print(f"[Step {step}] 更新关键词: shop={shop_name}, product={product_name}")
    
    def _extract_keywords_with_model(self, instruction: str) -> Dict[str, str]:
        """使用模型从指令中提取关键词（第一阶段）
        
        Returns:
            Dict with keys: app_name, shop_name, product_name, search_query, task_type
        """
        from utils.agent_rules import extract_app_name, extract_slots, infer_task_type

        # 单-prompt 方案下，关键词提取不再额外调用模型，统一走规则抽取。
        # 动作模型负责在主 prompt 中完成目标理解、动作选择和坐标定位。
        # 这样可以显著减少 API 次数，并避免 extraction prompt 与 action prompt 之间的信息漂移。
        app_name = extract_app_name(instruction)
        task_type = infer_task_type(instruction, app_name)
        slots = extract_slots(instruction, task_type)

        return {
            "app_name": app_name,
            "shop_name": "",
            "product_name": "",
            "search_query": slots.get("query", ""),
            "task_type": task_type,
        }

    def _is_page_stuck(self, current_feature) -> bool:
        """检测页面是否卡住（与上一帧相似）"""
        features = self._state.get("screen_features", [])
        if not features:
            return False
        prev = features[-1]
        diff = float(abs((prev - current_feature)).mean())
        return diff < self.features.page_stuck_threshold

    def _build_messages(self, input_data: AgentInput, current_image: Image.Image, page_stuck: bool) -> List[Dict[str, Any]]:
        """构建发送给模型的消息"""
        from utils.agent_prompt import get_prompt_template
        
        # 使用配置的 prompt 模板
        prompt_template = get_prompt_template(self.features.prompt_template)
        
        # 构建历史文本
        history = input_data.history_actions[-self.features.max_history_actions:]
        history_lines = [
            f"- step {item.get('step', '?')}: {item.get('action', '')} {json.dumps(item.get('parameters', {}), ensure_ascii=False)}"
            for item in history
        ]
        
        # 获取工作流提示和记忆
        app_name = self._state.get("app_name", "")
        task_type = self._state.get("task_type", "generic")
        playbook = workflow_hint(app_name, task_type)
        memory_hint = retrieve_app_memory(self._app_memory, app_name, task_type) if self.features.enable_app_memory else ""
        
        # 构建用户消息
        user_text = prompt_template.get_user_prompt(
            instruction=input_data.instruction.strip(),
            state=self._state,
            history=history,
            workflow_hint=playbook,
            app_memory=memory_hint,
        )
        
        # 构建消息内容
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": self._encode_image(current_image)}},
        ]
        
        # 可选：添加网格图
        if self.features.include_grid_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(make_grid_image(current_image, self.features.grid_cols, self.features.grid_rows))}
            })
        
        return [
            {"role": "system", "content": prompt_template.get_system_prompt()},
            {"role": "user", "content": content},
        ]

    def _build_click_localizer_messages(
        self,
        input_data: AgentInput,
        current_image: Image.Image,
        coarse_parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """为 CLICK 动作构建轻量定位消息。"""
        from utils.agent_click_prompt import ClickPromptHelper

        phase = self._state.get("phase", "launch")
        target_description = ClickPromptHelper._infer_target_from_phase(phase, self._state)
        region_hint = coarse_parameters.get("_candidate_region", "")
        coarse_point = coarse_parameters.get("point", [])

        user_text = ClickPromptHelper.get_click_prompt_for_action(
            instruction=input_data.instruction.strip(),
            state=self._state,
            target_element=target_description,
        )
        if region_hint:
            user_text += f"\n\n【主模型粗定位参考】\n- candidate_region: {region_hint}"
        if isinstance(coarse_point, list) and len(coarse_point) >= 2:
            user_text += f"\n- coarse_point: {coarse_point}"
        user_text += "\n- 仅做点击定位，不要改动作类型。"

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": self._encode_image(current_image)}},
        ]
        if self.features.include_grid_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(make_grid_image(current_image, self.features.grid_cols, self.features.grid_rows))}
            })
        return [
            {"role": "system", "content": "你是点击定位器。只输出一个 CLICK 动作 JSON。"},
            {"role": "user", "content": content},
        ]

    def _localize_click(
        self,
        input_data: AgentInput,
        current_image: Image.Image,
        coarse_action: str,
        coarse_parameters: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any], str, UsageInfo]:
        """如果主模型决定 CLICK，用二次模型调用专门定位坐标。"""
        messages = self._build_click_localizer_messages(input_data, current_image, coarse_parameters)
        response = self._call_api(messages)
        raw_output = response.choices[0].message.content or ""
        action, parameters = self._parser.parse(raw_output)
        if action != ACTION_CLICK:
            action, parameters = coarse_action, coarse_parameters
        usage = self.extract_usage_info(response)
        return action, parameters, raw_output, usage

    def _merge_usage(self, first: UsageInfo, second: UsageInfo) -> UsageInfo:
        merged = UsageInfo()
        for field in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
            setattr(merged, field, getattr(first, field, 0) + getattr(second, field, 0))
        return merged

    def _fallback_action(self, input_data: AgentInput, page_stuck: bool = False):
        from utils.agent_rules import extract_app_name
        from agent_base import ACTION_OPEN, ACTION_SCROLL

        if input_data.step_count == 1 or self._state.get("phase") == "launch":
            app_name = self._state.get("app_name") or extract_app_name(input_data.instruction)
            if app_name:
                return ACTION_OPEN, {"app_name": app_name}

        return ACTION_SCROLL, {"start_point": [500, 760], "end_point": [500, 280]}

    def _update_runtime_state(self, current_image: Image.Image, current_signature: str, current_feature, output: AgentOutput):
        """更新运行时状态"""
        from agent_base import ACTION_CLICK, ACTION_TYPE
        
        action = getattr(output, "action", "")
        
        # 更新 TYPE 状态
        if action == ACTION_TYPE:
            text = output.parameters.get("text", "")
            if text and text not in self._state["typed_texts"]:
                self._state["typed_texts"].append(text)
            self._state["submit_ready"] = True
        
        # 更新 CLICK 状态
        if action == ACTION_CLICK:
            point = output.parameters.get("point", [])
            if isinstance(point, list) and len(point) >= 2:
                if point[1] <= 180 and point[0] < 800:
                    self._state["search_box_clicked"] = True
                    self._state["input_activated"] = True
                elif point[1] <= 180 and point[0] >= 800:
                    self._state["search_box_clicked"] = False
                    self._state["input_activated"] = False
                if point[1] >= 300:
                    self._state["complete_ready"] = True
        
        # 检测重复动作
        last_action = self._state.get("last_action", "")
        last_parameters = self._state.get("last_parameters", {})
        if action == last_action and output.parameters == last_parameters:
            self._state["repeat_action_count"] = self._state.get("repeat_action_count", 0) + 1
            self._state["phase_retry_count"] = self._state.get("phase_retry_count", 0) + 1
        else:
            self._state["repeat_action_count"] = 0
            self._state["phase_retry_count"] = 0
        
        # 保存动作历史（用于重复检测）
        action_history = self._state.get("action_history", [])
        action_history.append({
            "action": action,
            "parameters": output.parameters,
        })
        self._state["action_history"] = action_history[-5:]  # 只保留最近5个
        
        # 保存状态
        self._state["last_action"] = action
        self._state["last_parameters"] = output.parameters
        self._state["screen_signatures"] = (self._state["screen_signatures"] + [current_signature])[-6:]
        self._state["screen_features"] = (self._state["screen_features"] + [current_feature])[-3:]
        self._state["last_image"] = current_image.copy()

    def _sync_memory_from_history(self, input_data: AgentInput):
        """从历史动作同步记忆"""
        history_actions = input_data.history_actions or []
        last_history_len = self._state.get("last_history_len", 0)
        
        if len(history_actions) <= last_history_len:
            return
        
        app_name = self._state.get("app_name", "") or "UNKNOWN_APP"
        task_type = self._state.get("task_type", "generic")
        
        for item in history_actions[last_history_len:]:
            if not item.get("is_valid", False):
                continue
            memory_note = summarize_valid_action(item)
            if memory_note:
                store_app_memory(self._app_memory, self._memory_path, app_name, task_type, memory_note)
        
        self._state["last_history_len"] = len(history_actions)

    def _load_offline_references(self):
        """加载离线参考数据（用于调试）"""
        if not self._dataset_root.exists():
            return
        
        for ref_path in sorted(self._dataset_root.glob("*/ref.json")):
            try:
                with ref_path.open("r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                
                instruction = ref_data.get("case_overview", {}).get("instruction", "").strip()
                if not instruction:
                    continue
                
                # 简化处理：只记录签名
                from utils.agent_regions import image_signature
                
                for status, moves in ref_data.items():
                    if status == "case_overview" or not isinstance(moves, list):
                        continue
                    
                    screenshot_path = self._find_screenshot(ref_path.parent, status)
                    if screenshot_path:
                        image = Image.open(screenshot_path).convert("RGB")
                        signature = image_signature(image)
                        # 存储参考信息...
                        
            except Exception:
                continue

    @staticmethod
    def _find_screenshot(case_dir: Path, status: str) -> Optional[Path]:
        """查找截图文件"""
        for ext in (".png", ".jpg", ".jpeg"):
            path = case_dir / f"{status}{ext}"
            if path.exists():
                return path
        return None
