"""
离线 Prompt 逐步分析工具 - 加入完整历史操作

用法：
  python analyze_prompts.py --data_dir ./test_data/offline/step_aiqiyi_onekey_0011
  python analyze_prompts.py --data_dir ./test_data/offline/step_aiqiyi_onekey_0011 --step 9
"""
import os, sys, json
from PIL import Image


def load_test_data(data_dir):
    with open(os.path.join(data_dir, "ref.json"), "rt", encoding="utf-8") as f:
        return json.load(f)


def get_screenshot(status, dir_path):
    for ext in [".png", ".jpg"]:
        path = os.path.join(dir_path, status + ext)
        if os.path.exists(path):
            return Image.open(path), path
    return Image.new("RGB", (1080, 2400)), path


def main():
    data_dir = sys.argv[sys.argv.index("--data_dir") + 1] if "--data_dir" in sys.argv else "./test_data/offline/step_aiqiyi_onekey_0011"
    target_step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else None

    ref_data = load_test_data(data_dir)
    instruction = ref_data["case_overview"]["instruction"]
    print("=" * 80)
    print(f"【原始指令】{instruction}")
    print(f"【指令长度】{len(instruction)} 字符")
    print("=" * 80)

    from utils.agent_state import make_initial_state
    from utils.agent_features import AgentFeatures
    from agent_base import AgentInput
    from agent import Agent

    state = make_initial_state()
    state["instruction"] = instruction
    agent = Agent(features=AgentFeatures(verbose_logging=True))

    current_status = "0"
    step_count = 1
    history_actions = []
    max_step = 11

    while current_status != "#" and step_count <= max_step:
        if target_step and step_count != target_step:
            # 模拟前面步骤的推进（使用参考动作）
            ref_list = ref_data.get(current_status, [])
            if ref_list:
                ra = ref_list[0]
                rp = ra.get("params", {})
                next_status = ra.get("next", "#")
                hist_entry = {"step": step_count, "action": ra["action"], "parameters": {}}
                if "text" in rp:
                    hist_entry["parameters"] = {"text": rp["text"]}
                    state.setdefault("typed_texts", []).append(rp["text"])
                    state["submit_ready"] = True
                elif "x" in rp:
                    cx = (rp["x"][0] + rp["x"][1]) / 2
                    cy = (rp["y"][0] + rp["y"][1]) / 2
                    hist_entry["parameters"] = {"point": [cx, cy]}
                    if cy >= 800:
                        state["phase"] = "confirm"; state["complete_ready"] = True
                    elif 180 < cy < 800:
                        state["phase"] = "detail" if state.get("typed_texts") else "results"
                    else:
                        if cx >= 800: state["phase"] = "search_entry"
                        else: state["phase"] = "search_input"
                elif "app" in rp:
                    hist_entry["parameters"] = {"app_name": rp["app"]}
                    state["phase"] = "home"

                history_actions.append(hist_entry)
                current_status = next_status
                step_count += 1
            continue

        print(f"\n{'='*80}")
        print(f"  Step {step_count}: Status={current_status}  |  模拟历史操作数: {len(history_actions)}")
        print(f"{'='*80}")

        screenshot, pic_path = get_screenshot(current_status, data_dir)
        print(f"  截图: {pic_path} ({screenshot.size[0]}x{screenshot.size[1]})")

        input_data = AgentInput(
            instruction=instruction,
            current_image=screenshot,
            step_count=step_count,
            history_messages=[],
            history_actions=history_actions,
        )

        # 关键：将模拟推进的 state 同步到 agent._state
        agent._state = state

        try:
            messages = agent._build_messages(input_data, screenshot.convert("RGB"), False)
        except Exception as e:
            print(f"  [ERROR] _build_messages 失败: {e}")
            import traceback; traceback.print_exc()
            break

        # ===== System Prompt =====
        sys_text = messages[0]["content"]
        print(f"\n  ═══ System Prompt ({len(sys_text)} 字符) ═══")
        for i, line in enumerate(sys_text.split("\n"), 1):
            print(f"    {i:3d}: {line}")

        # ===== User Prompt - 文本部分 =====
        user_content = messages[1]["content"]
        text_part = None
        img_parts = []
        for part in user_content:
            if part["type"] == "text":
                text_part = part["text"]
            else:
                img_parts.append(part)

        print(f"\n  ═══ User Prompt - 文本 ({len(text_part) if text_part else 0} 字符) ═══")
        if text_part:
            for i, line in enumerate(text_part.split("\n"), 1):
                marker = ""
                if any(kw in line for kw in ["指令", "真是太好看了", "评论", "狂飙"]):
                    marker = " ◆◆ 关键信息"
                if any(kw in line for kw in ["阶段侧重点", "动作语义", "决策提醒", "当前状态", "最近动作", "已输入", "typed_texts"]):
                    marker = " ★ 状态/指导"
                if any(kw in line for kw in ["步骤", "phase:", "search_box_clicked"]):
                    marker = " ▸ 阶段信息"
                print(f"    {i:3d}: {line}{marker}")

        # ===== 指令完整性检查 =====
        print(f"\n  ──── 信息完整性检查 ────")
        checks = [
            ("原始指令完整", instruction in (text_part or "")),
            ("包含'评论'", "评论" in (text_part or "")),
            ("包含'真是太好看了'", "真是太好看了" in (text_part or "")),
            ("包含'狂飙'", "狂飙" in (text_part or "")),
            ("有阶段侧重点", "阶段侧重点" in (text_part or "")),
            ("有动作语义", "动作语义" in (text_part or "")),
            ("有页面决策提醒", "页面决策提醒" in (text_part or "")),
            ("有当前状态段", "当前状态" in (text_part or "")),
            ("有历史动作段", "最近动作" in (text_part or "") or "执行进度" in (text_part or "")),
            ("图片数量", len(img_parts)),
        ]
        all_ok = True
        for name, result in checks:
            status = "✅" if (result is True or (isinstance(result, int) and result > 0)) else "❌"
            if isinstance(result, bool) and not result:
                all_ok = False
            val = result if isinstance(result, int) else ""
            print(f"    {status} {name}: {val}")

        if not all_ok:
            print(f"\n  ⚠️ 有检查项未通过！需要修复。")

        # ===== 约束触发预检 =====
        phase = state.get("phase", "launch")
        typed_texts = state.get("typed_texts", [])
        last_action = history_actions[-1]["action"] if history_actions else ""
        last_params = history_actions[-1].get("parameters", {}) if history_actions else {}
        lp_raw = last_params.get("point", []) or []
        already_bottom = (last_action == "CLICK" and len(lp_raw) >= 2 and lp_raw[1] >= 700)

        print(f"\n  ──── 约束触发条件 ────")
        c1_ok = phase in {"detail","confirm"} and agent._call_api is not None
        c2_ok = (phase == "confirm" and bool(typed_texts) and agent._call_api is not None
                 and step_count < 20 and not already_bottom)
        print(f"    约束1(顶部点击矫正): {'🔥 会触发' if c1_ok else '⬜ 不满足'}  phase={phase}, call_api={'有' if agent._call_api else '无'}")
        print(f"    约束2(过早COMPLETE):   {'🔥 会触发' if c2_ok else '⬜ 不满足'}  phase={phase}, typed_texts={typed_texts}, 上一步底部点击?={already_bottom}")

        # ===== 参考动作 & 推进 =====
        ref_list = ref_data.get(current_status, [])
        if ref_list:
            ra = ref_list[0]
            rp = ra.get("params", {})
            next_status = ra.get("next", "#")
            print(f"\n  ──── 参考答案 ────")
            print(f"    期望: {ra['action']} {rp} → next={next_status}")

            hist_entry = {"step": step_count, "action": ra["action"], "parameters": {}}
            if "text" in rp:
                hist_entry["parameters"] = {"text": rp["text"]}
                state.setdefault("typed_texts", []).append(rp["text"])
                state["submit_ready"] = True
            elif "x" in rp:
                cx = (rp["x"][0] + rp["x"][1]) / 2
                cy = (rp["y"][0] + rp["y"][1]) / 2
                hist_entry["parameters"] = {"point": [cx, cy]}
                if cy >= 800:
                    state["phase"] = "confirm"; state["complete_ready"] = True
                elif 180 < cy < 800:
                    state["phase"] = "detail" if state.get("typed_texts") else "results"
                else:
                    if cx >= 800: state["phase"] = "search_entry"
                    else: state["phase"] = "search_input"
            elif "app" in rp:
                hist_entry["parameters"] = {"app_name": rp["app"]}
                state["phase"] = "home"

            history_actions.append(hist_entry)
            current_status = next_status
            step_count += 1
        else:
            break

    print(f"\n{'='*80}")
    print(f"  全部 {step_count-1} 步分析完成")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
