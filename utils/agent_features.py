"""
功能开关配置模块 - 控制 Agent 各项功能的启用/禁用
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AgentFeatures:
    """Agent 功能开关配置"""

    # 解析功能
    enable_json_parser: bool = True
    enable_regex_fallback: bool = True

    # 优化功能
    enable_region_refinement: bool = True

    # 恢复功能
    enable_page_stuck_recovery: bool = True

    # 记忆功能
    enable_app_memory: bool = True
    enable_history_sync: bool = True

    # 调试功能
    verbose_logging: bool = False

    # Prompt 配置
    prompt_template: str = "grounded_action"
    include_grid_image: bool = True
    include_region_crops: bool = False
    max_history_actions: int = 8

    # 网格配置
    grid_cols: int = 4
    grid_rows: int = 6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_json_parser": self.enable_json_parser,
            "enable_regex_fallback": self.enable_regex_fallback,
            "enable_region_refinement": self.enable_region_refinement,
            "enable_page_stuck_recovery": self.enable_page_stuck_recovery,
            "enable_app_memory": self.enable_app_memory,
            "enable_history_sync": self.enable_history_sync,
            "verbose_logging": self.verbose_logging,
            "prompt_template": self.prompt_template,
            "include_grid_image": self.include_grid_image,
            "include_region_crops": self.include_region_crops,
            "max_history_actions": self.max_history_actions,
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AgentFeatures":
        return cls(**{
            k: v for k, v in config.items()
            if k in cls.__dataclass_fields__
        })

    @classmethod
    def minimal(cls) -> "AgentFeatures":
        return cls(
            enable_region_refinement=False,
            enable_app_memory=False,
            enable_history_sync=False,
            include_grid_image=False,
            max_history_actions=3,
        )

    @classmethod
    def fast(cls) -> "AgentFeatures":
        return cls(
            enable_region_refinement=False,
            enable_app_memory=False,
            enable_history_sync=False,
            include_grid_image=False,
            max_history_actions=3,
        )


FEATURE_PRESETS = {
    "default": AgentFeatures(),
    "minimal": AgentFeatures.minimal(),
    "fast": AgentFeatures.fast(),
}


def get_features(preset_name: str = "default", **overrides) -> AgentFeatures:
    if preset_name not in FEATURE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(FEATURE_PRESETS.keys())}")
    features = FEATURE_PRESETS[preset_name]
    if overrides:
        config = features.to_dict()
        config.update(overrides)
        return AgentFeatures.from_dict(config)
    return features


def list_presets() -> list:
    return list(FEATURE_PRESETS.keys())
