# 运行输出说明

## 关键文件

- `prompt_trace.jsonl`
  - 每次模型调用一条记录
  - 包含：调用来源、阶段、step、prompt 文本、脱敏后的 messages、模型输出、usage

- `single_case_result.json`
  - 单用例模式下的汇总结果

- `run_summary.json`
  - 全量模式下 TestRunner 的汇总结果

- `test_run.log`
  - 官方 TestRunner 日志

## 推荐查看顺序

1. 先看 `test_run.log`，确认哪一步失败
2. 再看 `prompt_trace.jsonl` 中同一步附近的记录
3. 对照：
   - `messages_text_only`
   - `model_output`
   - `prompt_origin`
4. 判断模型是：
   - 没理解任务
   - 没理解截图
   - 被 prompt 误导
   - 还是后处理 / 解析阶段出了问题
