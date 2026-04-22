#!/bin/bash
# 使用真实 GLM-4V API 测试完整用例

export VLM_API_KEY='60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
export DEBUG_API_URL='https://open.bigmodel.cn/api/paas/v4'
export DEBUG_MODEL_ID='glm-4v-flash'

CASE=${1:-"step_meituan_onekey_0001"}

echo "=========================================="
echo "真实 GLM-4V API 完整测试"
echo "测试用例: $CASE"
echo "=========================================="
echo ""

conda run -n swdtorch12 python test_runner.py --data_dir ./test_data/offline/$CASE 2>&1 | grep -E "(Step|Agent Output|action=|params=|Result:|PASS|FAIL|Score|Accuracy)"

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
