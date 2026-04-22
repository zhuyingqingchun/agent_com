#!/bin/bash
# 测试单个用例，显示模型原始输出

export VLM_API_KEY='60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
export DEBUG_API_URL='https://open.bigmodel.cn/api/paas/v4'
export DEBUG_MODEL_ID='glm-4v-flash'

# 默认测试第一个用例，可以通过参数指定
CASE=${1:-"step_meituan_onekey_0001"}

echo "=========================================="
echo "测试用例: $CASE"
echo "模型: glm-4v-flash"
echo "=========================================="
echo ""

conda run -n swdtorch12 python test_runner.py --data_dir ./test_data/offline/$CASE 2>&1 | grep -E "(Step|Agent Output|raw_output|action=|params=|Result:|PASS|FAIL|Score)"

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
