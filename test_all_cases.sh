#!/bin/bash
# 批量测试所有用例

export VLM_API_KEY='60b2acf929e44f9c8bf3d9710a465220.CFHrxV7faSlOUGzi'
export DEBUG_API_URL='https://open.bigmodel.cn/api/paas/v4'
export DEBUG_MODEL_ID='glm-4v-flash'

TEST_CASES=(
    "step_aiqiyi_onekey_0011"
    "step_baidumap_onekey_0008"
    "step_baidumap_onekey_0010"
    "step_bilibili_onekey_0008"
    "step_douyin_onekey_0008"
    "step_kuaishou_onekey_0003"
    "step_mangguo_onekey_0008"
    "step_meituan_onekey_0001"
    "step_quonekey_0030"
    "step_tengxunshipin_onekey_0005"
    "step_ximalaya_onekey_0001"
)

echo "=========================================="
echo "开始批量测试 - 使用 GLM-4V 模型"
echo "=========================================="
echo ""

for case in "${TEST_CASES[@]}"; do
    echo "------------------------------------------"
    echo "测试用例: $case"
    echo "------------------------------------------"
    conda run -n swdtorch12 python test_runner.py --data_dir ./test_data/offline/$case --no_debug_test 2>&1 | grep -E "(Result:|PASS|FAIL|Accuracy|Error)"
    echo ""
done

echo "=========================================="
echo "批量测试完成"
echo "=========================================="
