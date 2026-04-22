#!/bin/bash
# 使用真实 GLM-4V API 测试所有样例

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

RESULT_FILE="./output/glm4v_test_results_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$RESULT_FILE"
echo "GLM-4V 真实 API 全量测试" | tee -a "$RESULT_FILE"
echo "开始时间: $(date)" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=${#TEST_CASES[@]}

for case in "${TEST_CASES[@]}"; do
    echo "" | tee -a "$RESULT_FILE"
    echo "##########################################" | tee -a "$RESULT_FILE"
    echo "测试用例: $case" | tee -a "$RESULT_FILE"
    echo "##########################################" | tee -a "$RESULT_FILE"
    echo "开始时间: $(date '+%H:%M:%S')" | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    
    # 运行测试并记录详细输出
    conda run -n swdtorch12 python test_runner.py --data_dir ./test_data/offline/$case 2>&1 | tee -a "$RESULT_FILE"
    
    # 检查结果
    if grep -q "Result: PASS" "$RESULT_FILE" | tail -1; then
        ((PASS_COUNT++))
        echo "" | tee -a "$RESULT_FILE"
        echo "✅ $case: PASS" | tee -a "$RESULT_FILE"
    else
        ((FAIL_COUNT++))
        echo "" | tee -a "$RESULT_FILE"
        echo "❌ $case: FAIL" | tee -a "$RESULT_FILE"
    fi
    
    echo "结束时间: $(date '+%H:%M:%S')" | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
done

echo "" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"
echo "测试完成" | tee -a "$RESULT_FILE"
echo "结束时间: $(date)" | tee -a "$RESULT_FILE"
echo "==========================================" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"
echo "统计结果:" | tee -a "$RESULT_FILE"
echo "  总用例数: $TOTAL_COUNT" | tee -a "$RESULT_FILE"
echo "  通过: $PASS_COUNT" | tee -a "$RESULT_FILE"
echo "  失败: $FAIL_COUNT" | tee -a "$RESULT_FILE"
echo "  通过率: $(echo "scale=2; $PASS_COUNT * 100 / $TOTAL_COUNT" | bc)%" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"
echo "详细结果已保存到: $RESULT_FILE"
