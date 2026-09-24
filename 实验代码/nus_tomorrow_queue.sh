#!/bin/bash
# ============================================================================
# 明日 NUS 三任务队列（服务器版，AutoDL Linux）
#   T1  B2    : cons_form=soft(旧名 mse) 非承诺软目标一致性, gaussian 全量
#               -> 判定: vs 零样本 15.24 / 无门基线 13.86
#   T2  EPS   : corr_eps=0.001 (导师收缩正则), gaussian 全量
#               -> 判定: vs 13.86 (协议注意: 该臂故意开 eps, 其余臂关)
#   T3  CONTR : 基线 + 双门 + 纯特征门, contrast 全量 (崩溃态上门控行为)
#               -> 判定: vs 实验32 contrast 零样本 5.84 / 长流基线 5.27
#
# 依赖(已部署的 7 文件 + ckpt): run_real.py tta.py config.py real_model.py
#   utils_tools_port.py mldata.py vit_nuswide_best.pth
#   !! 部署前把本机这 6 个 .py 重新上传 —— 2026-09-16 晚的新改动
#   (cons_form/sc_confirm/sc_feat_only/cond_teacher_feat/corr_eps/drift_gate)
#   都在这些文件里, 云端若是旧版会直接 argparse 报错退出。
#
# 用法:
#   chmod +x nus_tomorrow_queue.sh
#   nohup bash nus_tomorrow_queue.sh > nus_tomorrow.log 2>&1 &
#   tail -f nus_tomorrow.log          # 看进度; 每臂 ~110-130 分钟(4090)
#
# 断点续跑: 每臂完成后写 done 标记到 MARK_DIR; 重跑同一命令自动跳过
#   已完成臂, 中断后再次 nohup 即可。
# ============================================================================
set -u
PY=${PY:-"python"}                      # AutoDL conda env 里的 python
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted"}
CLEAN_ROOT=${CLEAN_ROOT:-"/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr"}
CKPT=${CKPT:-"/root/autodl-tmp/ML-TTA/vit_nuswide_best.pth"}
CACHE=${CACHE:-"/root/autodl-tmp/ML-TTA/cache_tsml_nuswide_src_stats.pt"}
MARK_DIR=${MARK_DIR:-"/root/autodl-tmp/ML-TTA/marks_nus_tomorrow"}
mkdir -p "$MARK_DIR"
export NUS_CLEAN_ROOT="$CLEAN_ROOT"     # NuswideCsvClean 源统计的干净图根

# 长流适配定稿 + 全部对照臂通用的 --corr_eps 0 (与实验32 锚点 13.86 严格
# 同协议; NUS 有 6/81 类 cov diag < 1e-3, 默认 eps=0.001 会咬合)。
# drift_gate 10 = 只记录不触发 (健康信号进日志)。
BASE="--arch vit --checkpoint $CKPT --dataset_name nuswide \
--source_dir $DATA_ROOT/gaussian_noise_5/NUSWIDE \
--test_root $DATA_ROOT --test_pattern {corr}_5/NUSWIDE --cache $CACHE \
--methods full --no_refresh_weak --bn_warmup 0 --pi_freeze --ema 0.999 \
--rebalance --pi_hard --pi_floor 0.5 --tau_adaptive --rho 0.10 \
--proto_confirm --struct_form corr --lambda_struct 0.25 --corr_eps 0 \
--drift_gate 10"

run_arm () {   # run_arm <mark> <extra flags...>
  local mark=$1; shift
  if [ -f "$MARK_DIR/$mark.done" ]; then
    echo "=== [$mark] already done, skip ==="
    return 0
  fi
  echo "=== [$mark] start $(date '+%F %T') ==="
  $PY -u run_real.py $BASE "$@" && touch "$MARK_DIR/$mark.done" \
    || echo "!!! [$mark] FAILED (exit $?) — 继续后续臂"
}

# ---- T1: B2 非承诺软目标一致性 (cons_form=soft; 旧名 mse 也接受) ----------
run_arm B2_soft_gaussian    --cons_form soft --conc_gate 0.35 \
  --corruptions gaussian_noise

# ---- T2: eps 收缩正则 (唯一故意开 eps 的臂) --------------------------------
run_arm EPS001_gaussian     --corr_eps 0.001 \
  --corruptions gaussian_noise

# ---- T3: contrast 崩溃态 — 基线 / 双门 / 纯特征门 --------------------------
#   基线臂 = 与 13.86 同配置跑 contrast, 拿到"崩溃态无门基线"(实验32 只有
#   旧配置 5.27, 无长流适配版基线), 双门/纯特征门与它成对比较。
run_arm C_base_contrast     --corruptions contrast
run_arm SC_double_contrast  --sc_confirm     --corruptions contrast
run_arm SF_feat_contrast    --sc_feat_only   --corruptions contrast

echo "=== QUEUE COMPLETE $(date '+%F %T') — 5 marks: ==="
ls -la "$MARK_DIR"
echo "判定锚点: gaussian 零样本 15.24 / 无门基线 13.86 / S_c 门 12.610"
echo "          contrast 零样本 5.84 / 旧配置长流 5.27(实验32)"
