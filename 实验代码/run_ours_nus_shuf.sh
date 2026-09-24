#!/bin/bash
# ============================================================
# 教师-学生框架 NUS-WIDE 乱序（打乱顺序的测试流）全量实验 — AutoDL 版
# 目的: 为实验5/32 的 NUS 分块流主表补"随机打乱的流"对照设置:
#   ResNet: 实验35 §3.5 只测了 brightness/jpeg 两项 (翻正 +0.93/+0.81),
#           本脚本补全 15 项, 逐项量化 -2.68 里流结构税的份额;
#   ViT:    验证"LayerNorm 免疫流结构"——若乱序与分块流同分,
#           则 ViT 的 -2.42 全部归于教师质量+长流, 归因链闭合。
# 设计: 与分块流定稿配置逐字一致 (run_rn_nus.sh / run_vit_nus.sh),
#       唯一差别 --shuffle (seed 固定 0)。不加 --aug_mild 等新改动,
#       加了会破坏与实验5 的受控对照。零样本与流序无关 (无适应状态),
#       直接引用实验5 数字 (ViT 20.14 / RN 12.20), 不重跑。
# 依赖: run_real.py config.py real_model.py tta.py mldata.py
#       utils_tools_port.py + resnet50_nuswide_best.pth
#       + vit_nuswide_best.pth 同目录 (或用 CKPT_RN/CKPT_VIT 覆盖)
# 用法: nohup bash run_ours_nus_shuf.sh > ours_nus_shuf_log.txt 2>&1 &
#       只跑单骨干:   ARCHS="resnet50" nohup bash run_ours_nus_shuf.sh ...
#       只跑部分损坏: CORRS="brightness jpeg_compression" bash run_ours_nus_shuf.sh
#       续跑: 重复执行, 已完成项自动跳过 (ResNet 的 brightness/jpeg
#       若旧日志 logs_ours_rn_nus_shuf/ 还在, 同样自动跳过)
# 提醒: Windows 上传后先  sed -i 's/\r$//' run_ours_nus_shuf.sh
# ============================================================
set -u

PY=${PY:-python}
WORK=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT=${DATA_ROOT:-"/root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted"}
CLEAN_ROOT=${CLEAN_ROOT:-"/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr"}
SRC_DIR="$DATA_ROOT/gaussian_noise_5/NUSWIDE"
CKPT_RN=${CKPT_RN:-"$WORK/resnet50_nuswide_best.pth"}
CKPT_VIT=${CKPT_VIT:-"$WORK/vit_nuswide_best.pth"}
CACHE_RN=${CACHE_RN:-"$WORK/cache_tsml_nus_rn_src_stats.pt"}
CACHE_VIT=${CACHE_VIT:-"$WORK/cache_tsml_nus_vit_src_stats.pt"}
SEED=${SEED:-0}
ARCHS=${ARCHS:-"resnet50 vit"}
CORRS=${CORRS:-"gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"}

export NUS_CLEAN_ROOT="$CLEAN_ROOT"

cd "$WORK"

for f in run_real.py config.py real_model.py tta.py mldata.py \
         utils_tools_port.py "$CKPT_RN" "$CKPT_VIT"; do
    [ -e "$f" ] || { echo "MISSING: $f  (放到本脚本同目录, 或用环境变量覆盖)"; exit 1; }
done
[ -d "$DATA_ROOT/brightness_5/NUSWIDE" ] || {
    echo "MISSING data root: $DATA_ROOT  (用 DATA_ROOT=... 覆盖)"; exit 1; }

echo "===== OURS NUS-WIDE shuffle arms (seed=$SEED, archs: $ARCHS) ====="
echo "start: $(date)"

for arch in $ARCHS; do
    if [ "$arch" = "resnet50" ]; then
        TAG="ResNet-NUS"; LOG_DIR="$WORK/logs_ours_rn_nus_shuf"
        ARCH_FLAGS="--arch resnet50 --bn_tent --checkpoint $CKPT_RN --cache $CACHE_RN"
    else
        TAG="ViT-NUS"; LOG_DIR="$WORK/logs_ours_vit_nus_shuf"
        ARCH_FLAGS="--arch vit --no_refresh_weak --bn_warmup 0 --checkpoint $CKPT_VIT --cache $CACHE_VIT"
    fi
    mkdir -p "$LOG_DIR"
    echo ""
    echo "===== OURS $TAG 乱序对照 (15 项, 长流适配配置 + --shuffle) ====="
    for corr in $CORRS; do
        LOG="$LOG_DIR/$corr.log"
        if [ -f "$LOG" ] && grep -q "full TTA" "$LOG"; then
            echo "[skip] $corr : $(grep 'full TTA' "$LOG" | tail -1)"
            continue
        fi
        echo "--- $TAG $corr : $(date +%H:%M) ---"
        $PY -u run_real.py $ARCH_FLAGS \
            --dataset_name nuswide \
            --source_dir "$SRC_DIR" --test_root "$DATA_ROOT" \
            --seed "$SEED" --methods full --shuffle \
            --pi_freeze \
            --ema 0.999 --rebalance --pi_hard --pi_floor 0.5 \
            --tau_adaptive --rho 0.10 --proto_confirm \
            --struct_form corr --lambda_struct 0.25 \
            --corruptions "$corr" 2>&1 | tee "$LOG" > /dev/null
        grep "full TTA" "$LOG" | tail -1
        grep "F1@0.5" "$LOG" | tail -1
    done
done

# ---------------- 汇总 ----------------
echo ""
echo "===== shuffle-arm summary (mAP %) ====="
echo "  参照 (分块流, 实验5):  ViT 17.72 (-2.42, 零样本 20.14)"
echo "                          RN  9.52  (-2.68, 零样本 12.20)"
echo "  参照 (乱序试点, 实验35 §3.5): RN brightness 29.210 (+0.93) / jpeg 20.297 (+0.81)"
for arch in $ARCHS; do
    if [ "$arch" = "resnet50" ]; then
        TAG="RN";  LOG_DIR="$WORK/logs_ours_rn_nus_shuf"
    else
        TAG="ViT"; LOG_DIR="$WORK/logs_ours_vit_nus_shuf"
    fi
    total=0; n=0
    for corr in $CORRS; do
        LOG="$LOG_DIR/$corr.log"
        v=$(grep -aoE "[0-9]+\.[0-9]+ +\[full TTA\]" "$LOG" 2>/dev/null | grep -aoE "[0-9]+\.[0-9]+" | head -1)
        if [ -n "$v" ]; then
            echo "  $TAG $corr : $v"
            total=$(awk "BEGIN{print $total + $v}"); n=$((n+1))
        fi
    done
    [ $n -gt 0 ] && echo "  $TAG MEAN: $(awk "BEGIN{printf \"%.3f\", $total/$n}")  ($n/15)"
done
echo "done: $(date)"
