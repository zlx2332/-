# 明日 NUS 三任务队列 — 服务器部署说明

对应消融：B2（非承诺软目标一致性）、ε-on-NUS（corr 分母收缩）、NUS contrast（崩溃态门控行为）。
代码 = 本地 `mlabel_tta/` 2026-09-16 晚版本；实验34（S_c 设计空间）+ 实验33（噪声下限）是背景。

## 一、要上传的文件（7 个，全部覆盖云端旧版）

```
run_real.py  tta.py  config.py  real_model.py
utils_tools_port.py  mldata.py  nus_tomorrow_queue.sh
```

⚠️ **云端旧版必须覆盖**：`cons_form / sc_confirm / sc_feat_only / cond_teacher_feat / corr_eps / drift_gate` 全部是 09-16 晚的新 CLI，旧 `run_real.py` 会 argparse 直接报错。改完名后建议先跑下面的冒烟（30 秒出结果）确认部署无误。

上传后修行尾（Windows CRLF 会害死 bash）：

```bash
sed -i 's/\r$/' nus_tomorrow_queue.sh
```

## 二、服务器数据布局（沿用现有 AutoDL，无需新数据）

```
/root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted/{corr}_5/NUSWIDE   # 损坏流
/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr/<concept>/<file>      # 干净图(源统计)
/root/autodl-tmp/ML-TTA/vit_nuswide_best.pth                           # ASL ckpt(81类)
/root/autodl-tmp/ML-TTA/cache_tsml_nuswide_src_stats.pt                # 源统计缓存
```

缓存若在（旧跑生成过）自动复用；不在则首个臂会先花几分钟重建（30k 张，日志打印 "computing over 30000 images" 即正常）。路径不同就环境变量覆盖：`DATA_ROOT / CLEAN_ROOT / CKPT / CACHE / PY`（见脚本头注释）。

## 三、冒烟 + 启动

```bash
# 冒烟: 1024 张子集, 全部走一遍新 CLI 路径, ~1 分钟 (只验部署, 不出结论)
# 注意 --source_dir 必须显式传云端路径 (不传会落到 Windows 默认值报
# FileNotFoundError); train.csv 从该目录读, 图片由 NUS_CLEAN_ROOT 重定向。
PY=python NUS_CLEAN_ROOT=/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr \
python -u run_real.py --arch vit --checkpoint /root/autodl-tmp/ML-TTA/vit_nuswide_best.pth \
  --dataset_name nuswide \
  --source_dir /root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted/gaussian_noise_5/NUSWIDE \
  --limit 1024 --source_limit 2048 --cache /tmp/smoke_stats.pt \
  --methods full --no_refresh_weak --bn_warmup 0 --pi_freeze --ema 0.999 \
  --rebalance --pi_hard --pi_floor 0.5 --tau_adaptive --rho 0.10 \
  --proto_confirm --struct_form corr --lambda_struct 0.25 \
  --corr_eps 0 --cons_form soft --corruptions gaussian_noise

# 正式启动 (5 臂, 4090 约 9-11 小时; 3060 约 11-13 小时)
nohup bash nus_tomorrow_queue.sh > nus_tomorrow.log 2>&1 &
tail -f nus_tomorrow.log
```

断点续跑：每臂完成写标记到 `marks_nus_tomorrow/`；中断后重新 `nohup` 同一命令，已完成臂自动跳过。

## 四、五臂与判定锚点

| # | 臂 | 判定 |
|---|---|---|
| T1 | `--cons_form soft` gaussian 全量 | **vs 15.24 / 13.86**：≥15.24 → 非承诺监督修复弱教师区成立（最有价值结局）；13.86–15.24 之间 → 部分修复，与门控组合空间打开；<13.86 → 软目标也被教师拖累，NUS 定论为安全回退（predict-only） |
| T2 | `--corr_eps 0.001` gaussian 全量 | **vs 13.86**：唯一故意开 ε 的臂。NUS 有 6/81 类协方差对角 < 1e-3（会真咬合，收缩近死类噪声行）——看 mAP 动不动 + struct 损失尾段是否下降（实验33 预测量级是噪声主导，若 ε 使其下降即佐证） |
| T3a | 基线 contrast | **vs 实验32 旧配置 5.27**：拿长流适配版的崩溃态无门基线（此前没有），供 T3b/c 成对比较 |
| T3b | `--sc_confirm` contrast | 崩溃态（零样本 5.84、决策集中度 0.504）上门控是否仍 −1 一档 |
| T3c | `--sc_feat_only` contrast | 同上；两个设计是否仍然同分（gaussian 上是 12.610/12.610） |

日志关键行：每 50 批的 `drift/conc/proto_drift`（健康信号），流末 `[safe-fallback] never triggered`（drift_gate=10 只记录，不应触发；若触发说明决策分布漂移超预期，单看日志判定）。

## 五、结果回传

跑完把 `nus_tomorrow.log` 发我即可（含全部 5 臂 mAP / F1 / tail-run / 健康信号）。我会：
1. 对 5 个判定逐条出结论；
2. T3 三臂补进实验34 §3.1b 的 contrast 列；
3. 视结果决定是否需要 ε 单独成节、以及"门控×崩溃态"是否进主表。
