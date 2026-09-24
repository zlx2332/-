# NUS-WIDE 乱序（打乱顺序的测试流）实验 — AutoDL 部署说明

对应实验：教师-学生框架 NUS-WIDE 乱序对照，补齐实验5/32 主表的"随机打乱的流"行。
代码 = 本地 `mlabel_tta/` 当前版本（与实验35 乱序试点同一套，`--shuffle` 已在 `run_real.py` 落地）。

## 一、要上传的文件（9 个，全部覆盖云端旧版）

```
run_real.py  tta.py  config.py  real_model.py
utils_tools_port.py  mldata.py  run_ours_nus_shuf.sh
planA_rn_nus.py  run_plana_rn_nus_shuf.sh
```
（前 7 个属教师-学生管线，后 2 个属 SLEB 管线，`real_model/utils_tools_port` 两管线共用；另需两个检查点，见下。）

- 检查点：`resnet50_nuswide_best.pth`、`vit_nuswide_best.pth`，放脚本同目录。若实例里 `/root/autodl-tmp/ML-TTA/` 下已有，可用环境变量指过去，免上传：
  `CKPT_RN=/root/autodl-tmp/ML-TTA/resnet50_nuswide_best.pth CKPT_VIT=/root/autodl-tmp/ML-TTA/vit_nuswide_best.pth`

⚠️ 覆盖后修行尾（Windows CRLF 会害死 bash）：

```bash
sed -i 's/\r$//' run_ours_nus_shuf.sh
```

## 二、服务器数据布局（沿用现有 AutoDL，无需新数据）

```
/root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted/{corr}_5/NUSWIDE   # 15 种损坏流
/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr/<concept>/<file>      # 干净图(源统计)
```

源统计缓存自动复用（`cache_tsml_nus_rn_src_stats.pt` / `cache_tsml_nus_vit_src_stats.pt`，同目录或旧跑生成过即可；不在则首个项目先花几分钟重建，日志打印 "computing over 30000 images" 即正常）。路径不同就环境变量覆盖：`DATA_ROOT / CLEAN_ROOT / PY`（见脚本头注释）。

## 三、冒烟 + 启动

```bash
# 冒烟: 1024 张子集, 走一遍 --shuffle 全路径, ~2 分钟 (只验部署, 不出结论)
# 注意 --source_dir 和 --test_root 都必须显式传云端路径
# (不传会落到 Windows 默认值报 FileNotFoundError)
PY=python NUS_CLEAN_ROOT=/root/autodl-tmp/ML-TTA/DATASETS5/nuswide/Flickr \
python -u run_real.py --arch resnet50 \
  --checkpoint /root/autodl-tmp/ML-TTA/resnet50_nuswide_best.pth \
  --dataset_name nuswide \
  --source_dir /root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted/gaussian_noise_5/NUSWIDE \
  --test_root /root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted \
  --limit 1024 --source_limit 2048 --cache /tmp/smoke_stats.pt \
  --methods full --shuffle --seed 0 --bn_tent \
  --pi_freeze --ema 0.999 --rebalance --pi_hard --pi_floor 0.5 \
  --tau_adaptive --rho 0.10 --proto_confirm \
  --struct_form corr --lambda_struct 0.25 --corruptions gaussian_noise
# 若开头报 libgomp: Invalid value for OMP_NUM_THREADS (无害噪音),
# 先 export OMP_NUM_THREADS=1 清掉再跑

# 正式启动 (ResNet 15 项 → ViT 15 项, 当前实例合计约 6.5 小时)
nohup bash run_ours_nus_shuf.sh > ours_nus_shuf_log.txt 2>&1 &
tail -f ours_nus_shuf_log.txt

# 只跑单骨干 (省时): 只跑 ResNet
ARCHS="resnet50" nohup bash run_ours_nus_shuf.sh > ours_nus_shuf_log.txt 2>&1 &
```

时间参考（107,859 张/项，当前实例实测锚点）：ResNet 约 475 秒/项（15 项约 2 小时）；ViT 约为其 2.2 倍（冒烟比例），估约 17 分钟/项（15 项约 4.5 小时，待实测确认）。首项若需重建源统计会多花几分钟。注意：实验35 旧试点日志显示 1056 秒/项，那是上一次实例的速度，与本次不可比；旧日志 brightness/jpeg 建议移走备份后重跑，保证主表 15 项同条件（旧值 29.210/20.297 与新值的偏差即跨实例噪声带）。先跑 ResNet（核心行），ViT 为"LayerNorm 免疫流结构"的验证对照。

断点续跑：每项完成即写日志，中断后重新 `nohup` 同一命令，已完成项自动跳过；实验35 乱序试点已完成的 ResNet brightness / jpeg（旧日志 `logs_ours_rn_nus_shuf/` 若还在）同样自动跳过。

## 四、判定锚点（对照分块流，实验5/32）

| 设置 | ResNet-NUS | ViT-NUS |
|---|---|---|
| 零样本（与流序无关，不重跑） | 12.20 | 20.14 |
| 分块流 full（实验5 主表） | 9.52（−2.68） | 17.72（−2.42） |
| 乱序试点（实验35 §3.5，仅 2 项） | brightness 29.210（+0.93）/ jpeg 20.297（+0.81） | 未测 |

读法：
1. **ResNet 乱序均值**：接近甚至超过零样本 12.20 → 与 SLEB 同判（负增益全额是流结构税，方法本身在随机打乱的流上为正）；仍明显为负 → 教师质量/伪标签另有份额。
2. **ViT 乱序均值**：与分块流 17.72 同分（±0.3 噪声带）→ "LayerNorm 免疫流结构"成立，ViT 的 −2.42 归教师质量+长流；显著高于 17.72 → ViT 侧批级机制（τ 分位数、Σ 协方差 EMA）也吃了分块流的税，归因链需修订。

## 五、SLEB ResNet 乱序重跑（第二个队列，等教师-学生队列跑完再启）

```bash
# 冒烟 (2000 张, ~1 分钟)
SMOKE=1 bash run_plana_rn_nus_shuf.sh

# 教师-学生队列跑完后启动 (主日志 ours_nus_shuf_log.txt 出现 done: 即全部完成)
nohup bash run_plana_rn_nus_shuf.sh > plana_rn_nus_shuf_log.txt 2>&1 &
tail -f plana_rn_nus_shuf_log.txt

# 或者挂一条自动接力 (每 5 分钟检查一次, 教师-学生一完成就自动开跑)
cd /root/autodl-tmp/ML-TTA
nohup bash -c 'while ! grep -q "^done:" ours_nus_shuf_log.txt 2>/dev/null; do sleep 300; done; bash run_plana_rn_nus_shuf.sh' > plana_rn_nus_shuf_log.txt 2>&1 &
```

要点：
- 目的：实验35 §3.4 的 D 对照只测了 brightness/jpeg 两项（翻正 +4.70/+4.62，超零样本），本队列补全 SLEB 的乱序行 15 项，且与教师-学生乱序实验**同实例同条件**，主表两方法乱序行可直接对比；
- 配置与 `run_plana_rn_nus.sh` 的 tn 范式逐字一致（超参全默认：lr 1e-5、margin 0.7、warmup 5、N0 10 等），唯一差别 `--shuffle --seed 0`；
- 实验35 的 D 对照旧日志（`logs_plana_rn_nus_ctrl/`，brightness 32.978 / jpeg 24.104）**保留不动**，本队列用新目录 `logs_plana_rn_nus_shuf/` 全部重跑——新值与旧值的偏差即跨实例噪声带（同种子同配置）；
- 零样本与流序无关，但本队列**同实例复测**（`--zero_shot`）：汇总自带逐项 零样本 / tn / Δ（与 `run_plana_rn_nus.sh` 及实验7 主表同口径），新零样本均值与旧值 12.20 的对照即跨实例噪声带；
- 时长估 1~1.5 小时（含零样本前向；SLEB 单模型，比教师-学生快得多）。

## 六、结果回传

跑完把 `ours_nus_shuf_log.txt`（含 summary）或 `logs_ours_rn_nus_shuf/`、`logs_ours_vit_nus_shuf/` 下的日志，加上 `plana_rn_nus_shuf_log.txt`（或 `logs_plana_rn_nus_shuf/`）发我即可（关键行：教师-学生每项的 `[full TTA]` mAP、`F1@0.5`；SLEB 每项的 `mode=tn] mAP:`）。我会：
1. 出乱序主表：两方法 × 两骨干 × 15 项（零样本 / 乱序 / Δ，对照分块流逐项 Δ），含逐项用时；
2. 对判定逐条出结论（ResNet 乱序均值 vs 零样本 12.20；ViT 乱序 vs 分块 17.72；SLEB 乱序 vs 教师-学生乱序）；
3. 按统一声明写成实验7 报告，存实验报告文件夹。
