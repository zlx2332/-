# 源模型训练与切分脚本说明（VOC / COCO / NUS 三数据集）

整理日期：2026-09-28（当晚更新：NUS 口径决策变更为对齐 BCP 阵营，见第三节）。

## 一、文件清单

| 文件 | 数据集 | 用途 |
|---|---|---|
| `train_nuswide_v2.py` | NUS-WIDE | 2×2 统一训练（自包含；既训练官方切分版也训练净化版，csv 由参数指定） |
| `split_trainval_nuswide.py` | NUS-WIDE | 官方切分链的选优切分（既有检查点的产地） |
| `filter_and_split_clean.py` | NUS-WIDE | **净化链切分（新主路线）**：剔除无标注图后分层切分（BCP 阵营对齐） |
| `make_filtered_stream.py` | NUS-WIDE | **过滤测试流构建**：为每个损坏项生成 NUSWIDE_F 目录（软链资源 + 剔除无标注行的 test.csv），服务器上运行 |
| `train_coco_v2.py` | COCO 2017 | 2×2 统一训练（自包含；选优直接用官方 val2017，无需切分脚本） |
| `train_voc_asl.py` | VOC 2007 | 旧协议训练（当前 VOC 检查点的产地；依赖 `data/voc_custom.py`） |
| `data/voc_custom.py` | — | VOC 数据集类（`train_voc_asl.py` 的唯一外部依赖） |

三个 v2 训练脚本的核心配方逐字一致：ASL（γ⁻=4, γ⁺=1, clip=0.05）、AdamW lr=5e-4·wd=1e-4（ViT 为差分 lr：骨干 ×0.05）、batch 32、20 轮、AMP、seed 42；增强维度 = {标准增强（RandomResizedCrop + RandomHorizontalFlip(0.5) + ColorJitter(0.2,0.2,0.2)），无增强（Resize BICUBIC）}；逐轮保存、按选优验证 mAP 取最优、官方测试集不参与任何模型选择。

## 二、各数据集的数据流与口径事实

### NUS-WIDE（81 类）——双轨

**既有链（官方切分，既有检查点与已跑数字的产地，转为对照存档）**：
```
官方 train.csv（161,789，含 36,340 张全零标注图）
  --split_trainval_nuswide.py--> train_select（153,789）+ val_select（8,000）
  --train_nuswide_v2.py--> rn/vit_224_aug_bestval.pth 等（官方切分版检查点）
测试：全量流 107,859 适应 + 代码内过滤 83,898 评分
```

**净化链（新主路线，对齐 BCP/PVLR/BEM 阵营）**：
```
官方 train.csv --filter_and_split_clean.py（剔全零→125,449，分层切 8,000）-->
  train_select_clean（117,449）+ val_select_clean（8,000）
  --train_nuswide_v2.py（--train_csv/--val_csv 指向净化版）--> 净化版检查点（输出目录 2x2_ckpt_clean）
测试：make_filtered_stream.py 生成 NUSWIDE_F 过滤流目录（83,898 张流过）
  → 全方法 --data 指向 NUSWIDE_F 重跑（适应与评分同为 83,898）
```

**迁移分两阶段**：阶段 A = 净化 RN 训练（验证质量不降）+ 过滤流对照（量化测试侧差异，约 3~4 小时）；阶段 B = 净化 ViT 训练 + Fisher 重算 + NUS 全方法过滤流重跑（约一天）。

### COCO 2017（80 类）
```
train2017（118,287，全部有标注） --train_coco_v2.py（选优验证 = 官方 val2017 5,000）--> 2×2 检查点
```
- train2017 每张图都有标注，无训练侧过滤问题；
- 评估：**全量 val2017 5,000**（ASL/BEM 等文献标准；48 张无标注图计入负例池，与过滤口径 4,952 差 <0.1 且 Δ 不变）。

### VOC 2007（20 类）
```
trainval2007（5,011，全部有标注） --train_voc_asl.py（旧协议）--> 旧检查点
```
- 全部有标注，无过滤问题；评估 = test 4,952 全量；
- `train_voc_asl.py` 为旧协议（无选优切分/无训练打乱）；VOC 新代待按 v2 协议重训。

## 三、NUS-WIDE 切分阵营与本项目口径决策

| 切分体系 | 定义 | 使用者 |
|---|---|---|
| 官方切分 | 161,789 / 107,859（训练侧 125,449 有标注 + 36,340 无标注；测试侧含 23,961 无标注） | CapNet、PanCAN（**本项目既有链，转存档对照**） |
| **官方过滤切分** | 剔除无标注图、归属不变（**125,449 / 83,898**） | **PVLR、BEM、BCP（本项目新主路线，对齐目标）** |
| ASL 变体 | 约 22 万张可下载图随机 70/30 | ASL、Q2L、MlTr |
| SRN 策划切分 | 约 15 万 / 6 万 | VLPL、TRM-ML、PIAA |
| 零样本 | 官方测试集、不训练 | TaI-DPT、T2I-PAL |

**决策（2026-09-28）**：与 BCP 阵营完整对齐——训练侧净化（125,449）、测试侧过滤流（83,898 流过 + 评分）。既有官方切分系的检查点与全部 TTA 数字转为存档，并作为"官方切分 vs 过滤切分"的对照数据保留（论文附录素材）。

## 四、无标注图与适应流的关系（原理说明）

- **既有协议**（官方切分 + 代码内评估过滤）：适应集 107,859、评分集 83,898——无标注图全部参与适应（BN 批统计前向路径不可挡；SLEB mask 判据为模型置信度而非标注有无）、完全不参与评分。部署语义正确（适应时无标签不可知，用标注筛流 = 信息泄露），但与 BCP 阵营的流组成不同（对方流内无无标注图）。
- **净化协议**（新主路线）：适应与评分同为 83,898——流组成与 BCP/BEM 一致，跨阵营数字严格同协议。
- 两协议评估子集相同（83,898），差异仅在适应流的 22% 概念外图（量级预计 <0.5，由过滤流对照实验量化）。

## 五、历史检查点对应关系

- NUS 官方切分版：`rn_224_aug_bestval.pth`、`vit_224_aug_bestval.pth` 等 ← `train_nuswide_v2.py` + `split_trainval_nuswide.py`（存于 `E:\mttta\2x2_ckpt\`）；
- NUS 净化版（待产）：同脚本 + `filter_and_split_clean.py`，输出目录 `2x2_ckpt_clean/`（服务器 `/root/autodl-tmp/2x2_ckpt_clean`）；
- COCO：`coco_rn_224_aug_bestval.pth` 等 ← `train_coco_v2.py`；
- VOC：旧检查点 ← `train_voc_asl.py`。

## 六、命令示例

```bash
# ---- NUS 净化链（新主路线）----
# 本地：净化切分
E:/anaa/envs/pytorch/python.exe filter_and_split_clean.py
#   -> train_select_clean.csv / val_select_clean.csv（上传 DATASETS5/nuswide/）
# 服务器：过滤流目录（每损坏项建 NUSWIDE_F）
python make_filtered_stream.py --base /root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted
# 服务器：净化训练（输出 2x2_ckpt_clean）
python -u train_nuswide_v2.py --arch resnet50 --size 224 --aug \
  --train_csv DATASETS5/nuswide/train_select_clean.csv \
  --val_csv DATASETS5/nuswide/val_select_clean.csv \
  --flickr_root DATASETS5/nuswide/Flickr --workers 8 \
  --out_dir /root/autodl-tmp/2x2_ckpt_clean
# TTA 跑批：--data 指向 .../${c}_5/NUSWIDE_F（过滤流）

# ---- NUS 官方切分链（既有，存档）----
E:/anaa/envs/pytorch/python.exe split_trainval_nuswide.py
E:/anaa/envs/pytorch/python.exe train_nuswide_v2.py --arch resnet50 --size 224 --aug

# ---- COCO ----
E:/anaa/envs/pytorch/python.exe train_coco_v2.py --arch resnet50 --size 224 --aug

# ---- VOC（旧协议）----
E:/anaa/envs/pytorch/python.exe train_voc_asl.py --arch resnet50 --data <VOCdevkit 路径>
```

切分文件按项目规则从本地上传，不在服务器重新生成；服务器运行训练需改数据路径与 `--workers 8`（默认为本地 Windows 路径）。
