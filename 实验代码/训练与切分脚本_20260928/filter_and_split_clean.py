# 净化版数据处理（新主路线：对齐 BCP/PVLR/BEM 官方过滤切分阵营）
# 默认：剔除 train.csv 的全零标注行 -> train_clean_full.csv（125,449 张，全量训练，不切选优验证）
#       ——与阵营逐字对齐：训练量 125,449、无选优切分、固定 20 轮末轮出模（逐轮检查点仍保存备查）
# --split：额外输出分层切分版（117,449 训练 + 8,000 选优验证）——对照分析用，非主路线
# 训练用法（全量版）：train_nuswide_v2.py --train_csv train_clean_full.csv --val_csv train_clean_full.csv
#       （val_csv 传同一文件：无选优验证时逐轮 mAP 仅作监控记录，不用于选择，末轮出模）
# 运行：E:/anaa/envs/pytorch/python.exe filter_and_split_clean.py [--split]
import argparse
import random
from collections import defaultdict

SRC  = r'E:\mttta\DATASETS5\nuswide\train.csv'
OUT_FULL  = r'E:\mttta\DATASETS5\nuswide\train_clean_full.csv'      # 主路线：125,449 全量训练
OUT_TRAIN = r'E:\mttta\DATASETS5\nuswide\train_select_clean.csv'    # 对照版（--split 时更新）
OUT_VAL   = r'E:\mttta\DATASETS5\nuswide\val_select_clean.csv'
SEED = 42
VAL_TARGET = 8000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', action='store_true',
                    help='额外输出分层切分版（117,449+8,000，对照分析用）')
    args = ap.parse_args()

    lines = [l.strip() for l in open(SRC) if l.strip()]
    total = len(lines)

    keep_lines = []
    for line in lines:
        labels = [float(x) for x in line.split(',')[1:]]
        if any(v >= 0.5 for v in labels):
            keep_lines.append(line)
    dropped = total - len(keep_lines)
    print(f'官方训练集: {total} 行 -> 剔除全零 {dropped} 行（{dropped/total:.1%}）-> 保留 {len(keep_lines)} 行')

    with open(OUT_FULL, 'w') as f:
        f.writelines(l + '\n' for l in keep_lines)
    print(f'净化全量训练集（主路线，BCP 阵营对齐）: {len(keep_lines)} 张 -> {OUT_FULL}')
    print('训练命令：')
    print(r'  python train_nuswide_v2.py --arch resnet50 --size 224 --aug '
          f'--train_csv {OUT_FULL} --val_csv {OUT_FULL} '
          r'--flickr_root <Flickr 路径> --out_dir <净化输出目录>')

    if not args.split:
        return

    # ---- 对照版：与 split_trainval_nuswide.py 相同的分层切分 ----
    strata = defaultdict(list)
    for i, line in enumerate(keep_lines):
        labels = [float(x) for x in line.split(',')[1:]]
        first_pos = next((c for c, v in enumerate(labels) if v >= 0.5), 0)
        strata[first_pos].append(i)

    eligible = {k: v for k, v in strata.items() if len(v) >= 2}
    total_elig = sum(len(v) for v in eligible.values())
    raw = {k: VAL_TARGET * len(v) / total_elig for k, v in eligible.items()}
    alloc = {k: min(int(raw[k]), len(eligible[k]) - 1) for k in eligible}
    rem = VAL_TARGET - sum(alloc.values())
    order = sorted(eligible, key=lambda k: raw[k] - alloc[k], reverse=True)
    for k in order:
        if rem <= 0:
            break
        if alloc[k] < len(eligible[k]) - 1:
            alloc[k] += 1
            rem -= 1
    assert rem == 0 and sum(alloc.values()) == VAL_TARGET, f'分配未凑满: 剩余 {rem}'

    rng = random.Random(SEED)
    val_idx = set()
    for k, idxs in eligible.items():
        val_idx.update(rng.sample(idxs, alloc[k]))

    with open(OUT_TRAIN, 'w') as f:
        for i, line in enumerate(keep_lines):
            if i not in val_idx:
                f.write(line + '\n')
    with open(OUT_VAL, 'w') as f:
        for i, line in enumerate(keep_lines):
            if i in val_idx:
                f.write(line + '\n')
    print(f'对照切分版: 训练 {len(keep_lines) - len(val_idx)} 张 -> {OUT_TRAIN}')
    print(f'            选优验证 {len(val_idx)} 张 -> {OUT_VAL}')


if __name__ == '__main__':
    main()

