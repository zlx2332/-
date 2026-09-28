# 净化版切分（新主路线：对齐 BCP/PVLR/BEM 官方过滤切分阵营）
# 剔除 train.csv 的全零标注行后按"首个正概念"分层切 8,000 张选优验证
# 输出：train_select_clean.csv / val_select_clean.csv（与原切分同格式、同分层逻辑、同种子）
# 用途：净化版检查点训练（train_nuswide_v2.py --train_csv/--val_csv 指向这两个文件，
#       输出目录建议 2x2_ckpt_clean，与官方切分版检查点隔离）
# 运行：E:/anaa/envs/pytorch/python.exe filter_and_split_clean.py
import random
from collections import defaultdict

SRC       = r'E:\mttta\DATASETS5\nuswide\train.csv'
OUT_TRAIN = r'E:\mttta\DATASETS5\nuswide\train_select_clean.csv'
OUT_VAL   = r'E:\mttta\DATASETS5\nuswide\val_select_clean.csv'
SEED      = 42
VAL_TARGET = 8000

def main():
    lines = [l.strip() for l in open(SRC) if l.strip()]
    total = len(lines)

    # 第一步：剔除全零标注行（81 维全 0 = 无概念标注）
    keep_lines = []
    for line in lines:
        labels = [float(x) for x in line.split(',')[1:]]
        if any(v >= 0.5 for v in labels):
            keep_lines.append(line)
    dropped = total - len(keep_lines)
    print(f'官方训练集: {total} 行 -> 剔除全零 {dropped} 行（{dropped/total:.1%}）-> 保留 {len(keep_lines)} 行')

    # 第二步：与 split_trainval_nuswide.py 相同的分层切分（此时全零已不在，兜底分支不再触发）
    strata = defaultdict(list)
    for i, line in enumerate(keep_lines):
        labels = [float(x) for x in line.split(',')[1:]]
        first_pos = next((c for c, v in enumerate(labels) if v >= 0.5), 0)
        strata[first_pos].append(i)
    print(f'分层概念数: {len(strata)}')

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

    n_train = len(keep_lines) - len(val_idx)
    print(f'净化训练子集: {n_train} 张 -> {OUT_TRAIN}')
    print(f'净化选优验证: {len(val_idx)} 张 -> {OUT_VAL}')
    print('训练命令示例：')
    print(r'  python train_nuswide_v2.py --arch resnet50 --size 224 --aug '
          r'--train_csv E:\mttta\DATASETS5\nuswide\train_select_clean.csv '
          r'--val_csv E:\mttta\DATASETS5\nuswide\val_select_clean.csv --out_dir <净化输出目录>')

if __name__ == '__main__':
    main()
