# 方案A第1步：从官方 train.csv 分层切出选优验证集
# 固定种子，按"首个正标签概念"分层，切约 8000 张做选优验证，其余训练
# 输出：train_select.csv / val_select.csv（与 train.csv 同格式）
# 运行：E:/anaa/envs/pytorch/python.exe split_trainval_nuswide.py
import os
import random
from collections import defaultdict

SRC       = r'E:\mttta\DATASETS5\nuswide\train.csv'
OUT_TRAIN = r'E:\mttta\DATASETS5\nuswide\train_select.csv'
OUT_VAL   = r'E:\mttta\DATASETS5\nuswide\val_select.csv'
SEED      = 42
VAL_TARGET = 8000

def main():
    lines = [l.strip() for l in open(SRC) if l.strip()]
    print(f'官方训练集: {len(lines)} 行')

    # 分层键：首个正标签概念（0-80）
    strata = defaultdict(list)
    for i, line in enumerate(lines):
        parts = line.split(',')
        labels = [float(x) for x in parts[1:]]
        first_pos = next((c for c, v in enumerate(labels) if v >= 0.5), 0)
        strata[first_pos].append(i)
    print(f'分层概念数: {len(strata)}')

    # 只从 >=2 张的概念里抽验证（保证每个用到的概念在训练侧至少留 1 张）
    eligible = {k: v for k, v in strata.items() if len(v) >= 2}
    total_elig = sum(len(v) for v in eligible.values())

    # 按比例分配（最大余数法凑满 8000）
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
        for i, line in enumerate(lines):
            if i not in val_idx:
                f.write(line + '\n')
    with open(OUT_VAL, 'w') as f:
        for i, line in enumerate(lines):
            if i in val_idx:
                f.write(line + '\n')

    # ---- 校验与统计 ----
    train_idx = [i for i in range(len(lines)) if i not in val_idx]
    def class_pos(idxs):
        cnt = [0] * 81
        for i in idxs:
            labs = [float(x) for x in lines[i].split(',')[1:]]
            for c, v in enumerate(labs):
                if v >= 0.5:
                    cnt[c] += 1
        return cnt
    tp, vp = class_pos(train_idx), class_pos(val_idx)
    zero_train = [c for c in range(81) if tp[c] == 0]
    zero_val   = [c for c in range(81) if vp[c] == 0]
    ratios = [vp[c] / (tp[c] + vp[c]) for c in range(81) if tp[c] > 0]
    n_lab_t = sum(tp) / len(train_idx)
    n_lab_v = sum(vp) / len(val_idx)

    print(f'训练子集: {len(train_idx)} 张 -> {OUT_TRAIN}')
    print(f'选优验证集: {len(val_idx)} 张 -> {OUT_VAL}')
    print(f'训练侧 0 正样本的类别数: {len(zero_train)}（应为 0）')
    print(f'验证侧 0 正样本的类别数: {len(zero_val)}（允许，长尾类）')
    print(f'验证占比（按类别正样本计）: min {min(ratios):.3f} / mean {sum(ratios)/len(ratios):.3f} / max {max(ratios):.3f}')
    print(f'平均每图正标签: 训练 {n_lab_t:.2f} / 验证 {n_lab_v:.2f}')

if __name__ == '__main__':
    main()
