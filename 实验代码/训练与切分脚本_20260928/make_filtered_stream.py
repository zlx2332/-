# 过滤测试流构建（服务器运行）：为每个损坏项生成 NUSWIDE_F 目录
# 内容：软链 Flickr/ 与 pi_src.npy（不占空间）+ 剔除无标注行的 test.csv（83,898 行）
# 用途：BCP 阵营对齐的过滤流协议——TTA 适应与评分同为 83,898 张
# 运行：python make_filtered_stream.py --base /root/autodl-tmp/ML-TTA/DATASETS5/nuswide_corrupted
import argparse
import os

CORRS = ("gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur "
         "zoom_blur snow frost fog brightness contrast elastic_transform "
         "pixelate jpeg_compression").split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="nuswide_corrupted 根目录")
    args = p.parse_args()

    for c in CORRS:
        src = os.path.join(args.base, f"{c}_5", "NUSWIDE")
        dst = os.path.join(args.base, f"{c}_5", "NUSWIDE_F")
        if not os.path.isdir(src):
            print(f"[skip] {src} 不存在")
            continue
        os.makedirs(dst, exist_ok=True)
        for name in ("Flickr", "pi_src.npy"):
            link = os.path.join(dst, name)
            if not os.path.exists(link):
                os.symlink(os.path.join(src, name), link)
        kept, dropped = [], 0
        with open(os.path.join(src, "test.csv"), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if not parts[0]:
                    continue
                if any(float(x) >= 0.5 for x in parts[1:]):
                    kept.append(line if line.endswith("\n") else line + "\n")
                else:
                    dropped += 1
        with open(os.path.join(dst, "test.csv"), "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"[ok] {c}: 保留 {len(kept)} / 剔除 {dropped} -> {dst}")


if __name__ == "__main__":
    main()
