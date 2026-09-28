import os, torch
import numpy as np

base = r'E:\mttta\DATASETS5\nuswide'
train_list = os.path.join(base, 'ImageList', 'TrainImagelist.txt')
test_list  = os.path.join(base, 'ImageList', 'TestImagelist.txt')
label_dir  = os.path.join(base, 'Groundtruth', 'TrainTestLabels')
out_dir    = base

# 读取图像列表
with open(train_list) as f:
    train_imgs = [line.strip() for line in f]
with open(test_list) as f:
    test_imgs = [line.strip() for line in f]

num_train = len(train_imgs)
num_test  = len(test_imgs)

# 获取所有概念名称（去掉前缀和后缀）
concepts = sorted([f.replace('Labels_','').replace('_Train.txt','')
                   for f in os.listdir(label_dir) if f.endswith('_Train.txt')])
num_classes = len(concepts)
print(f'概念数: {num_classes}, 训练图片: {num_train}, 测试图片: {num_test}')

# 读取训练标签
train_labels = np.zeros((num_train, num_classes), dtype=np.int32)
for i, concept in enumerate(concepts):
    lbl_file = os.path.join(label_dir, f'Labels_{concept}_Train.txt')
    with open(lbl_file) as f:
        for j, line in enumerate(f):
            train_labels[j, i] = int(line.strip())

# 读取测试标签
test_labels = np.zeros((num_test, num_classes), dtype=np.int32)
for i, concept in enumerate(concepts):
    lbl_file = os.path.join(label_dir, f'Labels_{concept}_Test.txt')
    with open(lbl_file) as f:
        for j, line in enumerate(f):
            test_labels[j, i] = int(line.strip())

# 写入 CSV
def write_csv(img_list, labels, filename):
    with open(filename, 'w') as f:
        for img_path, lbl in zip(img_list, labels):
            full_path = os.path.join(base, img_path.replace('/', os.sep))
            label_str = ','.join(str(v) for v in lbl)
            f.write(f'{full_path},{label_str}\n')

print('生成 train.csv ...')
write_csv(train_imgs, train_labels, os.path.join(out_dir, 'train.csv'))
print('生成 test.csv ...')
write_csv(test_imgs, test_labels, os.path.join(out_dir, 'test.csv'))

# 重新计算并保存先验
label_sum = train_labels.sum(axis=0).astype(np.float32)
pi_src = (label_sum + 1.0) / (num_train + 2.0)
np.save(os.path.join(out_dir, 'pi_src.npy'), pi_src)
print('已更新 pi_src.npy')

print('完成！请重新运行 TTA 脚本。')