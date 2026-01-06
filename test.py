import sys
sys.path.append('/mnt/data_7tb/selena/projects/EncDiff')

from ldm.data.disdata import Cars3DTrain

# 测试加载数据集
print("Loading Cars3D dataset...")
dataset = Cars3DTrain()
print(f"✅ Dataset size: {len(dataset)}")
print(f"✅ Data shape: {dataset.data.shape}")

# 测试获取一个样本
sample = dataset[0]
print(f"✅ Sample keys: {sample.keys()}")
print(f"✅ Image shape: {sample['image'].shape}")
print("✅ Cars3D dataset loaded successfully!")