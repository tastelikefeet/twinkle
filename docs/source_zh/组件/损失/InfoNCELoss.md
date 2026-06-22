# InfoNCE 损失

`InfonceLoss` 实现带批内负样本和可选跨 rank 聚合的对比学习损失，用于嵌入/检索模型训练。

## 使用方法

```python
from twinkle.loss import InfonceLoss

loss_fn = InfonceLoss(
    temperature=0.1,
    use_batch=True,           # 启用批内负样本
    hard_negatives=7,         # 固定每样本负样本数
    mask_fake_negative=True,  # 遮蔽假负样本
)

model.set_loss(loss_fn)
```

## 输入格式

每个样本按 `锚点(1) + 正样本(1) + 负样本(n)` 排列。`inputs['labels']` 是一维掩码，`1` 标记每组的起始位置。

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `temperature` | float | 0.1 | 相似度缩放因子 |
| `use_batch` | bool | True | 使用跨样本批内负样本 |
| `hard_negatives` | int | None | 固定每样本负样本数（截断/上采样）|
| `mask_fake_negative` | bool | False | 遮蔽高于 positive + margin 的 logit |
| `include_qq` | bool | False | 添加 query-query 相似度块 |
| `include_dd` | bool | False | 添加 doc-doc 相似度块 |

## 跨 Rank 聚合

当 `use_batch=True` 且分布式训练激活时，嵌入会从所有 DP rank 聚合以最大化批内负样本多样性。仅本地分片保留梯度。
