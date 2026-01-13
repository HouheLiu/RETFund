# RETFund + LoRA 微调（ADAMD 二分类）

> 目标：在冻结原始 RETFund 权重的前提下，使用 LoRA 对 AMD 二分类进行微调。

## 1. 数据集准备（ADAMD）

1. 获取 ADAMD 数据集并整理成统一结构：
   - `train/`、`val/`、`test/` 三级目录
   - 每个目录下按类别子文件夹组织，例如 `AMD/` 和 `Normal/`
2. 建议在 `data/adamd/` 下保存原始数据与 split 后的数据：
   - `data/adamd/raw/`：原始数据
   - `data/adamd/splits/{train,val,test}/`：划分后的数据
3. 记录类别映射：
   - `AMD -> 1`
   - `Normal -> 0`

## 2. 冻结原模型 + LoRA 思路

- 冻结 RETFund 的全部 backbone 参数，只在 LoRA 适配层训练。
- LoRA 一般插入到注意力层的 `q/k/v` 或者 `q/v` 投影层上。
- 分类头（如果是新建的线性层）通常需要参与训练。

## 3. 建议的 LoRA 超参

- `r`（秩）：8 或 16
- `alpha`：16 或 32
- `dropout`：0.05 ~ 0.1
- `target_modules`：`["q_proj", "v_proj"]`（具体名称需与模型实现一致）

## 4. 训练流程示例（概念性）

> 以下为通用流程示例，具体训练脚本/命令需按项目结构调整。

1. 加载 RETFund 预训练权重
2. 冻结 backbone
3. 注入 LoRA
4. 替换分类头（2 类）
5. 训练 LoRA + 分类头

## 5. 评估与指标

- 主要指标：AUC、Accuracy、Sensitivity、Specificity
- 建议保存：
  - 最佳 `val` AUC 的 checkpoint
  - 最终 `test` 集评估报告

## 6. 结果记录建议

- 模型版本（RETFund checkpoint 版本）
- LoRA 超参（r/alpha/dropout/target_modules）
- 训练配置（batch size、lr、epoch、优化器）
- 数据划分版本（随机种子）

