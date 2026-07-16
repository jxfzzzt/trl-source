# dpo_example.py
# 用于调试 DPO(Direct Preference Optimization) 算法执行流程的最小示例脚本。
# 想看清 DPO 的核心逻辑，建议在下面这几个位置打断点(方法名对应当前仓库版本)：
#   1. trl/trainer/dpo_trainer.py -> DPOTrainer._prepare_dataset (内部 tokenize_fn)  数据预处理成 prompt/chosen/rejected ids
#   2. trl/trainer/dpo_trainer.py -> DataCollatorForPreference.torch_call           chosen/rejected 拼成一个 batch 并 padding
#   3. trl/trainer/dpo_trainer.py -> DPOTrainer.compute_ref_log_probs              reference 模型前向, 取 ref_chosen/rejected_logps
#   4. trl/trainer/dpo_trainer.py -> DPOTrainer._compute_loss                      policy 前向 + 核心 loss(F.logsigmoid(beta * (pi_logratios - ref_logratios)))
#   5. trl/trainer/dpo_trainer.py -> DPOTrainer.compute_loss                       Trainer 的入口, 转发到 _compute_loss / _compute_loss_liger
import os
import sys
from pathlib import Path

# export HF_HOME=/root/autodl-tmp/model_weights
os.environ["HF_HOME"] = "/root/autodl-tmp/model_weights"

sys.path.insert(0, str(Path(os.path.join(os.path.dirname(__file__), '..')).resolve()))
sys.path.insert(0, str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
import torch

# 加载数据集：trl-lib/ultrafeedback_binarized 是 TRL 官方 DPO 示例最常用的偏好数据集，
# 已经是标准的 preference 格式(每条样本含 chosen / rejected 两个对话)，可直接喂给 DPOTrainer。
# 为了方便调试，这里只取训练集前 256 条、测试集前 32 条，跑起来更快。
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:256]")
eval_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test[:32]")

training_args = DPOConfig(
    output_dir="output/Qwen2.5-0.5B-DPO",
    learning_rate=5e-6,          # DPO + LoRA 常用学习率
    beta=0.1,                    # DPO 的核心超参: 控制与 reference 模型的偏离程度, 越大越保守
    loss_type=["sigmoid"],       # 标准 DPO loss(可换成 'ipo'、'hinge' 等对比不同变体)
    logging_steps=1,             # 每步都打日志, 方便观察 loss / rewards 变化
    gradient_accumulation_steps=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    max_length=1024,             # 整条(prompt+response)截断长度, 调小可加速调试
    max_steps=20,                # 只跑少量 step, 调试用
    eval_strategy="steps",
    eval_steps=10,
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    report_to="none",            # 调试时关闭 wandb / trackio 等上报
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # 使用 LoRA 时无需单独的 reference 模型: DPO 会在计算 ref_logps 时临时关闭 adapter,
    # 用同一个基座模型充当 reference, 省显存。若想显式使用独立 reference 模型, 可删掉 peft_config
    # 并传入 ref_model=... (不传则 DPOTrainer 会自动 deepcopy 一份 policy 模型作为 reference)。
    peft_config=LoraConfig(task_type="CAUSAL_LM"),
)

trainer.train()
trainer.save_model(training_args.output_dir)
