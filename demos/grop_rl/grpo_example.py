# train_grpo.py
import os
import sys
from pathlib import Path
os.environ["HF_HOME"] = "/root/autodl-tmp/model_weights"

sys.path.insert(0, str(Path(os.path.join(os.path.dirname(__file__), '..')).resolve()))
sys.path.insert(0, str(Path(os.path.join(os.path.dirname(__file__), '..', '..')).resolve()))

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import torch

# Load the dataset
dataset = load_dataset("trl-lib/tldr", split="train")

training_args = GRPOConfig(
    output_dir="output/Qwen2-0.5B-GRPO",
    learning_rate=1e-4,
    logging_steps=10,
    gradient_accumulation_steps=2,
    max_completion_length=128,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    # generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
    #                       = 3 * 1 * gradient_accumulation_steps(2) = 6，必须能被 num_generations 整除
    num_generations=6,
    model_init_kwargs={"torch_dtype": torch.bfloat16},
    num_train_epochs=1,
    save_steps=1000,
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs="weqweasdas/RM-Gemma-2B",
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(task_type="CAUSAL_LM"),
)

trainer.train()
trainer.save_model(training_args.output_dir)