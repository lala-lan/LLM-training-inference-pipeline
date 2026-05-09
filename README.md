# LLM-training-inference-pipeline
A modular LLM training &amp; inference pipeline supporting full fine-tuning (SFT), LoRA, DPO alignment, and pretraining — built on HuggingFace Transformers + TRL + PEFT, with a unified JSON-driven CLI.
LLM Train-Inference Pipeline

一套完整的 LLM 训练与推理流水线，基于 HuggingFace Transformers / TRL / PEFT 生态构建。通过统一的 JSON 参数配置，一站式支持以下任务：

Pre-training — 从零预训练或继续预训练因果语言模型
Supervised Fine-Tuning (SFT) — 全参数指令微调，支持 ChatML / Alpaca 等多种 prompt 格式
LoRA Fine-Tuning — 低秩适配器高效微调，自动合并导出
DPO (Direct Preference Optimization) — 偏好对齐训练，支持多种 loss 类型
Inference — 对话生成测试
特性：

统一入口 train_master.py，通过 JSON 或 Base64 编码的 JSON 参数驱动所有任务
自动检测 bf16 / fp16 精度
支持 gradient checkpointing 节省显存
多数据源混合训练（分号分隔多路径）
跨平台启动脚本（train.sh / train.ps1）
自动处理中文 Windows GBK 编码与 UTF-8 冲突
技术栈： PyTorch · Transformers · TRL · PEFT · Accelerate · Datasets
