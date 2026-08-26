import argparse
import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on Hugging Face model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model identifier")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco", help="Dataset identifier or file path")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="Text column name in dataset")
    parser.add_argument("--prompt_template", type=str, default="none", choices=["none", "alpaca", "chatml", "llama3", "custom"], help="Prompt template format")
    parser.add_argument("--custom_prompt_format", type=str, default="Instruction: {instruction}\nInput: {input}\nResponse: {output}", help="Custom format string with placeholders")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", help="Comma-separated target modules")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit QLoRA quantization")
    parser.add_argument("--use_safetensors", action="store_true", default=True, help="Use safetensors format for loading and saving model weights")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Output directory for saved adapters")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push trained adapter to HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Target HF Hub repository ID")
    return parser.parse_args()

def format_dataset(example, template, custom_format, text_field):
    if template == "alpaca":
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", example.get("response", ""))
        if input_text:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        example[text_field] = text
    elif template == "chatml":
        system_msg = example.get("system", "You are a helpful assistant.")
        user_msg = example.get("user", example.get("instruction", ""))
        assistant_msg = example.get("assistant", example.get("output", example.get("response", "")))
        text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        example[text_field] = text
    elif template == "llama3":
        system_msg = example.get("system", "You are a helpful assistant.")
        user_msg = example.get("user", example.get("instruction", ""))
        assistant_msg = example.get("assistant", example.get("output", example.get("response", "")))
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_msg}<|eot_id|>"
        example[text_field] = text
    elif template == "custom":
        try:
            text = custom_format.format(**example)
        except Exception:
            text = str(example.get(text_field, ""))
        example[text_field] = text
    return example

def main():
    args = parse_args()
    logger.info(f"Starting LoRA training with settings: {vars(args)}")

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(f"Missing dependency for LoRA training: {e}")
        logger.info("Please ensure peft, transformers, datasets, accelerate, and trl are installed.")
        sys.exit(1)

    quantization_config = None
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit QLoRA Quantization enabled.")
        except ImportError:
            logger.warning("bitsandbytes not installed; continuing without 4-bit quantization.")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    logger.info(f"Loading dataset: {args.dataset_name}")
    if os.path.exists(args.dataset_name):
        ext = os.path.splitext(args.dataset_name)[-1].lower()
        if ext == ".json" or ext == ".jsonl":
            dataset = load_dataset("json", data_files=args.dataset_name, split="train")
        elif ext == ".csv":
            dataset = load_dataset("csv", data_files=args.dataset_name, split="train")
        else:
            dataset = load_dataset("text", data_files=args.dataset_name, split="train")
    else:
        dataset = load_dataset(args.dataset_name, split="train")

    if args.prompt_template != "none":
        logger.info(f"Applying prompt template: {args.prompt_template}")
        dataset = dataset.map(
            lambda ex: format_dataset(ex, args.prompt_template, args.custom_prompt_format, args.dataset_text_field)
        )

    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        fp16=torch.cuda.is_available(),
    )

    logger.info(f"Loading base model: {args.base_model}")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "use_safetensors": args.use_safetensors,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {args.output_dir} (safe_serialization={args.use_safetensors})")
    trainer.model.save_pretrained(args.output_dir, safe_serialization=args.use_safetensors)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing adapter to HF Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id, safe_serialization=args.use_safetensors)
        tokenizer.push_to_hub(args.hub_model_id)

    logger.info("LoRA training complete!")

if __name__ == "__main__":
    main()
