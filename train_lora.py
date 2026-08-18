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
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", help="Comma-separated target modules")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default="./lora_output", help="Output directory for saved adapters")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push trained adapter to HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Target HF Hub repository ID")
    return parser.parse_args()

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
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        fp16=torch.cuda.is_available(),
    )

    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Pushing adapter to HF Hub: {args.hub_model_id}")
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    logger.info("LoRA training complete!")

if __name__ == "__main__":
    main()
