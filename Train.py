# train.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL_NAME = "gpt2"
OUTPUT_DIR = "./models/checkpoint"

def load_and_prepare(dataset_path="data/stories.csv", text_column="text"):
    # expects CSV with column 'text' (full story or prompt+story)
    ds = load_dataset("csv", data_files=dataset_path)
    return ds

def tokenize_function(examples, tokenizer, text_column="text", block_size=512):
    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=block_size)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT2 doesn't have pad token by default; set it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    ds = load_and_prepare()
    text_column = "text"
    tokenized = ds["train"].map(lambda ex: tokenize_function(ex, tokenizer, text_column), batched=True, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=False  # set True if you have a GPU with mixed precision
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Saved model to", OUTPUT_DIR)

if __name__ == "__main__":
    main()

