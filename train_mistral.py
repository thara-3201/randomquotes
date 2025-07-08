from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

dataset = load_dataset("json", data_files="datasets.jsonl", split="train")

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit = True, device="auto")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

def tokenize(text):
    full_prompt = f"<s>[INST]{text['prompt']}[/INST] {text['completion']}</s>"
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./lora-mistral-affirmations",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

trainer.train()