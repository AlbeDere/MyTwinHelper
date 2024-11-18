from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset('json', data_files={'train': 'portfolio_conversations.json'})

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

def tokenize_function(examples):
    return tokenizer(examples['content'], truncation=True, padding="max_length", max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
