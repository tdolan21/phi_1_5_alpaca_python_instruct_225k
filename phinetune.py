import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
import json
import os

login()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5",
    device_map={"":0},
    trust_remote_code=True,
    quantization_config=bnb_config
)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["Wqkv", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

print(model)



def tokenize(sample):
    model_inps =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return model_inps

# Convert the chat_rounds to a single text string
def format_chat_rounds(chat_rounds):
    return " ".join([f"{cr['role']}: {cr['content']}" for cr in chat_rounds])

# Load gsm8k dataset
data = load_dataset("gsm8k", "main", split="train").to_pandas()

# Load code_instructions dataset
code_instructions = pd.read_json("data/code_instruction_122k_alpaca_style_filtered.json", lines=True)
code_instructions["text"] = "instruction: " + code_instructions["instruction"] + " output: " + code_instructions["output"]

# Load alpaca_data dataset
alpaca_data = pd.read_json("data/alpaca_evol_instruct_70k.json")
alpaca_data["text"] = "instruction: " + alpaca_data["instruction"] + " output: " + alpaca_data["output"]

# Load and format code_exercise_data
with open("data/CodeExercise-Python-27k.json", "r") as f:
    lines = f.readlines()
parsed_code_exercises = [json.loads(line) for line in lines]
code_exercise_data = pd.DataFrame(parsed_code_exercises)
# Assuming format_chat_rounds is defined elsewhere in your code
code_exercise_data['text'] = code_exercise_data['chat_rounds'].apply(format_chat_rounds)

# Format the gsm8k DataFrame to have a "text" column
data["text"] = data[["question", "answer"]].apply(lambda x: "question: " + x["question"] + " answer: " + x["answer"], axis=1)

# Concatenate all DataFrames (gsm8k, code_instructions, alpaca_data, and code_exercise_data)
all_data_df = pd.concat([data[["text"]], code_instructions[["text"]], alpaca_data[["text"]], code_exercise_data[["text"]]])

# Initialize the token counter
total_trained_tokens = 0

# Convert to Hugging Face Dataset and tokenize
data = Dataset.from_pandas(all_data_df)
tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)


training_arguments = TrainingArguments(
        output_dir="phi_1.5_alpaca_python_instruct_225k",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        max_steps=1000,
        num_train_epochs=1,
        push_to_hub=True
    )
     
# Modify your Trainer class initialization to include a custom training loop
class PhiTrainer(Trainer):
    def training_step(self, model, inputs):
        global total_trained_tokens  # Use the global counter
        # Call the original training_step
        step_output = super().training_step(model, inputs)
        # Update the token counter
        total_trained_tokens += inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        return step_output

trainer = PhiTrainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()
trainer.push_to_hub()

print(f"Total trained tokens: {total_trained_tokens}")



model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(model, "your_huggingface_username/phi_1.5_alpaca_python_instruct_225k", from_transformers=True)
model = peft_model.merge_and_unload()
     
model.push_to_hub("your_huggingface_username/phi_1.5_alpaca_python_instruct_225k")
