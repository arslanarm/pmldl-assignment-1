import os
from dataset import DetoxDataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration,  TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader


tokenizer_name = "ceshine/t5-paraphrase-paws-msrp-opinosis"
language_model_name = 'SkolkovoInstitute/t5-paraphrase-paws-msrp-opinosis-paranmt'

df = pd.read_csv("data/interm/dataset.csv")

tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name)


df_train, df_test = train_test_split(df, test_size=0.1)

train_references = tokenizer(df_train.references.tolist(), truncation=True)
train_translations = tokenizer(df_train.translations.tolist(), truncation=True)
test_references = tokenizer(df_test.references.tolist(), truncation=True)
test_translations = tokenizer(df_test.translations.tolist(), truncation=True)

train_dataset = DetoxDataset(train_references, train_translations)
test_dataset = DetoxDataset(test_references, test_translations)

train_dataloader = DataLoader(train_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, shuffle=False, num_workers=1)


model_save_path = "models/t5-detox"
if os.path.exists(model_save_path):
    language_model_name = model_save_path + "/checkpoint-5000"

model = T5ForConditionalGeneration.from_pretrained(language_model_name)



class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
        )
        ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
            padding=True,
        )
        batch['labels'] = ybatch['input_ids']
        batch['decoder_attention_mask'] = ybatch['attention_mask']

        return {k: torch.tensor(v) for k, v in batch.items()}


training_args = TrainingArguments(
    output_dir=model_save_path,   # output directory
    overwrite_output_dir=True,
    num_train_epochs=1,             # total # of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=100,               # number of warmup steps for learning rate scheduler
    weight_decay=0,                  # strength of weight decay
    learning_rate=3e-5,
    logging_dir='./logs',           # directory for storing logs
    logging_steps=50,
    eval_steps=4000,

    evaluation_strategy='steps',
    save_total_limit=1,
    save_steps=1000,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(model_save_path)