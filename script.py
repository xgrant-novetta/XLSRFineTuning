

from datasets import load_dataset, load_metric
from datasets import ClassLabel
from IPython.display import display, HTML
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import IPython.display as ipd
import yaml
import numpy as np
import random
import torch
import random
import pandas as pd
import torchaudio
import re
import json
import FTMethods as ftm


# load YAML config file
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)


# Load the train and test datasets
common_voice_train = load_dataset("common_voice", config['setup']['language'], split="train+validation")
common_voice_test = load_dataset("common_voice", config['setup']['language'], split="test")



#Remove unnecessary columns
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])


# Remove special characters, outside of the ones you want to ignore
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'
common_voice_train = common_voice_train.map(ftm.remove_special_characters)
common_voice_test = common_voice_test.map(ftm.remove_special_characters)


# Collect the vocabulary lists from the train and test sets
vocab_train = common_voice_train.map(ftm.extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(ftm.extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)


#Construct a vocabulary dictionary
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print("length of the vocab_dict is", len(vocab_dict))

with open(config['setup']['vocab_dir'], 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
    

    
# Create a CTC tokenizer using the vocabulary list
tokenizer = Wav2Vec2CTCTokenizer(config['setup']['vocab_dir'], unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Create the processor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Resample to 16kHz
common_voice_train = common_voice_train.map(ftm.speech_file_to_array_fn, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(ftm.speech_file_to_array_fn, remove_columns=common_voice_test.column_names)

common_voice_train = common_voice_train.map(ftm.resample, num_proc=4)
common_voice_test = common_voice_test.map(ftm.resample, num_proc=4)

# Process and ensure correct sampling rates
common_voice_train = common_voice_train.map(ftm.prepare_dataset, remove_columns=common_voice_train.column_names, batch_size=8, num_proc=4, batched=True)
common_voice_test = common_voice_test.map(ftm.prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=4, batched=True)

# Define the data collator and evaluation metric
data_collator = ftm.DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")


# Fine-tune model for working with small datasets
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=config['model']['attention_dropout'],
    hidden_dropout=config['model']['hidden_dropout'],
    feat_proj_dropout=config['model']['feat_proj_dropout'],
    mask_time_prob=config['model']['mask_time_prob'],
    layerdrop=config['model']['layerdrop'],
    gradient_checkpointing=config['model']['gradient_checkpointing'], 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

# Define training arguments and set up trainer
training_args = TrainingArguments(
  output_dir=config['setup']['model_dir'],
  group_by_length=True,
  per_device_train_batch_size=config['args']['batch_size'],
  gradient_accumulation_steps=config['args']['steps'],
  evaluation_strategy="steps",
  num_train_epochs=config['args']['epochs'],
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=config['args']['learning_rate'], 
  warmup_steps=500,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)

# Train model
trainer.train()

# Save model
model.freeze_feature_extractor()
processor.save_pretrained(config['setup']['model_dir'])


# Test model on data
input_dict = processor(common_voice_test["input_values"][0], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

common_voice_test_transcription = load_dataset("common_voice", config['setup']['language'], data_dir="./cv-corpus-6.1-2020-12-11", split="test")

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription["sentence"][0].lower())
