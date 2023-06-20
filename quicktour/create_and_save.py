from transformers import AutoModelForSeq2SeqLM
import os
from peft import LoraConfig, TaskType, get_peft_model


# load base model, change `transformer_dir` to your own path or use "" if load from HF_HOME cache dir
transformer_dir = "/data/huggingface/transformers"
model_name_or_path = os.path.join(transformer_dir, "bigscience/mt0-large")
tokenizer_name_or_path = model_name_or_path
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)


peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# wrap the model with peft model

model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters()) # check load success
"output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"


# this will only save lora model
model.save_pretrained("lora_output_dir")

# generat two file
# (torch13) me@change:/data/codes/peft_tour$ ls output_dir/
# adapter_config.json  adapter_model.bin