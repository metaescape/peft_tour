
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


from peft import PeftModel, PeftConfig

peft_model_id = ("output_dir")
config = PeftConfig.from_pretrained(peft_model_id)

# load base model
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)

# wrap the model with peft model
# model = PeftModel.from_pretrained(model, peft_model_id)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
inputs = tokenizer("Tweet text : @HondaCustSvc Your customer service has been horrible during the recall process. I will never purchase a Honda again. Label :", return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
'complaint'