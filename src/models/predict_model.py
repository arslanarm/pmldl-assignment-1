from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast
import pandas as pd
import torch



def translate(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    print("Loading model")
    tokenizer = T5TokenizerFast.from_pretrained("ceshine/t5-paraphrase-paws-msrp-opinosis", max_len=512)
    model = T5ForConditionalGeneration.from_pretrained("models/t5-detox/checkpoint-2000")
    print("Finished loading model")
    device = torch.device('cuda:0')
    model = model.to(device)
    
    while True:
        text = input("Query: ")
        if text == "stop":
            break
        print("Translation:", translate(text))

