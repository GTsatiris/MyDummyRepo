import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import deepl

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Feedback(BaseModel):
    text: str
    needsTranslation: bool

auth_key = "bfb93185-d8fe-4027-8984-fdcd26ff3cf5:fx"
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def review_feedback_sentiment(text):
    inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors="pt")
    with torch.no_grad():
        result = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = result.logits.detach()
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        categories = ['Terrible', 'Poor', 'Average', 'Good', 'Excellent']
        output_dict = {}
        for i in range(len(categories)):
            output_dict[categories[i]] = [round(float(probs[i]), 2)]
    return output_dict

@app.get("/feedback")
def evaluate_feedback(message: Feedback):
    finalText = message
    if message.needsTranslation:
        translator = deepl.Translator(auth_key)
        finalText = translator.translate_text(message.text, target_lang="EN-US")
    result = review_feedback_sentiment(finalText.text)
    # grade = result['Excellent'][0] + 0.5*result['Good'][0] - 0.5*result['Poor'][0] - result['Terrible'][0]
    # grade = grade*0.5 + 0.5
    # return {"grade_100": str(grade), "grade_95": str(grade*0.95)}
    return result