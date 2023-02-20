#https://www.youtube.com/watch?v=QEaBAZQCtwE&t=423s&ab_channel=AssemblyAI
#https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/pipelines 
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I have mixed opinions about this sandwich")

print(res)