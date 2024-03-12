from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# model = AutoModelForSequenceClassificat
# ion.from_pretrained("s-nlp/bart-base-xsum-ttd", trust_remote_code=True)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# from huggingface_hub import hf_hub_download
#
# hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="bigscience_t0")
# tokenizer.save_pretrained("/home/nashtech/PycharmProjects/LIT10/bigscience_t0")
# model.save_pretrained("/home/nashtech/PycharmProjects/LIT10/bigscience_t0")

