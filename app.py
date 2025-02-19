import gradio as gr
import torch
import torch.nn as nn
import json
from string import punctuation
from huggingface_hub import hf_hub_download


class SentimentAnalysisGRU(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob):
    super(SentimentAnalysisGRU, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim) 
    self.layer_norm = nn.LayerNorm(embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, batch_first=True, dropout=drop_prob)
    #need fully connected layer, activation function (Sigmoid since binary), dropout to prevent overfitting
    self.fc = nn.Linear(hidden_dim, output_dim)

    #activation function, Sigmoid because it is binary classification
    self.sigmoid = nn.Sigmoid()



  def forward(self, x):
    #first go through embedding layer
    embedded = self.embedding(x)
    embedded = self.layer_norm(embedded)
    #next, go through gru
    gru_out, _ = self.gru(embedded)
    #then, go through fc layer
    inp = gru_out[:, -1]
    out = self.fc(inp)

    #finally, go through activation function to get prediction
    return self.sigmoid(out)
  
textToNum_path = hf_hub_download(repo_id="athifsaleem/sentimentAnalysis", filename="textToNumReviews.json",repo_type="space")
with open(textToNum_path, "r") as file:
   textToNum = json.load(file)

print(len(textToNum)) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentAnalysisGRU(vocab_size=len(textToNum)+1, embedding_dim=400, hidden_dim=256, output_dim=1, n_layers=2, drop_prob=0.5)
model_path = hf_hub_download(repo_id="athifsaleem/sentimentAnalysis", filename="sentiment_analysis_model.pth", repo_type="space")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def tokenize_text(text, max_length=256):
   text = text.lower()
   text = "".join([ch for ch in text if ch not in punctuation])
   tokens = text.split()
   tokenized =[textToNum.get(word, 0) for word in tokens]

   if len(tokenized) < max_length:
      tokenized = tokenized + [0]*(max_length-len(tokenized))
   else:
      tokenized = tokenized[:max_length]
   return torch.tensor(tokenized, dtype=torch.long).unsqueeze(0).to(device) 
  
def function(text):
    text_inNum = tokenize_text(text)

    with torch.no_grad():
        output = model(text_inNum)
    
    if output.item() > 0.5:
        result = "Positive"
    else:
        result = "Negative"
    return result

demo = gr.Interface(function, inputs = "text", outputs = "text", title = "Sentiment Analysis")
demo.launch(share=True)
