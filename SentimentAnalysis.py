# #Importing necessary libraries
import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from string import punctuation
from collections import Counter

#reading training and testing data
#Training data contains 40,000 reviews and testing contains 5,000
test_data = pd.read_csv("Test.csv")
train_data = pd.read_csv("Train.csv")

#print a sample to see column labels and whether format is correct
# print(test_data.head())
# print(train_data.head())
# print("Test Data Shape = ", test_data.shape)
# print("Training Data Shape = ", train_data.shape)

#makes all params in numpy arrays
reviews_text = train_data['text'].to_numpy()
reviews_label = train_data ['label'].to_numpy()
test_review = test_data['text'].to_numpy()

#wrangling data to remove ambigious characters like (& , @ . ! ? ; :)
wrangledReview = []

#loops through text and appends each text
for text in reviews_text:
  text = text.lower()
  text = "".join([ch for ch in text if ch not in punctuation])
  wrangledReview.append(text)

#joins all together to one var
allReviewsWrangled = " ".join(wrangledReview)
words = allReviewsWrangled.split()

#counts the number of occurences of everything
EachWordCounter = Counter(words)


#converting all words to numbers (for nn to read and train) about 160000 unique words!
textToNum = {x:i+1 for i,(x,c) in enumerate(EachWordCounter.items())}
#print(textToNum)

# #making all reviews numbers, double for loop one iterates through every review
# #the other iterates through every word in the review, uses the textToNum var 
# #assign values to each word and appends them all together to understand
# #and learn.
textToNumReviews = []
for review in wrangledReview:
  textToNumReview = []
  for char in review.split():
   textToNumReview.append(textToNum.get(char, 0))
  textToNumReviews.append(textToNumReview)


# #ensures that the length of review is not too long
# #to prevent long training times.
# sequence_length=256
# features=np.zeros((len(textToNumReviews), sequence_length), dtype=int)
# for i, review in enumerate(textToNumReviews):
#   review_len=len(review)
#   if (review_len<=sequence_length):
#     new= review + [0] * (sequence_length-review_len)
#   else:
#     new=review[:sequence_length]
# features[i,:]=np.array(new)

sequence_length = 256
features = np.zeros((len(textToNumReviews), sequence_length), dtype=int)
for i, review in enumerate(textToNumReviews):
  review_len = len(review)
  if review_len <= sequence_length:
   new_review = review + [0] * (sequence_length - review_len)  
  else:
   new_review = review[:sequence_length]  
  features[i, :] = np.array(new_review)


# Save as JSON
with open("textToNumReviews.json", "w") as file:
    json.dump(textToNumReviews, file)

#prints shape of features, which shows a 40000 by 256, which is 
# 40000 reviews from train dataset set to max length of 256.
#print(f"Feature matrix shape: {features.shape}")


prop = int(0.8*len(features))

train_x = features[:prop]
train_y = reviews_label[:prop]

test_x = features[prop:]
test_y = reviews_label[prop:]


#Training & Setting Up Model!

#reproducible results
torch.manual_seed(42)


#creating Dataset to develop neural network
class ReviewsDataset(Dataset):
  def __init__(self, features, labels):
    self.features = torch.tensor(features, dtype = torch.long)
    self.labels = torch.tensor(labels, dtype = torch.float)

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, item):
    return self.features[item], self.labels[item]
  

#creating custom RNN for analysis
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

#Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(textToNum) + 1
hidden_dim = 256
embedding_dim=400
output_dim = 1  
n_layers = 2  
drop_prob = 0.5
learning_rate = 0.001
sequence_length = 256
batch_size = 64
epochs = 5  

#Setting up Dataloaders
train_dataset = ReviewsDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = ReviewsDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


#Initializing Model
model = SentimentAnalysisGRU(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, 
                             output_dim=output_dim, n_layers=n_layers, drop_prob=drop_prob)


#Use Binary Cross Entropy Loss as Binary Classfication
criterion = nn.BCELoss()

#Either SGD or Adam unsure for now, will have to find reasoning why one works better than the other
#Used Adam because not overfitting when it was being trained.
optimizer = optim.Adam(model.parameters(), lr =learning_rate)

train_on_gpu = torch.cuda.is_available()

def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for features, labels in train_loader:
      features, labels = features.to(device), labels.to(device)
      optimizer.zero_grad()
      predicted = model(features).squeeze()
      loss = criterion(predicted, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
if(train_on_gpu):
    model = model.cuda()
train_model(model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=epochs)

def test_model(model, test_loader):
  model.eval()

  test_loss =0 
  num_correct = 0
  total_samples = 0

  with torch.no_grad():
    for features, labels in test_loader:
      features, labels = features.to(device), labels.to(device)
      predicted = model(features).squeeze()
      loss = criterion(predicted, labels)

      predictions = (predicted >= 0.5).float()
      num_correct += (predictions == labels).sum().item()
      total_samples += labels.size(0)
      test_loss += loss.item()
  avg_loss = test_loss / len(test_loader)
  accuracy = (num_correct / total_samples) * 100

  print(f'Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f} %')

test_model(model, test_loader=test_loader)



torch.save(model.state_dict(), "sentiment_analysis_model.pth")
print("Model saved!!")
