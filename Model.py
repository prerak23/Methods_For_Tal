import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import prepare_data
import create_target
import re
def preparedata(vocab,target_sentance_list):
    print(target_sentance_list,vocab)
    idxs = [vocab[w] for w in target_sentance_list]
    return torch.tensor(idxs, dtype=torch.long)
class LSTMClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, category_list):
        super(LSTMClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.types_of_classes=len(category_list)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim ,3)
        self.hidden2tag = nn.Linear(hidden_dim, len(category_list))
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.zeros(3, 1, self.hidden_dim),
                torch.zeros(3, 1, self.hidden_dim))
    def forward(self,sentance):
        embeds=self.word_embeddings(sentance) #Matrice [len(Sentance)*embeddingsize]
        print("Embeds Size", embeds.size(),len(sentance))
        input_to_lstm=embeds.view(len(sentance),1,-1) #Resize to [len(Sentance),1,embeddingsize]
        print("Input_to_lstm", input_to_lstm.size())
        lstm_out,self.hidden=self.lstm(input_to_lstm,self.hidden)
        print("output_to_lstm", lstm_out.size())
        class_score=self.hidden2tag(lstm_out.view(len(sentance),-1))
        print("Linear Layer output", class_score.size())
        class_scores=torch.transpose(class_score,0,1)
        print("Transpose Linear Layer output", class_scores.size())
        maxfromclass_score=torch.max(class_scores,-1)[0]
        print("Max Linear Layer output", maxfromclass_score.size())
        maxfromclass_score=maxfromclass_score.unsqueeze(0)
        print("Unsquezze Linear Layer output", maxfromclass_score.size())

        #inverse class_scores [no of classes * sentance size] and then take max from each row so [no of classes * 1]


        return class_score


vocab,list_of_Category,data_list=prepare_data.prepare_data()
model=LSTMClassification(5,10,len(vocab), list_of_Category)

optimizer = optim.SGD(model.parameters(), lr=0.1)

losses=[]
for epoch in range(2):

    for i in range(0,10):
        model.zero_grad()
        model.hidden=model.init_hidden()
        remove_after_punc = re.sub("[-!,'.()`?;:]", "", data_list[i]["headline"])
        print(data_list[i]["headline"],i)
        list_of_word = remove_after_punc.split(" ")
        sentance_in=preparedata(vocab,list_of_word)
        class_scores=model(sentance_in)
        print("output from ll",class_scores.size())
        class_scores=torch.transpose(class_scores,0,1)
        target=create_target.create_target(data_list[i]["category"],list_of_Category)
        print("target",target.size())
        print(class_scores.size())

        loss = F.cross_entropy(class_scores, target)
        losses.append(loss)
        loss.backward()
        optimizer.step()

print(losses)




