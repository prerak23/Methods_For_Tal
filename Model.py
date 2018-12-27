import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import prepare_data
import create_target
import re
def preparedata(vocab,target_sentance_list):
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
        
        
        #inverse class_scores [no of classes * sentance size] and then take max from each row so [no of classes * 1]


        return class_scores


vocab,list_of_Category,data_list=prepare_data.prepare_data()
model=LSTMClassification(20,5,len(vocab), list_of_Category)

optimizer = optim.Adam(model.parameters(), lr=0.1)

losses=[]
with open("plotdata.txt","w+", encoding="utf-8") as file:
    for epoch in range(2):
        start=0
        end=500
        for j in range (0,6):
        
            print(start,end)
            for i in range(start,end):
            
                model.hidden=model.init_hidden()

                remove_after_punc = re.sub("[-!,'.()`?;:]", "", data_list[i]["headline"]+" "+data_list[i]["short_description"])
                list_of_word = remove_after_punc.split(" ")
                if len(list_of_word) > 2:
                    sentance_in=preparedata(vocab,list_of_word)
                    class_scores=model(sentance_in)
        
            
                    target=create_target.create_target(data_list[i]["category"],list_of_Category)
        
                    loss=F.cross_entropy(class_scores,target)
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
            start=start+500
            end=end+500
        losses.append("0101010")
    file.write(str(losses))

    print(len(data_list))
    testing_loss=[]
with open("plotd_test","w+", encoding="utf-8") as file2:
    for i in range(15000,15200):
        remove_after_punc = re.sub("[-!,'.()`?;:]","", data_list[i]["headline"])
        list_of_word = remove_after_punc.split(" ")
        if len(list_of_word) > 2:
            sentance_in=preparedata(vocab,list_of_word)
            class_scores=model(sentance_in)
        
            target=create_target.create_target(data_list[i]["category"],list_of_Category)
            loss2=F.cross_entropy(class_scores,target)
            print(class_scores,data_list[i]["category"])
            print("testing")
            testing_loss.append(loss2.item())
    file2.write(str(testing_loss))
    print(testing_loss)
    print(list_of_Category)

