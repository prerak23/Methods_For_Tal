import torch 
import torch.nn as nn
import torch.nn.functional as F
import prepare_data
import torch.optim as optim
import re
import create_target

list_imp=[]
cuda=torch.device('cuda')
def preparedata(vocab,target_sentance_list):
    
    idxs = [vocab[w] for w in target_sentance_list]
    return torch.tensor(idxs, dtype=torch.long).cuda()
def accuracy(output_of_model,tag,total_count):
    top_n,top_i=output_of_model.topk(1)
    index_from_output=top_i[0].item()
    print("index_from_output",index_from_output,tag)
    if index_from_output == tag:
        total_count=total_count+1
    return total_count
        
def validation(models):
    testing_loss=[]
    with open("plotd_tests","w+", encoding="utf-8") as file2:
        totalloss=0
        accu=0
        global list_imp
        dict_of_accu={}
        counter2=0
        for i in range(500,4000):
            remove_after_punc = re.sub("[-!,'.()`?;:]","", data_list[i]["headline"])
            remove_after_punc=remove_after_punc.lower()
            list_of_word = list(remove_after_punc)
            if len(list_of_word) > 4:
                sentance_in=preparedata(vocab,list_of_word)
                class_scores=models(sentance_in)
                class_scores=class_scores
                target=create_target.create_target(data_list[i]["category"],list_of_Category,class_scores.size()[0])
                
                loss2=F.cross_entropy(class_scores,target)
                totalloss=totalloss+loss2.item()
                accu=accuracy(class_scores,target[0].item(),accu)
                class_scores=torch.transpose(class_scores,0,1)
                print(torch.nn.Softplus(class_scores),data_list[i]["category"])
                print("testing")
                testing_loss.append((loss2.item(),data_list[i]["category"]))
                if data_list[i]["category"] in dict_of_accu:
                    dict_of_accu[data_list[i]["category"]]+=1
                else:
                    dict_of_accu[data_list[i]["category"]]=1
                counter2+=1
        list_imp.append((str(totalloss/counter2),str((accu/counter2)*100)))
        file2.write(str(list_imp)+str(dict_of_accu)+str(accu))
       
        print(list_of_Category)


class Net(nn.Module):
    def __init__(self,vocab_size,embed_dim,n_filters,filter_sizes,classifi):
        super(Net,self).__init__()
        self.embedding=torch.nn.Embedding(vocab_size,embed_dim)
        self.conv0=nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0],embed_dim))
        self.conv1=nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embed_dim)) 
        self.conv2=nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[2], embed_dim))
        self.fc=nn.Linear(n_filters*3,classifi+1)
        self.dropout=nn.Dropout(0.25)
        

    def forward(self,x):
        ip=x.unsqueeze(0)

        embed=self.embedding(ip)

        print("embed op",embed.size())

        ipconv=embed.unsqueeze(1)

        print("ip to conv",ipconv.size())

        result1=self.conv0(ipconv)

        print("output of conv1", result1.size())

        conv_0=F.relu(result1.squeeze(3)).cuda()

        print("conv_0 size", conv_0.size())
        conv_1=F.relu(self.conv1(ipconv).squeeze(3)).cuda()
        conv_2=F.relu(self.conv2(ipconv).squeeze(3)).cuda()
        
        pooled0=F.max_pool1d(conv_0, conv_0.shape[2]).cuda()
        print("Pooled Size",pooled0.size())

        pooled0=pooled0.squeeze(2)

        print("After Squeeze Pooled",pooled0.size())
        pooled1=F.max_pool1d(conv_1, conv_1.shape[2]).cuda()
        
        pooled1=pooled1.squeeze(2)
        pooled2=F.max_pool1d(conv_2, conv_2.shape[2]).cuda()
        pooled2=pooled2.squeeze(2)
        cat = self.dropout(torch.cat((pooled0,pooled1,pooled2),dim=1))

        print("After Cat ",cat.size())

        return self.fc(cat)


vocab,list_of_Category,data_list=prepare_data.prepare_data()
model=Net(len(vocab),50,100,[3,4,5],len(list_of_Category)).to(device=torch.device('cuda'))
optimizer=optim.SGD(model.parameters(), lr=0.01)
losses=[]
with open("plotdatas.txt","w+", encoding="utf-8") as file:
    losses_to_print=[]
    
    for j in range(2):
        finalloss=0
        counter=0
        current_count=0
        for i in range(40000,80000):
            remove_after_punc = re.sub("[-!,'.()`?;:]", "", data_list[i]["headline"])
            remove_after_punc=remove_after_punc.lower()
            list_of_word = list(remove_after_punc)
            print(data_list[i]["headline"],list_of_word)
            if len(list_of_word) > 4:
                model.zero_grad()
                sentance_in=preparedata(vocab,list_of_word)
                class_scores=model(sentance_in).to(device=torch.device('cuda'))
                
                target=create_target.create_target(data_list[i]["category"],list_of_Category,class_scores.size()[0])
                target=target
                print("Model Output",class_scores.size(),target.size(),target)
                loss=F.cross_entropy(class_scores,target)
                print(loss)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()                
                finalloss=finalloss+loss.item()
                counter=counter+1
                current_count=accuracy(class_scores,target[0].item(),current_count)
        losses_to_print.append((str(finalloss/counter),str((current_count/counter)*100)))
        validation(model)
    file.write(str(losses_to_print))
    print(len(data_list))
    testing_loss=[]




