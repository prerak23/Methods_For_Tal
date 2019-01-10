import matplotlib.pyplot as plt
import ast
import re
file = open("plotdata_test")#Log data of Validation On The Model By Using The Test Set
string = file.read()
string = re.sub("/[\[\]']","",string)
train_data=eval(string) #Convert String To A List Of Tuples
counter=0
output_data=[]
accuracy_data=[]
epoch=[]
epoch_value=1
for data in train_data:
    output_data.append(float(data[0]))
    accuracy_data.append(float(data[1]))
    epoch.append(counter)
    counter+=1
plt.title("loss as a function of epoch")
plt.xlabel("epoch")
plt.ylabel("Accuracy Percentage")
plt.plot(accuracy_data) #Print The Data To The Plot
plt.savefig("acu_For_test")#Save The Data As A PNG File
print(output_data)


