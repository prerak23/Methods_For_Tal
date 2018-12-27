import matplotlib.pyplot as plt
import ast
import re
file = open("plotdata.txt")
string = file.read()
string = re.sub("/[\[\]']","",string)
output=string.split(",")
counter=0
output_data=[]
epoch=[]
epoch_value=1
for data in output:
    if counter > 1 :
      if  "'0101010'" in data:
          epoch_value=epoch_value+1
        
      else:
          output_data.append(float(data))
          epoch.append(counter)
    counter+=1
plt.title("loss as a function of epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(epoch,output_data,"r--")
plt.savefig("loss_as_whole")
print(output_data)
    


