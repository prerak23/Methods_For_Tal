import matplotlib.pyplot as plt
import ast
import re
file = open("plotdata.txt")
string = file.read()
string = re.sub("/[\[\]']","",string)
output=string.split(",")
counter=0
output_data=[]
for data in output:
    if counter > 1 and  "'0101010'" not in data:
        
        output_data.append(float(data))
    counter+=1
plt.title("loss as a function of epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot([x  for x in range(len(output_data))],output_data)
plt.savefig("loss_as_whole")
print(output_data)
    


