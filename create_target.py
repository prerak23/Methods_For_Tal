import torch
def create_target(tag,category_list,length_of_input):
    one_hot_tag=torch.zeros(length_of_input).cuda()
    value=category_list[tag]
    one_hot_tag[0]=value
    
    return one_hot_tag.long()
