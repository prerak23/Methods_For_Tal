import torch
def create_target(tag,category_list):
    one_hot_tag=torch.zeros(len(category_list))
    value=category_list[tag]
    one_hot_tag[value]=1
    return one_hot_tag.long()
