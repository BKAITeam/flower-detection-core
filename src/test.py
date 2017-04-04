import os
from setting import NAME_LIST


def list_item(input_path):
    for item in os.listdir(input_path):
        if item in NAME_LIST:
            path = os.path.join(input_path, item)
            for hoa_item in os.listdir(path):
                if hoa_item.endswith('.png'):
                    path_hoa = os.path.join(path, hoa_item)
                    yield path_hoa

gen = list_item()
while(hasnext(list_item))
print dir(list_item)
