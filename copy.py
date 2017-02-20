from os import listdir, makedirs
from os.path import isfile, join, exists
import shutil

for item in listdir("./Hoa"):
    loc_binary_path = join("./Hoa", item, "Loc_Binary")
    origin_path = join("./Hoa", item, "Origin")
    output_path = join("./Hoa", item, "Loc_Origin")
    if not exists(output_path):
        makedirs(output_path)
    for item in listdir(loc_binary_path):
        if not item.endswith(".png"):
            continue
        des = join(output_path, item)
        src = join(origin_path, item)
        shutil.copy(src,des)
