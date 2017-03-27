import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dir path of this file (setting.py)

NAME_LIST = {
    # "callalily": 'cal',
    # "daisy": "dai",
    # "hibiscus":"hib",
    "hortensia":"hor",
    "hortensia-violet":"hor",
    "marigold":"mar",
    "marigold-red":"mar"
}

NAME_LIST = {
    "Test": 'test',
    "Hoa Canh Buom": 'cos',
    # "Hoa Hong": "ros",
    # "Hoa Hong Mon": "ant",
    "Hoa Huong Duong": "sun",
    # "Hoa Ly": "lil",
    # "Hoa Mao Ga": "coc",
    # "Hoa Sen": "lot",
    "Hoa Thien Dieu": "str",
    # "Hoa Thuoc Duoc": "dah",
    # "Hoa Trang": "ixo",
    "Hoa But": 'hib',
    # "Hoa Cuc Trang": 'dai',
    "Hoa Rum": 'cal',
}


test_pic = "../test.jpg"
source_flowers_version_path = "./Hoa2"
output_flowers_version_path = "../Build/version-4/"
test_pic = "../test.jpg"

INPUT_PATH = os.path.join(BASE_DIR, "Build/Binary")
OUT_PATH = os.path.join(BASE_DIR, "Build")
TEST_OUT_PATH = os.path.join(BASE_DIR, "Test")
TEST_IN_PATH = os.path.join(BASE_DIR, "Test")
