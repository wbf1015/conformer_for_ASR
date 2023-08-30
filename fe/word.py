import numpy as np
import random

ground_truth_dic = {}
word_int_dic = {'P':0, 'S':1, 'E':2}
int_word_dic = {0:'P', 1:'S', 2:'E'}
num_class = None

def init_dic():
    global ground_truth_dic, word_int_dic, int_word_dic, num_class
    # 初始化编号和识别结果的字典
    if ground_truth_dic == {}:
        f = open('/root/autodl-tmp/AIshell1_truth.txt', encoding='utf-8')
        for line in f:
            s = line.strip()
            key = s[:16]
            value = s[16:]
            value = value.replace(' ', '')
            ground_truth_dic[key] = value
    word_int_dic, int_word_dic = {}, {}
    if word_int_dic == {}:
        f = open('/root/autodl-tmp/AIshell1_truth.txt', encoding='utf-8')
        unique_chars = set()
        for line in f:
            s = line.strip()
            value = s[16:]
            value = value.replace(' ', '')
            unique_chars.update(value)
        # 打乱顺序
        list_chars = list(unique_chars)
        random.shuffle(list_chars)
        # 填入字典
        index_count = 3
        for char in list_chars:
            word_int_dic[char] = index_count
            int_word_dic[index_count] = char
            index_count += 1
        num_class = index_count
        
        
        
def get_ground_truth(index):
    global ground_truth_dic, word_int_dic
    str_sentence = ground_truth_dic[index]
    int_sentence = []
    for char in str_sentence:
        int_sentence.append(word_int_dic[char])
    return str_sentence, np.array(int_sentence)

def get_num_class():
    global num_class
    return num_class

init_dic()
