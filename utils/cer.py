import torch
import Levenshtein
import torch.nn.functional as F

def pre_process_sentence(sentence):
    ret_sentence = []
    for char in sentence:
        if 0<=char<=2:
            pass
        else:
            ret_sentence.append(char)
    return ret_sentence


# 直接传进来两个list就好
def cer(r: list, h: list):
    """
    Calculation of CER with Levenshtein distance.
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint16)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / float(len(r))


def record_log(predict, epoch, count):
    f = open('record.txt','a')
    f.write('EPOCH:'+str(epoch)+' COUNT:'+str(count))
    f.writelines(str(predict.tolist()))
    f.close()

def cer_for_batch(predict, target, output_lengths, epoch=None, count=None):
    ret_list = []
    ol = output_lengths.tolist()
    for index in range(0, predict.shape[0]):
        p = predict[index]
        t = target[index]
        p = torch.argmax(F.softmax(p, dim=1), dim=1)
        p = p.tolist()
        t = t.tolist()
        p = p[:ol[index]]
        # print(p)
        # record_log(predict, epoch, count)
        p = pre_process_sentence(p)
        t = pre_process_sentence(t)
        
        ret_list.append(Levenshtein.distance(p,t))
    return ret_list
        

if __name__ == "__main__":
    # r = '从卡耐基梅隆大学几代研发人员开始，本文对过去40年人们从语音识别技术进步所获得的启示进行了探讨。'
    # h = '从卡耐基梅隆大学几代研发人员开始，对过去40年人们从ASR技术进步所获得的启示进行了深入探讨。'
    # r = [x for x in r]
    # h = [x for x in h]
    # print(r)
    # print(h)

    # r = [3,5,7,9]
    # h = [3,5,7,9,11]
    # r = '北京师范'
    # h = '北京师范牛'
    # r = [x for x in r]
    # h = [x for x in h]
    # print(cer(r, h))
    
    s1=[1,2,3,4,5]
    s2=[11,35,67,89,111]
    print(Levenshtein.distance(s1,s2))