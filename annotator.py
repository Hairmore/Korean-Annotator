import torch
from supar.models.dep.biaffine.model import BiaffineDependencyModel
from transformers import XLMRobertaModel
from supar import Parser
import os

if torch.cuda.is_available():
    print("cuda")
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)
model_path = input("Please enter the path of your model")
parser = Parser.load(model_path)
#print(parser.evaluate('/home/hairmore/Desktop/parser-main/dataset_conllu/train/SXD_data/testSXD.conllu', verbose=False))

#a = parser.predict("22일 국회 원내대표 회의실에서 열린 세월호 티에프 발족식에서 우 원내대표는 이같이 밝히며 “성역없는 조사가 진행돼야지, 어디는 빼자는 게 세월호 문제를 둘러싸고 여야가 할 이야기냐”고 비판했다.", lang="ko")
#print(str(a[0]))

def annotator(text):
    text = text.rstrip()
    t = text[:-1]
    text_write= "\n" + text +"\n"
    dataset = parser.predict(t, lang="ko")
    #data = open("/home/hairmore/Desktop/parser-main/testSXD_NXDresult.conllu", "a+")
    dependent_stored = input("Please enter the path to the conllu file where you would like to store annotated sentences.")
    data = open(dependent_stored, "a+")
    d = str(dataset[0])
    list_d = d.split('\n')
    list_d_clean = [e for e in list_d if e != ""]
    w = list_d_clean[-1].split('\t')
    w[1] = w[1]+text[-1]
    w_last = "\t".join(w)
    list_d_clean[-1] = w_last
    d_new = "\n".join(list_d_clean) +'\n'
    
    data.write(text_write)
    data.write(d_new)
    data.close()

sentences_tobe_annotated = input("Please enter the path to the file where stored sentences to be annotated.")
list_sentence = open(sentences_tobe_annotated,"r").readlines()

#list_sentence = open("/home/hairmore/Desktop/parser-main/差异比较/test_sent.txt","r").readlines()
c = 0
#print(len(list_sentence))
for s in list_sentence[1228732:1260001]:
    c+=1
    print(c)
    try:
    	annotator(s)
    except IndexError:
    	continue
    	


