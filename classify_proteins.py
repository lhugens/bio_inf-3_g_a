'''
Group A
Leonardo Hügens up201705764
Hynek Šabacký up202311458
Lohras Karimi up202311453
'''

'''
Goal: given the protein sequences from two protein families
(globin and zinc finger), create a predictive model for a 
binary classification problem.
'''

import sklearn 
import pandas as pd

def file_to_seqs(filepath):
    file = open(filepath)
    lines = "".join(file.readlines())
    seqs = {}
    for line in lines.split("\n"):
        if ">sp" in line:
            id = line.split("|")[1]
            seqs[id] = seqs.get(id, "") 
        else:
            seqs[id] += line
    return seqs

def gen_all_2mers():
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 
               'E', 'G', 'H', 'I', 'L', 'K', 
               'M', 'F', 'P', 'S', 'T', 'W', 
               'Y', 'V']
    all = []
    for a in amino_acids:
        for b in amino_acids:
            if a+b not in all:
                all.append(a+b)
    return all

def create_df():
    columns = gen_all_2mers()
    data = [[0] * len(columns)]
    df = pd.DataFrame(data, columns=columns)
    return df

globin_seqs = file_to_seqs("./sequences/globin.fasta")
zinc_seqs = file_to_seqs("./sequences/zincfinger.fasta")

print(create_df())
