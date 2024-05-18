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

from sklearn import svm
import pandas as pd

def gen_all_2mers():
    # amino_acids =  ['A', 'R', 'N', 'D', 'C', 'Q', 
    #                 'E', 'G', 'H', 'I', 'L', 'K', 
    #                 'M', 'F', 'P', 'S', 'T', 'W', 
    #                 'Y', 'V']
    amino_acids =  ['A', 'R', 'N', 'D', 'C', 'Q', 
                    'E', 'G', 'H', 'I', 'L', 'K', 
                    'M', 'F', 'P', 'S', 'T', 'W', 
                    'Y', 'V', 'X', 'Z', 'B'] # with X, Z, B added
    all = []
    for a in amino_acids:
        for b in amino_acids:
            if a+b not in all:
                all.append(a+b)
    return all

def create_df():
    columns = gen_all_2mers()
    columns.append("class")
    data = [[0] * len(columns)]
    df = pd.DataFrame(data, columns=columns)
    return df

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

def seq_to_pairs(seq):
    d = {}
    for i in range(0, len(seq)-1, 2):
        pair = seq[i]+seq[i+1]
        d[pair] = d.get(pair, 0) + 1
    return d

def fill_df(df, seqs, type):
    norm = len(df.columns)-1

    if type == "zincfinger":
        label = 0
    elif type == "globin":
        label = 1

    for protein, seq in seqs.items():
        df.loc[protein] = ([0.0] * len(df.columns)) 
        df.at[protein, "class"] = label
        d = seq_to_pairs(seq)
        for amino, freq in d.items():
            df.at[protein, amino] = freq / norm
    return df

globin_seqs = file_to_seqs("./sequences/globin.fasta")
zincfinger_seqs = file_to_seqs("./sequences/zincfinger.fasta")

df = create_df()
fill_df(df, zincfinger_seqs, "zincfinger")
fill_df(df, globin_seqs, "globin")

print(df)

# # find letters that are not in our alphabet
# amino_acids =  ['A', 'R', 'N', 'D', 'C', 'Q', 
#                 'E', 'G', 'H', 'I', 'L', 'K', 
#                 'M', 'F', 'P', 'S', 'T', 'W', 
#                 'Y', 'V']

# for protein, seq in globin_seqs.items():
#     if len(set(seq)-set(amino_acids)) != 0:
#         print(set(seq)-set(amino_acids))
# # we have letters that appear: ['X', 'Z', 'B'], why??

# '''
# 'X': This letter is used to represent an unknown or unspecified amino acid. 
#     In cases where the identity of an amino acid cannot be determined with certainty, 
#     'X' is used as a placeholder. 
#     This could be due to ambiguity in the sequencing data or when the amino acid at that position is not yet identified.

# 'Z': This letter is used to denote either glutamine (Q) or glutamic acid (E). 
#     This ambiguity often arises because these two amino acids can be difficult to 
#     distinguish in certain experimental contexts, such as mass spectrometry.

# 'B': This letter represents either asparagine (N) or aspartic acid (D). 
#     Similar to 'Z', this ambiguity occurs because these two amino acids are 
#     chemically similar and may not be easily differentiated in some types of analysis.
# '''

# Use 2 or 3 ML algorithms: SVM, Random Forests of NayveBayes