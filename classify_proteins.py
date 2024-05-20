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
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score

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

# print(df)

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

# Use 2 or 3 ML algorithms: SVM, Random Forests of NaiveBayes
# Evaluate the models with 3 or 4 measures, e.g. recall, precision or F1-score

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

skf = StratifiedKFold(n_splits=10)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    # print(f"    Train: index={train_index}")
    # print(f"    Test: index={test_index}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # SVM
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("    SVM:")
    print(f"        Recall: {round(recall_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        Precision: {round(precision_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        F1 score: {round(f1_score(y_test, y_pred, average='micro'), 3)}")

    # Random Forests
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print("    Random Forests:")
    print(f"        Recall: {round(recall_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        Precision: {round(precision_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        F1 score: {round(f1_score(y_test, y_pred, average='micro'), 3)}")

    # NaiveBayes
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("    NaiveBayes:")
    print(f"        Recall: {round(recall_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        Precision: {round(precision_score(y_test, y_pred, average='micro'), 3)}")
    print(f"        F1 score: {round(f1_score(y_test, y_pred, average='micro'), 3)}")
