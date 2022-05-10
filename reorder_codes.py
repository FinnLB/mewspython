import pandas as pd

modifications = {
    'Richtigkeit_der_Wortwahl': {10: 0},
    'Grammatik': {10: 0},
    'Orthographie': {10: 0},
}


def modify_value(mods, x):
    if x in mods.keys():
        return mods[x]
    return x


corpus = pd.read_csv('data.csv')

for column in corpus.columns:
    for m_column in modifications.keys():
        if m_column in column and 'Coder' not in column:
            corpus[column] = corpus[column].map(lambda x: modify_value(modifications[m_column], x))
            break

corpus.to_csv('data_ordinal.csv', index=False)
