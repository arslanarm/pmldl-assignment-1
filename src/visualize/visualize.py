import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
import matplotlib.pyplot as plt


df = pd.read_csv("../data/raw/filtered.tsv", delimiter='\t')

plt.hist(df['ref_tox'])
plt.title('Toxicity levels of reference text')
plt.xlabel('Toxicity level')
plt.ylabel('Frequency')
plt.savefig('toxicity_levels_reference.png')

plt.hist(df['trn_tox'])
plt.title('Toxicity levels of translated text')
plt.xlabel('Toxicity level')
plt.ylabel('Frequency')
plt.savefig('toxicity_levels_translated.png')

plt.hist(df['similarity'])
plt.title('Similarity levels')
plt.xlabel('Similarity level')
plt.ylabel('Frequency')
plt.savefig('similarity_levels.png')

plt.hist(df['lenght_diff'])
plt.title('Length differences between reference and translated text')
plt.xlabel('Length difference')
plt.ylabel('Frequency')
plt.savefig('length_differences.png')


df = pd.read_csv("../data/interm/dataset.csv")


def get_word_diffs(df):
    ref_words = Counter(" ".join(df['references']).split())
    trans_words = Counter(" ".join(df['translations']).split())
    diff_words = ref_words - trans_words
    return diff_words


word_diffs = get_word_diffs(df)

plt.figure(figsize=(10,5))
most_common_diffs = word_diffs.most_common(20)
plt.bar([word for word, count in most_common_diffs], [count for word, count in most_common_diffs])
plt.title('Words that disappeared after translation')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()
