import numpy as np
import flair
from flair.data import Sentence
from flair.models import SequenceTagger

from deeppavlov.dataset_readers.conll2003_reader import Conll2003DatasetReader
dataset = Conll2003DatasetReader().read(data_path='data/', dataset_name='conll2003')

tagger = SequenceTagger.load('ner')

tags = ['B-ORG', 'B-PER', 'B-LOC', 'B-MISC', 'I-ORG', 'I-PER', 'I-LOC', 'I-MISC', 'O']
labels = {'B-ORG': 0, 'B-PER': 1, 'B-LOC': 2, 'B-MISC': 3, 'I-ORG': 4, 'I-PER': 5, 'I-LOC': 6, 'I-MISC': 7, 'O': 8}
# flair_labels = {'O': 1, 'S-ORG': 2, 'S-MISC': 3, 'B-PER': 4, 'E-PER': 5, 'S-LOC': 6, 'B-ORG': 7, 'E-ORG': 8, 'I-PER': 9,
# 				'S-PER': 10, 'B-MISC': 11, 'I-MISC': 12, 'E-MISC': 13, 'I-ORG': 14, 'B-LOC': 15, 'E-LOC': 16, 'I-LOC': 17}
flair_tags = [None, 'O', 'B-ORG', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'I-PER',
				'B-PER', 'B-MISC', 'I-MISC', 'I-MISC', 'I-ORG', 'B-LOC', 'I-LOC', 'I-LOC', None, None]

def gen_predict(sentence):
	predict = []
	for token in sentence:
		score = [0.0] * 9
		for i, label in enumerate(token.tags_proba_dist['ner']):
			if flair_tags[i] is not None:
				score[labels[flair_tags[i]]] += label.score
		score = np.divide(score, np.sum(score))
		predict.append(score)
	return predict

def gen_tag(predict):
	return [tags[np.argmax(score)] for score in predict]

for x, y in dataset['test'][:10]:
	sentence = Sentence(' '.join(x))
	tagger.predict(sentence, all_tag_prob=True)
	predict = gen_predict(sentence)
	tag = gen_tag(predict)

	print('Label :', y)
	print('Output:', tag)