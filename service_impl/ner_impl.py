import os
import time
import jieba
import torch
import pickle
import tempfile
from util.ner_loader import *
from torch.autograd import Variable


class NERLoader(object):
    def __init__(self,
                 ner_stat='data/stat/ner_stat.pkl',
                 ner_train='data/aminer_train.dat',
                 pre_emb='data/cleaned_zh_vec'):
        self.ner_train = ner_train
        self.pre_emb = pre_emb
        self.stat = ner_stat
        if not os.path.isfile(ner_stat):
            self.save_stat()

    def save_stat(self):
        print('First start...\nSaving stat...')
        st = time.time()
        train_sentences = load_sentences(self.ner_train, lower=False, zeros=False)
        update_tag_scheme(train_sentences, 'iob')
        dico_words_train = word_mapping(train_sentences, lower=False)[0]
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            self.pre_emb, None
        )
        dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with codecs.open(self.stat, 'wb') as fout:
            pickle.dump({
                'word_to_id': word_to_id,
                'char_to_id': char_to_id,
                'tag_to_id': tag_to_id,
                'id_to_tag': id_to_tag
            }, fout)
        ed = time.time()
        print('Stat saved. Save time:', ed - st)

    def load_stat(self, model_path):
        print('Loading model...')
        st = time.time()
        ner_model = torch.load(model_path, map_location=torch.device('cpu'))
        ner_model.use_gpu = 0
        with codecs.open(self.stat, 'rb') as fin:
            stat = pickle.load(fin)
            word_to_id = stat['word_to_id']
            char_to_id = stat['char_to_id']
            tag_to_id = stat['tag_to_id']
            id_to_tag = stat['id_to_tag']
        ed = time.time()
        print('Model loaded. Load time:', ed - st)
        return ner_model, word_to_id, char_to_id, tag_to_id, id_to_tag


def evaluate(model, datas):
    data = datas[0]
    chars2 = data['chars']
    caps = data['caps']

    chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
    d = {}
    for i, ci in enumerate(chars2):
        for j, cj in enumerate(chars2_sorted):
            if ci == cj and not j in d and not i in list(d.values()):
                d[j] = i
                continue

    chars2_length = [len(c) for c in chars2_sorted]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
    for i, c in enumerate(chars2_sorted):
        chars2_mask[i, :chars2_length[i]] = c
    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    dwords = Variable(torch.LongTensor(data['words']))
    dcaps = Variable(torch.LongTensor(caps))

    val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
    return out


def ner(txt, model, word_to_id, char_to_id, tag_to_id, id_to_tag, lower=True):
    # txt = txt.strip()
    word = []
    for item in txt.split():
        word.extend(jieba.cut(item))
    tmp_file = tempfile.mkstemp(text=True)
    with codecs.open(tmp_file[1], 'w', 'utf-8') as fout:
        for w in word:
            fout.write(w + ' O\n')
        fout.write('\n-DOCSTART- -X- B-DATE B-DATE')
        fout.close()
    sentences = load_sentences(tmp_file[1], lower=lower, zeros=False)

    print(sentences)
    input_data = prepare_dataset(
        sentences, word_to_id, char_to_id, tag_to_id, lower=lower
    )

    prediction_id = evaluate(model=model, datas=input_data)
    prediction_tag = []
    for i in prediction_id:
        prediction_tag.append(id_to_tag[i])

    label = set()
    for tag in tag_to_id.keys():
        if tag.startswith('B-'):
            label.add(tag[2:])
    ans = {
        'O': []
    }
    for l in label:
        ans[l] = []

    stat = None
    tmp = []
    sep = ' '
    for w, i in zip(word, prediction_id):
        found_b = False
        for s in label:
            if i == tag_to_id['B-' + s]:
                if stat is not None:
                    ans[stat].append(sep.join(tmp))
                stat = s
                tmp = [w]
                found_b = True
                break
        if found_b:
            continue
        else:
            if stat is not None and 'I-' + stat in tag_to_id and i == tag_to_id['I-' + stat]:
                tmp.append(w)
            else:
                if stat is not None:
                    ans[stat].append(sep.join(tmp))
                stat = 'O'
                tmp = [w]
    if stat is not None:
        ans[stat].append(sep.join(tmp))

    try:
        os.remove(tmp_file[1])
    except OSError:
        pass

    return ans
