import service_impl.ner_impl


def ner(text, name):
    loader = service_impl.ner_impl.NERLoader(
        ner_stat='models/' + name + '/ner_mapping.pkl',
        ner_train='dataset/' + name + '/ner/train.dat',
        pre_emb='models/glove/mixed_vec')
    model, word_to_id, char_to_id, tag_to_id, id_to_tag = loader.load_stat('models/' + name + '/ner.ml')
    return service_impl.ner_impl.ner(text, model, word_to_id, char_to_id, tag_to_id, id_to_tag)
