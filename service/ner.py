import service_impl.ner_impl

ner_model_cache = {}

def ner(text, name):
    if name not in ner_model_cache:
        loader = service_impl.ner_impl.NERLoader(
            ner_stat='models/' + name + '/ner_mapping.pkl',
            ner_train='dataset/' + name + '/ner/train.dat',
            pre_emb='models/glove/mixed_vec')
        ner_model_cache[name] = loader.load_stat('models/' + name + '/ner.ml')
    model, word_to_id, char_to_id, tag_to_id, id_to_tag = ner_model_cache[name]
    return service_impl.ner_impl.ner(text, model, word_to_id, char_to_id, tag_to_id, id_to_tag)
