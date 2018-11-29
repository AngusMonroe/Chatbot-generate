import service_impl.classify_impl

classify_model_cache = {}

def classify(text, name):
    if name not in classify_model_cache:
        loader = service_impl.classify_impl.ClassifyLoader(
            name=name,
            pre_emb='models/glove/mixed_vec')
        classify_model_cache[name] = loader.load_stat('models/' + name + '/classify.ml')
    model, d_word_index = classify_model_cache[name]
    return service_impl.classify_impl.classify(text, model, d_word_index)


# def classify(text, lang):
#     return service_impl.classify_impl.classify(text, model, d_word_index, lang)

if __name__ == '__main__':
    classify('hi', 'test')
