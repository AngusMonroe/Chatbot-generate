import service_impl.classify_impl


def classify(text, name):
    loader = service_impl.classify_impl.ClassifyLoader(
        name=name,
        pre_emb='models/glove/mixed_vec')
    model, d_word_index = loader.load_stat('models/' + name + '/classify.ml')
    return service_impl.classify_impl.classify(text, model, d_word_index)


# def classify(text, lang):
#     return service_impl.classify_impl.classify(text, model, d_word_index, lang)

if __name__ == '__main__':
    classify('hi', 'test')
