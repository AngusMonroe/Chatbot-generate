# chatbot-generate

## requirement

python3.6

pytorch



## Data Format

NER data

```
<word> <label>
```

classifier data

```
label	sentence
<lable> <sentence>
```


## usage:

`python main.py` or call `main(bot_id, ner_file_path, classify_file_path)` from `main.py`
	
## File Orgnization

```
|- [dir] dataset: data directorys named by bot_id
|- [dir] log
|- [dir] models: pertrained word vectors and well-trained model directorys named by bot_id
|- [dir] ner_evaluation: some result of NER model
|- [dir] service: service interface
|- [dir] service_impl: model loading and forecasting
|- [dir] util: some tools used
|- app.py
|- dataloader.py 
|- loader.py
|- main.py
|- model.py
|- monitor.py
|- train_classify.py
|- train_ner.py
|- utils.py
|- vocab.py
```

## Reference

[QAapi](https://github.com/wzyjerry/QAapi)

[BLSTM-CRF-NER](https://github.com/AngusMonroe/BLSTM-CRF-NER)

[text-classification](https://github.com/AngusMonroe/text-classification)
