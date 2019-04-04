# rasa_nlu_zh
本github是基于[rasa_nlu](https://github.com/RasaHQ/rasa_nlu/tree/master) 0.14.6版本，添加pkuseg(0.0.12)替换jieba分词，提高分词准确率。  

pipeline实例：

	language: "zh"
	
	pipeline:
	- name: "nlp_mitie"
	  model: "data/total_word_feature_extractor.dat"
	- name: "tokenizer_pkuseg"
	  dictionary_path: "pkuseg_userdict/ids_userdict.txt"
	  model_path: "pkuseg_pretrained_model/"
	- name: "ner_mitie"
	- name: "ner_synonyms"
	- name: "intent_featurizer_mitie"
	- name: "intent_classifier_sklearn"



注意：由于pkuseg导入用户词典时只支持单个文件，不支持文件夹，所以在配置是参数只能是文件  
	- name: "tokenizer_pkuseg"
  	dictionary_path: "pkuseg_userdict/ids_userdict.txt"

## pkuseg模型训练
	# train.txt 训练数据，仅支持utf-8编码，所有单词以单个或多个空格分开
	# test.txt 训练数据，仅支持utf-8编码，所有单词以单个或多个空格分开
	pkuseg.train('train.txt', 'test.txt', './models')
	# train_iter 训练轮数
	# init_model 预训练模型存放目录，预训练模型[下载](https://github.com/lancopku/pkuseg-python/releases)，
	# 一般使用混合领域分词模型作为重训练基础模型  
	pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
	pkuseg.test('msr_test.raw', 'output.txt', user_dict=None)