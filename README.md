# rasa_nlu_zh
本github是基于[rasa_nlu](https://github.com/RasaHQ/rasa_nlu/tree/master) 0.14.6版本，添加pkuseg(0.0.12)替换jieba分词，提高分词准确率。  

注意：由于pkuseg导入用户词典时只支持单个文件，不支持文件夹，所以在配置是参数只能是文件  
	- name: "tokenizer_pkuseg"
  	dictionary_path: "pkuseg_userdict/ids_userdict.txt"
