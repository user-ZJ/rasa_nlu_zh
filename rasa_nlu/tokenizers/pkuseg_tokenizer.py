from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import logging
import os
import shutil

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Text

logger = logging.getLogger(__name__)

PKUSEG_CUSTOM_DICTIONARY_PATH = "tokenizer_pkuseg"


class PkusegTokenizer(Tokenizer, Component):
    # 在pipeline中使用的名称
    name = "tokenizer_pkuseg"
    # 组件在调用时提供的属性
    provides = ["tokens"]
    # 语言支持列表
    language_list = ["zh"]
    # 此组件需要pipeline中前一个组件提供的属性，如果require包含“tokens”，则管道中的前一个组件需要在上述“provide”属性中具有“tokens”
    requires = []
    #定义组件的默认配置参数，这些值可以在模型的管道配置中覆盖。 组件应选择合理的默认值，并且应该能够使用默认值创建合理的结果。
    defaults = {
        "dictionary_path": None  # default don't load custom dictionary
    }

    def __init__(self, component_config=None):
        # type: (Dict[Text, Any]) -> None
        """Construct a new intent classifier using the MITIE framework."""

        super(PkusegTokenizer, self).__init__(component_config)

        # path to dictionary file or None
        self.dictionary_path = self.component_config.get('dictionary_path')

        # load dictionary
        if self.dictionary_path is not None:
            self.load_custom_dictionary(self.dictionary_path)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pkuseg"]

    @staticmethod
    def load_custom_dictionary(path):
        # type: (Text) -> None
        """Load all the custom dictionaries stored in the path.

        More information about the dictionaries file format can
        be found in the documentation of pkuseg.
        """
        import pkuseg

        pkuseg_userdicts = glob.glob("{}/*".format(path))
        for pkuseg_userdict in pkuseg_userdicts:
            logger.info("Loading Pkuseg User Dictionary at "
                        "{}".format(pkuseg_userdict))
            pkuseg.load_userdict(pkuseg_userdict)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        import pkuseg
        seg = pkuseg.pkuseg()
        words = seg.cut(text)
        tokens = []
        running_offset = 0
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> PkusegTokenizer

        meta = model_metadata.for_component(cls.name)
        relative_dictionary_path = meta.get("dictionary_path")

        # get real path of dictionary path, if any
        if relative_dictionary_path is not None:
            dictionary_path = os.path.join(model_dir, relative_dictionary_path)

            meta["dictionary_path"] = dictionary_path

        return cls(meta)

    @staticmethod
    def copy_files_dir_to_dir(input_dir, output_dir):
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob("{}/*".format(input_dir))
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory."""

        model_dictionary_path = None

        # copy custom dictionaries to model dir, if any
        if self.dictionary_path is not None:
            target_dictionary_path = os.path.join(model_dir,
                                                  PKUSEG_CUSTOM_DICTIONARY_PATH)
            self.copy_files_dir_to_dir(self.dictionary_path,
                                       target_dictionary_path)

            # set dictionary_path of model metadata to relative path
            model_dictionary_path = PKUSEG_CUSTOM_DICTIONARY_PATH

        return {"dictionary_path": model_dictionary_path}