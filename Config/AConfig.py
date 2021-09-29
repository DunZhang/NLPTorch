""" 通用配置类 """
import json
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class AConfig():
    """ 通用配置类 """

    def save(self, save_path):
        with open(save_path, "w", encoding="utf8") as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=1)

    def load(self, conf_path: str):
        with open(conf_path, "r", encoding="utf8") as fr:
            kwargs = json.load(fr)
        for key, value in kwargs.items():
            try:
                if key not in self.__dict__:
                    logger.warning("key:{} 不在类定义中,".format(key))
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err
