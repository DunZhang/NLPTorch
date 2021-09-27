""" 通用配置类 """
import json
from Utils.LoggerUtil import LoggerUtil

logger = LoggerUtil.get_logger()


class AConfig():
    """ 通用配置类 """

    def save(self, save_path):
        with open(save_path, "w", encoding="utf8") as fw:
            json.dump(self.__dict__, fw, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, conf_path: str):
        class_instance = cls()
        with open(conf_path, "r", encoding="utf8") as fr:
            kwargs = json.load(fr)
        for key, value in kwargs.items():
            try:
                if isinstance(value, dict):
                    logger.warning("key:{} 对应的value是字典不予赋值,".format(key))
                    continue
                if key not in class_instance.__dict__:
                    logger.warning("key:{} 不在类定义中,".format(key))
                    continue
                setattr(class_instance, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, class_instance))
                raise err
        return class_instance
