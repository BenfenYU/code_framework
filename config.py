from base_code import EasyDict

class Base_config:
    def __init__(self):
        self.config_name = self.__class__.__name__.split("_")[-1]

        """
        公有的，会被继承的参数
        """
        self.default = EasyDict(
            inference = False,
            checkpoints_dir = "",
            )

    def get_config(self,config_num):
        complete_config = self.default.update(getattr(self,f"config_{self.config_name}_{config_num}"))

        return complete_config


class Config_a(Base_config):
    def __init__(self):
        super().__init__()

    """
    配置自有的，或需要额外设置的参数
    """
    config_a_1 = EasyDict(

    )

    config_a_2 = EasyDict(

    )

    config_a_inference_1 = EasyDict(

    )