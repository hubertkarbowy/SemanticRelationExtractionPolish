from abc import ABC, abstractmethod
from DataProvider import DataProviderFactory

class EncjoSzukacz(ABC):
    def __init__(self, *, config):
        self.config = config
        if self.config.get("task") is None:
            raise ValueError("Please set a task type in the configuration file")
        self.data_provider = DataProviderFactory.get_instance(config['input_data']['reader'], config)
        if config['input_data'].get('deserialize') is True:
            self.data_provider.deserialize()
        else:
            self.data_provider.slurp()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass
