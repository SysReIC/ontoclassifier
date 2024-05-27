from ontoclassifier import OntoFeature
from abc import abstractmethod, ABCMeta


# class PropertyWrapper(metaclass=ABCMeta):
class PropertyWrapper():
    
    def __init__(self, model, feature: OntoFeature, frange: dict={}):
        super().__init__()
        self.model = model
        self.feature = feature
        self.range = self.init_feature_range(frange)

    # @abstractmethod
    def init_feature_range(self, frange: dict={}):
        # pass
        return frange
    
    def get_feature(self):
        return self.feature
    
    # @abstractmethod
    def extract_from(self, result):
        # pass
        return result
    