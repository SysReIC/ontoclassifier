import torch
from ontoclassifier import OntoFeature
from ontoclassifier.wrappers import PropertyWrapper


class IsAPropertyWrapper(PropertyWrapper):
    def __init__(self, feature: OntoFeature, frange: dict = ...):
        super().__init__(None, feature, frange)
        
    def init_feature_range(self, frange: dict = ...):
        self.classes_to_prop = {}
        for key, value_ in frange.items():
            if isinstance(value_, list):
                value = value_
            else:
                value = [value_]
            mapping = self.feature.encode_labels([value])  
            self.classes_to_prop[key] = mapping.squeeze(0)
    
    def extract_from(self, result):  
        zeros = torch.zeros((len(self.feature.get_range()))).int()        
        output = torch.stack([zeros if int(key) not in self.classes_to_prop.keys() else self.classes_to_prop[int(key)] for key in result])
        return output
        