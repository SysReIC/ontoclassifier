from ontoclassifier import OntoFeature


class PropertyWrapper():
    
    def __init__(self, model, feature: OntoFeature, frange: dict={}):
        super().__init__()
        self.model = model
        self.feature = feature
        self.range = self.init_feature_range(frange)

    def init_feature_range(self, frange: dict={}):
        return frange
    
    def get_feature(self):
        return self.feature
    
    def extract_from(self, result):
        return result
    