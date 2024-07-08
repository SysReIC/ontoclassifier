import torch
import torch.nn as nn
from ontoclassifier import OntoFeature
from ontoclassifier.nn import FeaturesExtractor


class OntologicalExtractor(nn.Module):
    
    def __init__(self, features_extractors: dict[int, list[FeaturesExtractor]] | list[FeaturesExtractor]) -> None:
        super().__init__()
        
        if isinstance(features_extractors, list):
            self.features_extractors = {0: features_extractors}
        else:
            self.features_extractors = features_extractors
    
    def get_ontological_features(self) -> list[OntoFeature]:
        features = []
        for i in self.features_extractors.keys():
            for fe in self.features_extractors[i]:
                for pw in fe.property_wrappers: 
                    if pw.get_feature() not in features: 
                        features.append(pw.get_feature())
        return features
    
    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        if len(x) != len(self.features_extractors.keys()):
            raise ValueError(
                "The length of x must be equal to the number of feature extractors inputs"
            )
        extracted_features = []
        for i in self.features_extractors.keys():
            x_for_feature = x[i]
            for feature_extractor in self.features_extractors[i]:
                extracted_features.append(feature_extractor(x_for_feature))

        merged_extracted_features = torch.cat(extracted_features, dim=1)
        ic("OE output:"+ str(merged_extracted_features.shape))
        return merged_extracted_features
