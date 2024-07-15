import torch
from ontoclassifier.wrappers import *
from ontoclassifier.nn import OntoClassifier, OntologicalExtractor 
from ontoclassifier.explainers import OntoClassifierExplainer, OCEViz


class OntologicalExtractorExplainer:
    
    def __init__(self, ontological_extractor: OntologicalExtractor, ontoclassifier: OntoClassifier) -> None:
        self.onto_extractor = ontological_extractor
        self.ontoclassifier = ontoclassifier
        
        self.all_pw = {}
        
        for i, fes in ontological_extractor.features_extractors.items():
            for fe in fes:
                for pw in fe.property_wrappers:
                    self.all_pw[pw] = i
        
    def __get_feature_extractor(self, feature):
        for pw in self.all_pw.keys():
            if pw.get_feature().property == feature:
                return pw
        return None
    
    def explain(self, onto_class, x):
        
        if onto_class not in self.ontoclassifier.getTargettedClasses():
            raise ValueError(
                "The target class must be in the ontology and among target classes. Available classes: "
                + str(self.ontoclassifier.getTargettedClasses())
            )
            
        merged_extracted_features = self.onto_extractor(x)

        explainer = OntoClassifierExplainer(self.ontoclassifier)
        explanations, results, textual_explanation = explainer.explain(
            onto_class, merged_extracted_features.squeeze(0)
        )

        for key in explanations.keys():
            reason, result = explanations[key]["reason"], explanations[key]["result"]
            feature_extractor = self.__get_feature_extractor(key.property)
            corresponding_inputs = x[self.all_pw[feature_extractor]]
                
            if str(type(feature_extractor)) == str(YoloV8PropertyWrapper):
                if reason is None:
                    boxes = None
                else:
                    indexes_to_highlight = reason.any(1).int().tile((6, 1)).transpose(1, 0)
                    boxes = torch.mul(feature_extractor.last_prediction, indexes_to_highlight)
                OCEViz.show_boxes(
                    corresponding_inputs.squeeze(0),
                    boxes,
                    feature_extractor.feature_range_names,
                    title=str(key) + "\n is " + str(result),
                )
