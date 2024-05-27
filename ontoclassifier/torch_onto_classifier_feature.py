# coding=utf-8

from sklearn.preprocessing import MultiLabelBinarizer
import owlready2
import torch

from ontoclassifier import OntoClassifierHelper

# TODO: à la création : rajouter au range les superclasses de l'ontologie 
# TODO: dans l'encoding:  allumer automatiquement les superclasses

class OntoFeature(object):
    def __init__(self, property, property_range=None):
        self.property = property
        if property_range:
            self.property_range = property_range
        else:
            self.property_range = OntoClassifierHelper.get_possible_range(property)

        if isinstance(property, owlready2.ObjectPropertyClass):
            # IRI encoder
            iris = [r.iri for r in self.property_range]
            self.enc = MultiLabelBinarizer()
            self.enc.fit_transform([iris])
            # Short Label encoder
            self.range_labels = [str(item).split("#")[-1] for item in self.enc.classes_]
            self.labels_enc = MultiLabelBinarizer()
            self.labels_enc.fit_transform([self.range_labels])

    def encode_iris(self, elt_or_list):
        if type(elt_or_list) != list:
            elt_or_list = [elt_or_list]
        result = self.enc.transform(elt_or_list)
        return torch.Tensor(result).int()

    def encode_labels(self, elt_or_list):
        if type(elt_or_list) != list:
            elt_or_list = [elt_or_list]
        result = self.labels_enc.transform(elt_or_list)
        return torch.Tensor(result).int()

    def get_encoder(self):
        return self.enc
    
    def get_labels_encoder(self):
        return self.labels_enc

    def get_iri(self):
        return self.property.iri

    def get_property(self):
        return self.property

    def get_range(self):
        return self.property_range

    def get_range_labels(self):
        return self.range_labels


class OntoFeatureIsA(OntoFeature):
    def __init__(self, ontology, property_range=None):
        if property_range:
            range = property_range
        else:
            range = list(ontology.classes())
        super().__init__(owlready2.ThingClass, range)

        # IRI encoder
        iris = [r.iri for r in range]
        self.enc = MultiLabelBinarizer()
        self.enc.fit_transform([iris])
        # Short Label encoder
        self.range_labels = [str(item).split("#")[-1] for item in self.enc.classes_]
        self.labels_enc = MultiLabelBinarizer()
        self.labels_enc.fit_transform([self.range_labels])
        