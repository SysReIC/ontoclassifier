import sys
import owlready2
from owlready2 import *
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load
import time

# from ontoclassifier.nn import OntologicalExtractor
from ontoclassifier.nn import *


from pprint import pprint
from icecream import ic
from icecream import install
install()
ic.configureOutput(prefix=f'Debug | ', includeContext=True)


class OntoClassifier(torch.nn.Module):

    OPERATORS_REPR = {
        owlready2.And : "AND",
        owlready2.Or : "OR",
        owlready2.Not : "NOT",
        owlready2.VALUE : "=",
        owlready2.MIN: "MIN",
        owlready2.MAX: "MAX",
        owlready2.EXACTLY: "EXACTLY",
        owlready2.ONLY: "ONLY",
        owlready2.SOME: "SOME",
    }

    CLASSES_KEY = 'CLASSES'
    FEATURES_KEY = "FEATURES"

    def __init__(self, 
                 ontology: owlready2.Ontology, 
                 target_classes: list[owlready2.ThingClass]=[], 
                 ontological_extractor: OntologicalExtractor|None=None, 
                 to_ignore: list=[], 
                 debug: bool=False,
                 java_exe: str|None=None):
        super().__init__()
        if java_exe is not None:
            owlready2.JAVA_EXE = java_exe
        self.ontology = ontology
        self.inferences = owlready2.get_ontology("http://test.org/onto_inferences.owl")
        self.syncReasoner()
        self._graphs = None
        self._explainers = None
        self._targetted_classes = set()
        self._targetted_classes_encoder = None
        self._targetted_ontofeatures = {}
        self._feature_input_focus = {}
        self._input_size = 0
        
        self.to_ignore = to_ignore
        self._anonymous_classes_cache = DictionaryCache()
        self._layers_cache = DictionaryCache()
        
        if debug:
            ic.enable()
        else:
            ic.disable()
        
        if target_classes is not None and len(target_classes) > 0:
            self.setTargettedClasses(target_classes)
            
            if ontological_extractor is not None:
                for feature in ontological_extractor.get_ontological_features():
                    self.setTargettedFeature(feature)
                # for property_wrapper in features_extractor.property_wrappers:
                    # self.setTargettedFeature(
                    #     property_wrapper.get_feature()
                    # )
                self.buildGraph()
        
    def __buildFeatureInputFocus(self):
        start = 0
        end = 0
        for iri, v in self._targetted_ontofeatures.items():
            end = start + len(v.get_range())
            self._feature_input_focus[iri] = (start, end)
            start = end
        self._input_size = end

    def buildGraph(self):
        self._feature_input_focus = {}
        self.__buildFeatureInputFocus()

        self._graphs = torch.nn.ModuleList()
        self._explainers = {}

        to_remove = []

        for class_iri in self._targetted_classes_encoder.classes_:
            onto_class = self.ontology.ontology.search(iri=class_iri)[0]
            # sub = self.__parse(onto_class.equivalent_to)
            # ICI TEST
            try:
                sub = self.__parse(onto_class.equivalent_to)
            except Exception as err:
                pprint(err)
                sub = None
            if(sub is None):
                # raise Exception("PROBLEM with  " + onto_class.name + " : no property to work with !")
                pprint("[ REMOVING  " + onto_class.name + " : no property to work with. ]")
                to_remove.append(onto_class)
            else:
                try:
                    self._graphs.append(sub)
                    self._explainers[onto_class] = sub
                except: 
                    print(" ERROR for ", onto_class)
                    to_remove.append(onto_class)

        if len(to_remove) > 0:
            classes = self._targetted_classes
            for c in to_remove:
                classes.remove(c)
            self.setTargettedClasses(classes)
            print("Removed :")
            pprint(to_remove)

    def __buildTargetClassEnc(self):
        self._targetted_classes_encoder = MultiLabelBinarizer()
        iris = [c.iri for c in self._targetted_classes]
        self._targetted_classes_encoder.fit_transform([iris])

    def encode(self, instance):
        res = torch.zeros(self._input_size, dtype=torch.int64)
        for p in instance.get_properties():

            if isinstance(p, owlready2.ObjectPropertyClass):
                try:
                    encoder = self.getTargettedFeaturesEncoder(p)
                    range = eval('instance.' + p.name)

                    try:
                        iterator = iter(range) # check if list or single entity
                    except TypeError:
                        range = [range]

                    stack = []
                    for idx, r in enumerate(range):
                        encoded_entity = self.encode_entity(r, encoder)
                        if not encoded_entity is None:
                            stack.append(encoded_entity)
                    bits = torch.stack(stack).transpose(0,1).int()
                    encoded = OntoClassifierHelper.bin2int(bits)
                    # ic(p, encoded)

                    start, end = self._feature_input_focus[p.iri]
                    res[start:end] = encoded
                except KeyError:
                    continue

            elif isinstance(p, owlready2.DataPropertyClass):
                try:
                    start, end = self._feature_input_focus[p.iri]
                    res[start:end] = eval('instance.' + p.name)
                except KeyError:
                    continue

            else:
                ic(p, " : cas non traité !")
        return res

    def encode_entity(self, entity, encoder):
        encoded = torch.zeros(len(encoder.classes_), dtype=torch.int64)
        try:
            id = self.getIdFromElement(entity, encoder)
            if not id is None:
                encoded[id] = 1
        except: pass

        for c in entity.is_a:
            encoded_entity = self.encode_entity(c, encoder)
            encoded = torch.logical_or(encoded, encoded_entity)

        return encoded.int()

    def forward(self, inputs):
        preds = []
        for l in self._graphs:
            if l is None: print("ERROR: graph is empty !")
            preds.append(l(inputs))
        return torch.stack(preds, dim=1)

    def getGraphFor(self, onto_class):
        graph_index = np.argwhere(self._targetted_classes_encoder.transform([[onto_class.iri]])[0]).squeeze()
        return self._graphs[graph_index]

    def getExplainerFor(self, onto_class):
        return self._explainers[onto_class]

    def getClassesProperties(self, classes, with_parents=True, with_equivalent=True):
        self._treated = []
        res = []
        for c in classes:
            res.extend(self.getClassProperties(c, with_parents, with_equivalent))
        return list(set(res))

    def getClassProperties(self, elt, with_parents=True, with_equivalent=True):
        res = []

        # avoid RecursionError
        if elt in self._treated:
            return res

        self._treated.append(elt)

        if type(elt) == owlready2.ThingClass:

            # direct properties
            for prop in elt.get_class_properties():
                res.append(prop)

            # properties from parents
            if with_parents:
                for c in self.getParentsOf(elt):
                    res.extend(self.getClassProperties(c, with_parents, with_equivalent))

            # properties from equivalent
            if with_equivalent:
                res.extend(self.getClassProperties(elt.equivalent_to, with_parents, with_equivalent))

        elif type(elt) == owlready2.entity._EquivalentToList or type(elt) == list:
            for sub in elt:
                res.extend(self.getClassProperties(sub, with_parents, with_equivalent))

        elif type(elt) == owlready2.And or type(elt)==owlready2.Or:
            for c in elt.Classes:
                res.extend(self.getClassProperties(c, with_parents, with_equivalent))

        elif type(elt) == owlready2.Not:
            res.extend(self.getClassProperties(elt.Class, with_parents, with_equivalent))

        elif type(elt) == owlready2.Restriction:
            res.extend(self.getClassProperties(elt.property, with_parents, with_equivalent))

        elif type(elt) == owlready2.ObjectPropertyClass:
            res.append(elt)

        return list(set(res))

    def getIdFromElement(self, entity, encoder):
        if not entity.iri in encoder.classes_: return None
        return np.argwhere(encoder.transform([[entity.iri]]).squeeze())[0][0]

    def getLeafs(self, onto_class_list):
        subs = set()
        for c in onto_class_list:
            ssubs = list(c.subclasses())
            if len(ssubs) == 0:
                subs.add(c)
            else:
                new_leafs = set(self.getLeafs(ssubs))
                subs = subs.union(new_leafs)
        return list(subs)

    def _getMaskEquiv(self, elt_or_list, encoder):

        # check if list or single entity
        try:
            iterator = iter(elt_or_list)
        except TypeError:
            elt_or_list = [elt_or_list]

        targets = []
        for entity in elt_or_list:

            if entity.iri in encoder.classes_ :
                targets.append(self._getMaskSimple(entity, encoder))
                # supers = self.getSupers(entity)
                # targets.append(self._getMaskSimple(supers, encoder))

            if type(entity) == owlready2.entity.ThingClass:
                for i in entity.instances():
                    targets.append(self._getMaskEquiv(i, encoder))
                for sub in entity.__subclasses__():
                    targets.append(self._getMaskEquiv(sub, encoder))

        # mask = torch.zeros(len(encoder.classes_))
        # for t in targets:
        #     mask = torch.logical_or(mask, t)
        # return mask.int()
        
        if len(targets) > 0:
            masks = []
            for t in targets :
                if (t.sum() > 0):
                    if len(t.shape) == 1: 
                        masks.append(torch.stack([t]))
                    else:
                        masks.append(t)
            if len(masks) > 0:
                return torch.cat(tuple(masks), 0).int()
            else:
                return torch.zeros(len(encoder.classes_)).int()
        else:
            return torch.zeros(len(encoder.classes_)).int()

    def _getMaskSimple(self, elt_or_list, encoder):

        try: # check if list or single entity
            iterator = iter(elt_or_list)
        except TypeError:
            elt_or_list = [elt_or_list]

        mask = torch.zeros(len(encoder.classes_), dtype=torch.int)
        for t in elt_or_list:
            try:
                idx = self.getIdFromElement(t, encoder)
                if not idx is None: mask[idx] = 1
            except:
                pass
                # ic("[ IGNORED : %s ]" % t)

        return mask.int()

    # def _getMaskSupers(self, elt_or_list, encoder):

    #     try: # check if list or single entity
    #         iterator = iter(elt_or_list)
    #     except TypeError:
    #         elt_or_list = [elt_or_list]

    #     targets = self.getSupers(elt_or_list)
    #     return self._getMaskSimple(targets, encoder).int()


    def getParentsOf(self, onto_class, inferred=False, full_hierarchy=False):
        parents = set()
        for c in self.ontology.get_parents_of(onto_class):
            parents.add(c)
        if inferred:
            for c in self.inferences.get_parents_of(onto_class):
                parents.add(c)

        if (full_hierarchy):
            ancestors = set()
            for c in parents:
                if isinstance(c, owlready2.ThingClass):
                    ancestors.update(set(self.getParentsOf(c, True)))
            parents.update(ancestors)

        return list(parents)


    def getTargettedClasses(self):
        return list(self._targetted_classes)

    def getTargettedClassesEncoder(self):
        return self._targetted_classes_encoder

    def getTargettedFeatures(self):
        return self._targetted_ontofeatures

    def getTargettedFeaturesEncoder(self, property):
        return self._targetted_ontofeatures[property.iri].get_encoder()

    def loadEntities(self, filename):
        entities = load(filename)

        classes = []
        for iri in entities[OntoClassifier.CLASSES_KEY]:
            entity = self.ontology.search(iri=iri)[0]
            classes.append(entity)
        self.setTargettedClasses(classes)

        for iri, elts in entities[OntoClassifier.FEATURES_KEY].items():
            property = self.ontology.search(iri=iri)[0]
            elements = []
            for e in elts:
                if type(e) == str :
                    elements.append(self.ontology.search(iri=e)[0])
                else:
                    elements.append(e)
            feature = OntoFeature(property, elements)
            self.setTargettedFeature(feature)

    def __parse(self, elt, encoder=None):

        try:                
            if elt.property in self.to_ignore:
                print("[ IGNORING", elt, "]")
                return None
        except:
            pass
        
        try:
            if not(isinstance(elt, list)) and self._layers_cache.contains(elt):
                # print("[ USING CACHE FOR ",elt, " ]")
                return self._layers_cache.get(elt)
        except:
            pass
        
        if isinstance(elt, owlready2.entity._EquivalentToList) or isinstance(elt, list):
            if len(elt) == 0:
                return None
            else:
                for c in elt:
                    layer = self.__parse(c)
                    self._explainers[c] = layer
                    return layer

        elif isinstance(elt, owlready2.And):
            parts = []
            layer = None
            for c in elt.Classes:
                part = self.__parse(c, encoder=encoder)
                if not part is None:
                    parts.append(part)
                # else:
                #     elt.Classes.remove(c)
            if len(parts) > 0:
                if isinstance(parts[0], nn.Module) :
                    layer = OntoAndLayer(parts)
                else:
                    mask = parts[0]
                    for m in parts[1:]:
                        # TODO Pourra poser un PB si AND entre plusieurs Multi-classes
                        try: 
                            mask = torch.logical_or(mask, m)
                        except: return None
                    layer = mask.int()
            self._explainers[elt] = layer
            self._layers_cache.add(elt, layer)
            return layer

        elif isinstance(elt, owlready2.Or):
            parts = []
            layer = None
            for c in elt.Classes:
                part = self.__parse(c, encoder=encoder)
                if not part is None:
                    parts.append(part)
                # else:
                #     elt.Classes.remove(c)
            if len(parts) > 0:
                if isinstance(parts[0], nn.Module) :
                    layer = OntoOrLayer(parts)
                else:
                    masks = []
                    for m in parts :
                        if len(m.shape) == 1:
                            masks.append(torch.stack([m]))
                        else:
                            masks.append(m)
                    layer = torch.cat(tuple(masks),0).int()
            self._explainers[elt] = layer
            self._layers_cache.add(elt, layer)
            return layer

        elif isinstance(elt, owlready2.Not):
            part = self.__parse(elt.Class, encoder=encoder)
            layer = OntoNotLayer(part)
            self._explainers[elt] = layer
            self._layers_cache.add(elt, layer)
            return layer

        elif isinstance(elt, owlready2.Restriction):
            prop = elt.property
            
            if isinstance(prop, owlready2.ObjectPropertyClass):
                try:
                    prop_str = prop.name
                    part = self.__parse(elt.value, encoder=self._targetted_ontofeatures[prop.iri].get_encoder())

                    if part is None:
                        raise Exception("PROBLEM with  " + str(elt) + " : No property wrapper")
                        # pprint("[ IGNORED : " + str(elt) + " ]")
                        # return None

                    feature_input_focus = self._feature_input_focus[prop.iri]
                except KeyError as err:
                    if encoder is None:
                        raise Exception("PROBLEM with  " + str(elt) + " : No property wrapper")
                        # return None

                    # ic(err, elt)
                    # ontology_cloned = self.ontology.clone()
                    # with ontology_cloned:
                    if self._anonymous_classes_cache.contains(elt):
                        subs = self._anonymous_classes_cache.get(elt)
                    else:
                        with self.ontology:
                            try: self.temp_class_count += 1
                            except: self.temp_class_count = 0
                            new_class = type("temp_class_%d"%self.temp_class_count, (Thing,), {})
                            new_class.equivalent_to.append(elt)
                            # print("- ", str(elt))
                            # ic(new_class.equivalent_to)
                            self.syncReasoner()
                            subs = OntoClassifierHelper.get_subs([new_class])
                            subs.remove(new_class)
                            # ic(subs)
                            new_class.equivalent_to = []
                            destroy_entity(new_class)
                            # self.syncReasoner()
                            self._anonymous_classes_cache.add(elt, subs)
                            
                    if len(subs) > 0 :
                        # destroy_entity(new_class)
                        # del new_class
                        mask = self._getMaskEquiv(subs, encoder)
                        return mask
                    else:
                        # ICI TEST
                        # raise Exception("PROBLEM with  " + str(elt) + " : property not present")
                        ic("[ IGNORED : " + str(elt) + "]")
                        return None

                mask = part
                if elt.type == owlready2.SOME:
                    if len(mask.shape) == 1: mask = torch.stack([mask])
                    layer = OntoSomeLayer(mask, feature_name=prop_str, input_focus=feature_input_focus)
                elif elt.type == owlready2.ONLY:
                    if len(mask.shape) == 1: mask = torch.stack([mask])
                    layer = OntoOnlyLayer(mask, feature_name=prop_str, input_focus=feature_input_focus)
                elif elt.type == owlready2.VALUE:
                    layer = OntoValueLayer(mask, feature_name=prop_str, input_focus=feature_input_focus)
                elif elt.type == owlready2.MIN:
                    layer = OntoMinLayer(elt.cardinality, mask, feature_name=prop_str, input_focus=feature_input_focus)
                elif elt.type == owlready2.MAX:
                    layer = OntoMaxLayer(elt.cardinality, mask, feature_name=prop_str, input_focus=feature_input_focus)
                elif elt.type == owlready2.EXACTLY:
                    layer = OntoExactlyLayer(elt.cardinality, mask, feature_name=prop_str, input_focus=feature_input_focus)
                else:
                    print("PROBLEM : NO MATCH FOR " + str(elt))
                self._explainers[elt] = layer
                self._layers_cache.add(elt, layer)
                return layer

            elif isinstance(elt.property, owlready2.DataPropertyClass):

                try:
                    feature_input_focus = self._feature_input_focus[prop.iri]
                except KeyError:
                    # ICI TEST
                    raise Exception("PROBLEM with  " + str(elt) + " : data property not present")
                    # pprint("[ IGNORED 😵 : " + str(elt) + " ]")
                    # return None

                prop_str = prop.name
                if elt.type == owlready2.VALUE:
                    elt_type = "VALUE"
                    layer = OntoDatatypeEQLayer(elt.value, feature_name=prop_str, input_focus=feature_input_focus)
                else:
                    constrainedDatatype = elt.value
                    # part, explainer = self.__parse(elt.value, encoder=None)
                    feature_input_focus = self._feature_input_focus[prop.iri]
                    layer = OntoConstrainedDatatypeLayer(constrainedDatatype, feature_name=prop_str, input_focus=feature_input_focus)

                self._explainers[elt] = layer
                self._layers_cache.add(elt, layer)
                return layer

            else:
                ic(elt.property, "(", type(elt.property), ") cas non traité")

        elif isinstance(elt, owlready2.ObjectPropertyClass):
            self._explainers[elt] = elt
            return elt

        elif isinstance(elt, owlready2.ThingClass):
            try:
                # TODO  
                isA_features_enc = self._targetted_ontofeatures[owlready2.ThingClass.iri].get_encoder()
                # print(">> ", elt.name, isA_features_enc.classes_)
                if elt.iri in isA_features_enc.classes_:
                    mask = self._getMaskEquiv(elt, isA_features_enc).any(0).int()
                    # print (">>", elt, " mask = ", mask, mask.shape, mask.any(0))
                    feature_input_focus = self._feature_input_focus[owlready2.ThingClass.iri]
                    layer = OntoIsALayer(mask, feature_name=elt.name, input_focus=feature_input_focus)
                    self._explainers[elt] = layer
                    # print(">> ", elt.name, mask.shape)
                    self._layers_cache.add(elt, layer)
                    return layer
            except:
                pass             
            
            if not encoder is None :
                mask = self._getMaskEquiv(elt, encoder)
                if mask.sum() == 0:
                    pprint("[ IGNORED : " + str(elt) + " ]")
                    return None

                self._explainers[elt] = mask
                # print("<<>>", elt.iri, mask.shape)
                return mask
            else:
                equiv = []
                for c in elt.equivalent_to:
                    if isinstance(c, owlready2.ThingClass):
                        if not elt in self.getParentsOf(c, True, True):
                            equiv.append(c)
                    else:
                        equiv.append(c)

                part = self.__parse(equiv)
                
                # print(">>>> encoder none ", elt.iri, part)
                
                self._explainers[elt] = part
                return part

        else:
            if not encoder is None :
                mask = self._getMaskEquiv(elt, encoder)
                self._explainers[elt] = mask
                return mask
            else:
                ic("TODO[" + str(elt) + "]")
                return None

    def removeTargettedFeature(self, property):
        iri = property.iri
        try:
            self._targetted_ontofeatures.pop(iri)
        except Exception as e:
            print("ERROR: " + property.iri + " is not in targetted features", file=sys.stderr)

    def saveEntities(self, filename):
        entities = {
            OntoClassifier.CLASSES_KEY : set(),
            OntoClassifier.FEATURES_KEY : {}
        }

        for c in self._targetted_classes:
            entities[OntoClassifier.CLASSES_KEY].add(c.iri)

        for iri, elements in self._targetted_ontofeatures.items():
            elts = []
            for e in elements.get_range():
                try:
                    elts.append(e.iri)
                except:
                    elts.append(e)
            entities[OntoClassifier.FEATURES_KEY][iri] = elts

        dump(entities, filename)

    def setTargettedClasses(self, classes):
        
        checked_target_classes = []
        if not isinstance(classes, list):
            target_classes = [classes] # donc ça sert à rien. # TODO si je corrige, tests échouent. 
        for target_class in classes:
            if target_class in self.ontology.ontology.classes():
                checked_target_classes.append(target_class)
        if len(checked_target_classes) != len(classes):
            pprint(
                "Unable to retrieve some target classes. Keeping only: "
                + str(checked_target_classes)
            )

        self._targetted_classes = set()
        for c in checked_target_classes:
            self._targetted_classes.add(c)
        self.__buildTargetClassEnc()

    def setTargettedFeature(self, onto_feature):
        self._targetted_ontofeatures[onto_feature.get_iri()] = onto_feature


    def syncReasoner(self):
        with self.inferences:  # place le résultat du raisonnement dans cette ontologie
            owlready2.sync_reasoner(debug=0, infer_property_values=True)
        return self.inferences


class DictionaryCache:
    def __init__(self):
        self.cache = {}

    def add(self, key, value):
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def contains(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()
        
# class AddTargettedFeatureException(Exception):
#     def __init__(self, e1, e2, message=""):
#         msg = "PROBLEM with %s and %s ! " % (e1, e2)
#         msg += message
#         super().__init__(msg)
