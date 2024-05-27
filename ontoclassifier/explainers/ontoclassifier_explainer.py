# coding=utf-8

from ontoclassifier import OntoClassifierHelper
from ontoclassifier.nn import OntoClassifier, OntoIsALayer
import owlready2

from icecream import ic


class OntoClassifierExplainer:
    def __init__(self, onto_classifier):
        self.onto_classifier = onto_classifier

    def __buildFound(self, elt, reason, tab):
        reason = reason[0]

        explanation = "\n  " + tab + "found "

        reason = reason.transpose(0, 1)
        onto_feature = self.onto_classifier.getTargettedFeatures()[elt.property.iri]
        encoder = onto_feature.get_encoder() 
        
        counts = {}
        for instance in reason:
            nb_classes = instance.sum()
            if nb_classes > 0:
                iris = tuple(encoder.inverse_transform(instance.unsqueeze(0))[0])
                try:
                    count = counts[iris]
                except KeyError:
                    count = 0
                counts[iris] = count + 1

        items = counts.items()
        last_idx = len(items) - 1
        details = ""
        for idx, (iris, count) in enumerate(items):
            if count > 1: details += "%d " % count
            classes = []
            for iri in iris:
                entity = self.onto_classifier.ontology.ontology.search(iri=iri)[0]
                classes.append(entity)
            classes = OntoClassifierHelper.get_leafs(classes)
            multiclass = len(classes) > 1
            if multiclass: details += "["
            for c in classes:
                details += "%s" % c.name
                if c != classes[-1]:
                    details += " & "
            if multiclass: details += "]"

            # multiclass = len(iris) > 1
            # if multiclass: details += "["
            # for iri in iris:
            #     entity = self.onto_classifier.ontology.search(iri=iri)[0]
            #     details += "%s" % entity.name
            #     if iri != iris[-1]:
            #         details += " & "
            # if multiclass: details += "]"

            if idx < last_idx:
                details += ", "

        if details:
            explanation += details
        else:
            explanation += "nothing"

        return explanation

    def explain(self, onto_class, features, verbose=True, filter_closest=False):
        input = features.unsqueeze(0)
        self.explanations = {}
        classification = self.onto_classifier.getExplainerFor(onto_class)(input).item()
        self.__explain_sub(onto_class, classification, input, verbose=verbose, filter_closest=filter_closest)
        print()
        return self.explanations, classification

    def __explain_sub(self, elt, classification, input, indent=0, verbose=True, filter_closest=False):

        tab = ""
        for i in range(indent) : tab = tab + " "

        try:
            layer = self.onto_classifier.getExplainerFor(elt)
        except:
            if verbose: print(tab + "[ EXPLAINER IGNORED : %s ]" % str(elt))
            return 0

        if type(elt) == owlready2.ThingClass:
            if not layer is None:
                # TODO: isinstance marche pas toujours ??? ???
                if isinstance(layer, OntoIsALayer) or "OntoIsALayer" in str(type(layer)):
                    result = layer(input).item()
                    explanation = "This is%sa %s " % (" " if result else " NOT ", elt.name)
                    if (verbose):
                        print(tab + explanation)
                    return 1
                else:
                    result = layer(input).item()
                    explanation = "This is%sa %s because : " % (" " if result else " NOT ", elt.name)
                    if (verbose):
                        print(tab + explanation)
                    equiv = elt._equivalent_to
                    if len(equiv) > 0 :
                        nb = 0
                        for c in equiv:
                            try:
                                if elt in c._equivalent_to:
                                    continue
                            except: pass
                            nb += self.__explain_sub(c, classification, input, indent+2, verbose, filter_closest=filter_closest)
                        return nb

        elif type(elt) == owlready2.And: #, owlready2.Or):
            result = layer(input).item()
            # op = OntoClassifier.OPERATORS_REPR[type(elt)]
            # count = 0
            # for c in elt.Classes:
            #     if not self.onto_classifier.getExplainerFor(c) is None : count += 1
            # if count > 1:
            #     explanation = "%s : %s" % (op, result)
            #     print(tab + explanation)
            
            nb = 0
            for c in elt.Classes:
                try:
                    c_layer = self.onto_classifier.getExplainerFor(c)
                    c_result = c_layer(input).item()
                    if (c_result == result): 
                        nb += self.__explain_sub(c, classification, input, indent, verbose, filter_closest=filter_closest)
                except:
                    self.__explain_sub(c, classification, input, indent, verbose, filter_closest=filter_closest)
            return nb
                
            # for c in elt.Classes:
            #     self.__explain_sub(c, classification, input, indent, verbose, filter_closest=filter_closest)

            # print(tab + "---")

        elif type(elt) == owlready2.Or: #, owlready2.Or):
            result = layer(input).item()

            if result != classification: return

            # op = OntoClassifier.OPERATORS_REPR[type(elt)]
            # count = 0
            # for c in elt.Classes:
            #     if not self.onto_classifier.getExplainerFor(c) is None : count += 1
            # if count > 1:
            #     explanation = "%s : %s" % (op, result)
            #     print(tab + explanation)

            # for c in elt.Classes:
            #     self.__explain_sub(c, classification, input, indent, verbose, filter_closest=filter_closest)
            #

            nb = {}
            for c in elt.Classes:                
                try:
                    c_layer = self.onto_classifier.getExplainerFor(c)
                    c_result = c_layer(input).item()
                    if (c_result == result): 
                        nb[c] = self.__explain_sub(c, classification, input, indent, result == True or not(filter_closest), filter_closest=filter_closest)
                except:
                    self.__explain_sub(c, classification, input, indent, verbose, filter_closest=filter_closest)

            min_key = min(nb, key=lambda k: nb[k])
            if (result == False and filter_closest):
                self.__explain_sub(min_key, classification, input, indent, verbose, filter_closest=filter_closest)
            return nb[min_key]
            # print(tab + "---")

        elif type(elt) == owlready2.Not:
            # result = layer(input).item()
            # op = OntoClassifierHelper.OPERATORS_REPR[type(elt)]
            # cl = self.strForRestriction(elt.Class)
            # explanation = "%s %s is %s" % (op, cl, result)
            # print(tab + explanation)
            return self.__explain_sub(elt.Class, not classification, input, indent, verbose, filter_closest=filter_closest)
            # print(tab + "---")

        elif type(elt) == owlready2.Restriction:
            reason = None
            explanation = ""
            result = layer(input).item()

            nb_missing = 0

            if (type(elt.property) == owlready2.prop.ObjectPropertyClass):

                if elt.type == owlready2.class_construct.SOME:

                    if result != classification : return

                    if result:
                        reason = layer.explain(classification, input)
                        explanation = self.__buildFound(elt, reason, tab)
                        nb_missing = reason.any(dim=1).sum()
                    else: nb_missing = 1  

                elif elt.type == owlready2.class_construct.ONLY:
                    if not result:
                        reason = layer.explain(classification, input)
                        explanation = self.__buildFound(elt, reason, tab)
                    else:
                        reason = layer.explain(classification, input)

                elif elt.type == owlready2.class_construct.VALUE:
                    if result == classification :
                        try:
                            reason = layer.explain(classification, input)
                        except:
                            ic(elt, reason)
                        explanation = self.__buildFound(elt, reason, tab)
                        nb_missing = 0
                    else:
                        return

                elif hasattr(elt, 'cardinality'):
                    reason = layer.explain(classification, input)
                    explanation = self.__buildFound(elt, reason, tab)
                    nb_missing = abs(elt.cardinality - reason.any(dim=1).sum())

            elif (type(elt.property == owlready2.prop.DatatypeProperty)):
                if result != classification: return
                reason = layer.explain(classification, input)
                if not result :
                    explanation = "\n  " + tab + "found " + str(reason.item())

            else:
                ic(elt.property, "cas non traité")
            
            explanation = "%s is %s %s" % (self.strForRestriction(elt), result, explanation)
            if (verbose):
                print(tab + explanation)
            
            self.explanations[elt] = {}
            self.explanations[elt]['reason'] = reason
            self.explanations[elt]['result'] = result

            return nb_missing

        else:
            print("TODOOOO", type(elt))

    def strForConstrainedDataType(self, constrainedDataType, property):
        prop = property.name
        repr = ""

        if hasattr(constrainedDataType, 'min_inclusive'):
            repr = '%s >= %d' % (prop, constrainedDataType.min_inclusive)

        if hasattr(constrainedDataType, 'min_exclusive'):
            if len(repr) > 0 : repr += " and "
            repr += '%s > %d' % (prop, constrainedDataType.min_exclusive)

        if hasattr(constrainedDataType, 'max_inclusive'):
            if len(repr) > 0 : repr += " and "
            repr += '%s <= %d' % (prop, constrainedDataType.max_inclusive)

        if hasattr(constrainedDataType, 'max_exclusive'):
            if len(repr) > 0 : repr += " and "
            repr += '%s < %d ' % (prop, constrainedDataType.max_exclusive)

        return repr

    def strForRestriction(self, restriction):
        op = OntoClassifier.OPERATORS_REPR[restriction.type]
        value = restriction.value
        if type(value) == owlready2.class_construct.ConstrainedDatatype:
            return self.strForConstrainedDataType(value, restriction.property)
        else:
            try: value = value.name
            except: 
                if type(restriction.value) == owlready2.class_construct.Or:
                    value = restriction.value.Classes[0].name
                    for i in range(1, len(restriction.value.Classes)) :
                        value += " or " + restriction.value.Classes[i].name
                if (type(restriction.value) == owlready2.class_construct.Restriction):
                    value = "(" + self.strForRestriction(restriction.value) + ")"

            try:
                cardinalyty = restriction.cardinality
                op += " %d" % cardinalyty
            except: pass

        name = restriction.property.label[0] if len(restriction.property.label) > 0 else restriction.property.name

        return "%s %s %s" % (name, op, value)
