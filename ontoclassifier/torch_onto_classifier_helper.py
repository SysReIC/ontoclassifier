import owlready2
import torch
from functools import cmp_to_key


class OntoClassifierHelper:
    @staticmethod
    def bin2int(inputs):
        dim = len(inputs.shape) - 1
        mask = 2 ** torch.arange(inputs.shape[-1] - 1, -1, -1, dtype=torch.int64).to(
            inputs.device, torch.int64 #inputs.dtype
        )  # dtype important pour le bitwise_and !!!
        return torch.sum(mask * inputs, dim)

    @staticmethod
    def int2bin(inputs):
        if not inputs.any():
            return torch.zeros_like(inputs, dtype=torch.int8).unsqueeze(0)

        shape = inputs.shape
        view = inputs.view(shape[0] * shape[1])
        maximum = inputs.max()
        nb_bits = OntoClassifierHelper.nb_bits_for(maximum)
        mask = 2 ** torch.arange(nb_bits - 1, -1, -1, dtype=torch.int64).to(
            inputs.device, torch.int64 #inputs.dtype
        )  # dtype important pour le bitwise_and !!!
        mask = mask.unsqueeze(-1)
        result = view.bitwise_and(mask).ne(0).int().transpose(0, 1)
        return result.view(shape[0], shape[1], -1)

    @staticmethod
    def nb_bits_for(value):
        return int((torch.log2(value) + 1).int())

    @staticmethod
    def partition(entity_list):
        partitions = []
        hierarchies = []
        instances = []
        classes = []
        classes_to_remove = []

        for e in entity_list:
            if type(e) == owlready2.ThingClass:
                classes.append(e)
            else:
                instances.append(e)
                for c in e.is_a:
                    classes.append(c)
                    if c not in entity_list:
                        classes_to_remove.append(c)

        for c in classes:
            found = False
            for idx, h in enumerate(hierarchies):
                if c in h:
                    found = True
                    partitions[idx].add(c)
            if not found:
                new_partition = {c}
                partitions.append(new_partition)
                new_hierarchy = {c}
                new_hierarchy = new_hierarchy.union(c.ancestors())
                new_hierarchy = new_hierarchy.union(c.descendants())
                hierarchies.append(new_hierarchy)

            for i in instances:
                for c in i.is_a:
                    for idx, h in enumerate(hierarchies):
                        if c in h:
                            partitions[idx].add(i)

            for c in classes_to_remove:
                for p in partitions:
                    try:
                        p.remove(c)
                    except:
                        pass

        return [list(p) for p in partitions]

    @staticmethod
    def compare_entities(c1, c2):
        if type(c1) != owlready2.ThingClass and type(c2) != owlready2.ThingClass:
            return 0

        try:
            if isinstance(c1, c2):
                return 1
        except:
            pass

        try:
            if isinstance(c2, c1):
                return -1
        except:
            pass

        if c1 == c2:
            return 0

        c1_ancestors = c1.ancestors()
        hierarchy = c1_ancestors.union(c1.descendants())
        if c2 not in hierarchy:
            return 0
            print(c1.name, "::",  hierarchy)
            print(c2.name,)
            raise Exception(
                "%s and %s are not in the same hierachy" % (c1.name, c2.name)
            )
        c1_ancestors.remove(c1)
        if c2 in c1_ancestors:
            return 1
        else:
            return -1

    @staticmethod
    def sort_entities(homogen_class_list):
        new_list = homogen_class_list.copy()
        new_list.sort(key=cmp_to_key(OntoClassifierHelper.compare_entities))
        return new_list

    @staticmethod
    def get_leaf(homogen_class_list):
        new_list = OntoClassifierHelper.sort_entities(homogen_class_list)
        return new_list[-1]

    @staticmethod
    def get_leafs(heterogen_class_list):
        partitions = OntoClassifierHelper.partition(heterogen_class_list)
        leafs = []
        for p in partitions:
            leafs.append(OntoClassifierHelper.get_leaf(p))
        return leafs

    @staticmethod
    def get_possible_range(property):
        result = set()
        for cl in property.range:
            result.update(OntoClassifierHelper.__get_possible_range_part(cl))
        return list(result)

    @staticmethod
    def __get_possible_range_part(range_part):
        result = set()
        if isinstance(range_part, owlready2.class_construct.Or):
            for cl in range_part.get_Classes():
                result.update(OntoClassifierHelper.__get_possible_range_part(cl))
        else:
            result.add(range_part)

            if type(range_part) == owlready2.entity.ThingClass:
                for i in range_part.instances():
                    result.add(i)
            try:
                for sub in OntoClassifierHelper.get_subs([range_part]):
                    result.add(sub)
                    if type(sub) == owlready2.entity.ThingClass:
                        for i in sub.instances():
                            result.add(i)
            except AttributeError:
                pass
        return result

    @staticmethod
    def get_subs(onto_class_list):
        try:  # check if list or single entity
            _ = iter(onto_class_list)
        except TypeError:
            onto_class_list = [onto_class_list]

        subs = set()
        for c in onto_class_list:
            subs.add(c)
            ssubs = list(c.subclasses())
            if len(ssubs) != 0:
                new_leafs = set(OntoClassifierHelper.get_subs(ssubs))
                subs = subs.union(new_leafs)
        return list(subs)
