import abc
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod

from ontoclassifier import OntoClassifierHelper

class OntoFocusedInputLayer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, feature_name=None, input_focus=None):
        super().__init__()
        if (feature_name is not None and type(feature_name) != str):
            raise TypeError("feature_name sould be a string")
        if (input_focus is not None and type(input_focus) != tuple):
            raise TypeError("input_focus sould be a tuple")
        self._feature_name = feature_name
        self._input_focus = input_focus

    def getFocusedInputs(self, inputs):
        if(self._input_focus is not None):
            start, end = self._input_focus
            inputs = inputs[:, start:end]
        return inputs

    def focusedInputs2originalShape(self, focusedInputs, originalShape):
        if(self._input_focus is not None):
            start, end = self._input_focus
            inputs = torch.cat( (torch.zeros(originalShape[0], start), 
                                 focusedInputs, 
                                 torch.zeros(originalShape[0], originalShape[1]-end)),
                               1) 
        else:
            inputs = focusedInputs
        return inputs
    
    @abstractmethod
    def forward(self, inputs):
        pass

class OntoMaskLayer(OntoFocusedInputLayer):
    def __init__(self, mask, feature_name=None, input_focus=None):
        super().__init__(feature_name, input_focus)
        if (type(mask) != torch.Tensor):
            raise TypeError("mask sould be an instance of torch.Tensor")
        self._mask = mask

    def checkDevice(self, inputs):
        dev = inputs.get_device()
        if (dev > -1):
            if torch.cuda.is_available(): # CUDA (NVIDIA)
                self._mask = self._mask.to(dev)
            if torch.backends.mps.is_available():  # MPS (MAC M1)
                self._mask = self._mask.to(torch.device("mps"))
        else:
            self._mask = self._mask.cpu()

    @abstractmethod
    def explain(self, value, inputs):
        pass

    def getMask(self):
        return self._mask

    def __repr__(self):
        feature_info = self._feature_name + " : " if self._feature_name is not None else ""
        return self.__class__.__name__ \
               + "(" \
               + feature_info \
               + str(np.array(self._mask.cpu())) \
               + ")"

    @abstractmethod
    def forward(self, inputs):
        pass

class OntoSimplifiedInputLayer(OntoMaskLayer):
    @abstractmethod
    def forward(self, inputs):
        self.checkDevice(inputs)
        inputs = self.getFocusedInputs(inputs)
        res = torch.where(inputs>0,
                             torch.ones_like(inputs),
                             torch.zeros_like(inputs))
        return res

class OntoPerfectMatchLayer(OntoSimplifiedInputLayer):
    def forward(self, inputs):
        raise Exception('TODO')

    def explain(self, value, inputs):
        raise Exception('TODO')


class OntoSomeLayer(OntoSimplifiedInputLayer):
    def forward(self, inputs):
        self.checkDevice(inputs)
        # res_sum = torch.einsum('ikj, lk->ilj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        # result = res_sum == self._mask.sum(dim=1).unsqueeze(1)
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.einsum('ikj, lk->ilkj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        res_sum = result.sum(dim=2) 
        result = res_sum == torch.tile(self._mask.sum(dim=1), (focused_inputs.shape[0], res_sum.shape[2],1)).transpose(1,2)
        return result.any(dim=1).any(dim=1)

    def explain(self, value, inputs):
        self.checkDevice(inputs)
        # res_sum = torch.einsum('ikj, lk->ilj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        # final = res_sum >= self._mask.sum(dim=1).unsqueeze(1)
        # explanation = torch.einsum('ij, ilj->ilj', final.any(dim=1), res_sum)
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.einsum('ikj, lk->ilkj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        res_sum = result.sum(dim=2) 
        final = res_sum >= torch.tile(self._mask.sum(dim=1), (focused_inputs.shape[0], res_sum.shape[2],1)).transpose(1,2)
        explanation = torch.einsum('ij, iklj->ilj', final.any(dim=1), result)
        return explanation

class OntoOnlyLayer(OntoSimplifiedInputLayer):
    def forward(self, inputs):
        self.checkDevice(inputs)
        bin_inputs = self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs))
        # focused_inputs = self.getFocusedInputs(inputs)
        # print("bin_inputs ", bin_inputs.shape)
        # result = torch.einsum('ikj, lk->ilkj', bin_inputs, self._mask)#.transpose(0,1))
        # res_sum = result.sum(dim=2) 
        res_sum = torch.einsum('ikj, lk->ilj', bin_inputs.float(), self._mask.float()).int()  # .float() for MPS and CUDA support
        final = res_sum == self._mask.sum(dim=1).unsqueeze(1)
        # print("final ", final.shape)
        # inter = torch.tile(bin_inputs.any(dim=1).int(), (1,self._mask.shape[0]) ).reshape(final.shape)
        # print("bin_inpyts summé ", bin_inputs.any(dim=1).shape)
        # print("inter ", inter.shape)
        # final_ = torch.logical_or(final, 1-inter.int() ) # pour gérer le 0-fill des individus 
        final = torch.logical_or(final, 1-bin_inputs.any(dim=1).unsqueeze(1).int() ) # pour gérer le 0-fill des individus 
        # print("PAREILS ? ", final.equal(final_))
        # print("final!! ", final.shape)
        result = final.any(dim=1).all(dim=1)
        return result#.int()

    def explain(self, value, inputs):
        self.checkDevice(inputs)
        bin_inputs = self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs))
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.einsum('ikj, lk->ilkj', bin_inputs, self._mask)#.transpose(0,1))
        res_sum = result.sum(dim=2) 
        final = res_sum == torch.tile(self._mask.sum(dim=1), (focused_inputs.shape[0], res_sum.shape[2],1)).transpose(1,2)
        inter = torch.tile(bin_inputs.any(dim=1).int(), (1,self._mask.shape[0]) ).reshape(final.shape)
        final = torch.logical_or(final, 1-inter.int() ) # pour gérer le 0-fill des individus 
        explanation = torch.einsum('ijk, ik->ijk', bin_inputs , 1-final.any(dim=1).int() )
        return explanation

class OntoValueLayer(OntoSimplifiedInputLayer):
    def forward(self, inputs):
        inputs = super().forward(inputs)
        # inputs = self.getFocusedInputs(inputs)
        res = inputs * self._mask
        return torch.all(res == self._mask, dim=1)

    def explain(self, value, inputs):
        inputs = super().forward(inputs)
        if value:
            explanation = inputs * self._mask
        else:
            explanation = inputs
        explanation = OntoClassifierHelper.int2bin(explanation)
        return explanation

class OntoCardinalityLayer(OntoMaskLayer):
    def __init__(self, cardinality, mask, feature_name=None, input_focus=None):
        super().__init__(mask, feature_name, input_focus)
        self.cardinalyty = cardinality

    def __repr__(self):
        feature_info = self._feature_name + " : " if self._feature_name is not None else ""
        return self.__class__.__name__ \
                + "(" \
                + feature_info \
                + str(self.cardinalyty) + " : "\
                + str(np.array(self._mask.cpu())) \
               + ")"

    def getCount(self, inputs):
        self.checkDevice(inputs)
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.einsum('ikj, lk->ilkj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        res_sum = result.sum(dim=2) 
        result = res_sum == torch.tile(self._mask.sum(dim=1), (focused_inputs.shape[0], res_sum.shape[2],1)).transpose(1,2)
        return result.any(dim=1).sum(dim=1)

    def explain(self, value, inputs):
        self.checkDevice(inputs)
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.einsum('ikj, lk->ilkj', self.getFocusedInputs(OntoClassifierHelper.int2bin(inputs)), self._mask)
        res_sum = result.sum(dim=2) 
        final = res_sum >= torch.tile(self._mask.sum(dim=1), (focused_inputs.shape[0], res_sum.shape[2],1)).transpose(1,2)
        explanation = torch.einsum('ij, iklj->ilj', final.any(dim=1), result)
        ones = torch.ones(explanation.shape)
        explanation = torch.logical_and(explanation, ones).int()
        return explanation
    
class OntoMinLayer(OntoCardinalityLayer):
    def forward(self, inputs):
        count = self.getCount(inputs)
        return (count >= self.cardinalyty)#.int()

class OntoMaxLayer(OntoCardinalityLayer):
    def forward(self, inputs):
        count = self.getCount(inputs)
        return (count <= self.cardinalyty)#.int()

class OntoExactlyLayer(OntoCardinalityLayer):
    def forward(self, inputs):
        count = self.getCount(inputs)
        return (count == self.cardinalyty)#.int()

class StackLayer(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self._module_list = nn.ModuleList(module_list)

    def forward(self, inputs):
        results = []
        for m in self._module_list:
            results.append(m(inputs))
        return torch.stack(results)

class OntoAndLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self._stack = StackLayer(layers)

    def forward(self, inputs):
        inputs = self._stack(inputs)
        return (inputs.sum(dim=0) == inputs.shape[0])#.int()

class OntoOrLayer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self._stack = StackLayer(layers)

    def forward(self, inputs):
        inputs = self._stack(inputs)
        return (inputs.sum(dim=0) > 0)#.int()

class OntoNotLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, inputs):
        inputs = self._layer(inputs)
        return inputs.logical_not()#.int()


class OntoDatatypeLayer(OntoFocusedInputLayer):
    def __init__(self, number, feature_name=None, input_focus=None):
        super().__init__(feature_name, input_focus)
        if ( type(number) not in (int, float) ):
            raise TypeError("number sould be a number")
        self._number = number

    def checkDevice(self, inputs):
        dev = inputs.get_device()
        if (dev > -1):
            self._number = self._number.to(dev)
        else:
            self._number = self._number.cpu()

    def __repr__(self):
        feature_info = self._feature_name + " : " if self._feature_name is not None else ""
        return self.__class__.__name__ \
               + "(" \
               + feature_info \
               + str(np.array(self._number)) \
               + ")"

class OntoDatatypeLTLayer(OntoDatatypeLayer):
    def forward(self, inputs):
        inputs = self.getFocusedInputs(inputs)
        return torch.lt(inputs, self._number).squeeze(1)

class OntoDatatypeLELayer(OntoDatatypeLayer):
    def forward(self, inputs):
        inputs = self.getFocusedInputs(inputs)
        return torch.le(inputs, self._number).squeeze(1)

class OntoDatatypeGTLayer(OntoDatatypeLayer):
    def forward(self, inputs):
        inputs = self.getFocusedInputs(inputs)
        return torch.gt(inputs, self._number).squeeze(1)

class OntoDatatypeGELayer(OntoDatatypeLayer):
    def forward(self, inputs):
        inputs = self.getFocusedInputs(inputs)
        return torch.ge(inputs, self._number).squeeze(1)

class OntoDatatypeEQLayer(OntoDatatypeLayer):
    def forward(self, inputs):
        inputs = self.getFocusedInputs(inputs)
        return torch.eq(inputs, self._number).squeeze(1)

    def explain(self, value, inputs):
        inputs = self.getFocusedInputs(inputs)
        return inputs


class OntoConstrainedDatatypeLayer(OntoFocusedInputLayer):
    def __init__(self, constrainedDatatype, feature_name=None, input_focus=None):
        super().__init__(feature_name, input_focus)
        self._constrainedDatatype = constrainedDatatype
        self._module_list = nn.ModuleList()
        if hasattr(constrainedDatatype, 'min_inclusive'):
            number = constrainedDatatype.min_inclusive
            self._module_list.append(OntoDatatypeGELayer(number, feature_name, input_focus))
        if hasattr(constrainedDatatype, 'min_exclusive'):
            number = constrainedDatatype.min_exclusive
            self._module_list.append(OntoDatatypeGTLayer(number, feature_name, input_focus))
        if hasattr(constrainedDatatype, 'max_inclusive'):
            number = constrainedDatatype.max_inclusive
            self._module_list.append(OntoDatatypeLELayer(number, feature_name, input_focus))
        if hasattr(constrainedDatatype, 'max_exclusive'):
            number = constrainedDatatype.max_exclusive
            self._module_list.append(OntoDatatypeLTLayer(number, feature_name, input_focus))

    def __repr__(self):
        feature_info = self._feature_name + " : " if self._feature_name is not None else ""
        return self.__class__.__name__ \
               + "(" \
               + feature_info \
               + str(self._constrainedDatatype) \
               + ")"

    def forward(self, inputs):
        results = []
        for m in self._module_list:
            results.append(m(inputs))
        return torch.all(torch.stack(results), dim=0)

    def explain(self, value, inputs):
        inputs = self.getFocusedInputs(inputs)
        return inputs

class OntoIsALayer(OntoSimplifiedInputLayer):
    def forward(self, inputs):
        self.checkDevice(inputs)
        focused_inputs = self.getFocusedInputs(inputs)
        result = torch.logical_and(focused_inputs, self._mask)
        return result.any(dim=1)

    def explain(self, value, inputs):
        pass

