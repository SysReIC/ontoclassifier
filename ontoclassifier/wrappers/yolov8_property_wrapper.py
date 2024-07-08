import torch
from ultralytics.utils import ops
from ontoclassifier.torch_onto_classifier_helper import OntoClassifierHelper
from ontoclassifier.wrappers import PropertyWrapper
from ontoclassifier import OntoFeature

class YoloV8PropertyWrapper(PropertyWrapper):

    def __init__(self, yolov8_model, feature: OntoFeature, frange: dict={}):
        super().__init__(yolov8_model, feature, frange)
        self.max_det = 62 # TODO
        
    def init_feature_range(self, frange: dict = ...):
        self.feature_range_names = {}
        not_found = {}
        classes2prop_mapping = []
        
        if len(frange) == 0:
            # keep all yolov8_model.names as feature range classes
            for key, value_ in self.model.names.items():
                value = value_.split(".")[-1]
                self.feature_range_names[key] = value
                mapping = self.feature.encode_labels([[value]])
                classes2prop_mapping.append(mapping.squeeze(0))
                if mapping.sum() == 0:
                    not_found[key] = value
        else:
            # keep only the given feature range classes
            for key in self.model.names.keys():
                if key in frange.keys():
                    value_ = frange[key]
                    if isinstance(value_, list):
                        value = value_
                    else:
                        value = [value_]
                    self.feature_range_names[key] = value
                    mapping = self.feature.encode_labels([value])
                    classes2prop_mapping.append(mapping.squeeze(0))
                    if len(value) != mapping.sum():
                        not_found[key] = value
                else:
                    classes2prop_mapping.append(
                        torch.zeros((len(self.feature.get_range()))).int()
                    )
                    
        # Adding an empty tensor for "None" class
        classes2prop_mapping.append(
            torch.zeros((len(self.feature.get_range()))).int()
        )

        self.classes_to_prop = torch.stack(classes2prop_mapping)


    def extract_from(self, result):
        if isinstance(result, (list, tuple)):
            result = result[0]
        
        yolov8_predictions = ops.non_max_suppression(
            result.detach().clone(),
            classes=list(self.feature_range_names.keys()),
            max_det=self.max_det,
        )
        
        yolov8_predictions_tensor = torch.stack(
            [
                torch.nn.functional.pad(
                    item,
                    (0, 0, 0, self.max_det + 1 - item.shape[0]),
                    # +1 pour ajouter un item "vide" et être sur que le padding est complet.
                    value=len(self.model.names),
                )
                for item in yolov8_predictions
            ]
        )

        # remembering the last prediction for future explaining
        self.last_prediction = yolov8_predictions_tensor

        # keeping only classes of each box (last column)
        result = torch.index_select(
            yolov8_predictions_tensor,
            2,
            torch.Tensor([yolov8_predictions_tensor.shape[2] - 1]).int().to(device=self.model.device),
        ).squeeze(2)         
        
        # transforming yolo box classes to map ontoclassifier inputs
        class2prop_mapping = self.classes_to_prop.transpose(1, 0).flip([1])
        mask = (
            OntoClassifierHelper.int2bin(2 ** result.int())
            .unsqueeze(2)
            .tile((self.classes_to_prop.shape[1], 1))
        )

        ontoclassifier_inputs = torch.einsum("ijkl, kl->ikj",
                                             mask.float(), class2prop_mapping.float().to(device=self.model.device)).int()

        return OntoClassifierHelper.bin2int(ontoclassifier_inputs)
    