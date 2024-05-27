import torch
from ultralytics.utils import ops
from ontoclassifier.torch_onto_classifier_helper import OntoClassifierHelper
from ontoclassifier.wrappers import PropertyWrapper
from ontoclassifier import OntoFeature
import time

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
            # # keep only the given feature range classes
            # for key, value_ in frange.items():
            #     if isinstance(value_, list):
            #         value = value_
            #     else:
            #         value = [value_]
            #     self.feature_range_names[key] = value
            #     mapping = self.feature.encode_labels([value])
            #     classes2prop_mapping.append(mapping.squeeze(0))
            #     if len(value) != mapping.sum():
            #         not_found[key] = value
                    
        # Adding an empty tensor for "None" class
        classes2prop_mapping.append(
            torch.zeros((len(self.feature.get_range()))).int()
        )

        self.classes_to_prop = torch.stack(classes2prop_mapping)


    def extract_from(self, result):
        if isinstance(result, (list, tuple)):
            result = result[0]

        t1 = time.time()
        
        yolov8_predictions = ops.non_max_suppression(
            result.detach().clone(),
            classes=list(self.feature_range_names.keys()),
            # conf_thres=0.18,
            # iou_thres=0.1,
            max_det=self.max_det,
            #nc=len(self.model.names),
        )

        t2 = time.time()
        with open('/tmp/oc_stats.csv', 'a') as file:
            file.write(str(t2 - t1) + ';')
        # print (" nms :", str(t2 - t1) )
        
        # batch inference working ?
        # see https://github.com/ultralytics/ultralytics/issues/1310

        # stacking whole batch of results
        # (padding missing spaces with len(names))
        # TODO: OR TRUNCATE IF MORE THAN max_det BOXES
        
        # print ("ITEMS to stack ; size ")
        # for item in yolov8_predictions:
        #     print("  - ", item.shape)
        #     print("    original: ", item)
        #     print("    padded: ", torch.nn.functional.pad(
        #             item,
        #             (0,0,0, 10 + 1 - item.shape[0]),
        #             # +1 pour ajouter un item "vide" et être sur que le padding est complet.
        #             value=len(self.model.names),
        #         ))
        # print("TRY Stacking")
        # try:
        #     torch.stack(yolov8_predictions)
        # except:
        #     print("error")
        
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

        # print(yolov8_predictions_tensor)
        
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
        ## TODO : si le nb de classes est trop grand (>31) ça marche plus...(à cause du 2**result)
        ## Donc restrictions : NB classes < 32 ET  NB détections < 32 ... bof bof bof. Faudrait déjà passer à 64
        # print(mask.shape, class2prop_mapping.shape)
        ontoclassifier_inputs = torch.einsum("ijkl, kl->ikj",
                                             mask.float(), class2prop_mapping.float().to(device=self.model.device)).int()

        # t3 = time.time()
        # print (" stacking :", str(t3 - t2) )

        return OntoClassifierHelper.bin2int(ontoclassifier_inputs)
    