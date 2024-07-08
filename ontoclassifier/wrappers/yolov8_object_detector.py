import torch
import torch.nn as nn
from ultralytics.utils import ops
import math


class Yolov8ObjectDetector(nn.Module):
    def __init__(self, 
                 yolov8_model, classes_to_detect=[], max_det=300, output_crops=False):
        super().__init__()
        
        self.output_crops = output_crops
        self.yolov8 = yolov8_model
        
        if len(classes_to_detect) == 0:
            self.classes_to_detect = list(self.yolov8.names.keys())
        else:
            yolo_classes = list(self.yolov8.names.keys())
            self.classes_to_detect = []
            for cl in classes_to_detect:
                if cl not in yolo_classes:
                    print("Class", cl, "not recognized by model")
                else:
                    self.classes_to_detect.append(cl)

        self.max_det = max_det

    def forward(self, inputs):
        if torch.max(inputs).int() > 1:
            corrected_inputs = inputs / 255
        else:
            corrected_inputs = inputs
        
        # Padding input dimensions to be multiple of 32 
        # (otherwise may have exception in some cases)
        if ((inputs.shape[2] + inputs.shape[3]) % 32 > 0):
            multiple = 32
            padding1_mult = math.floor(corrected_inputs.shape[2] / multiple) + 1
            padding2_mult = math.floor(corrected_inputs.shape[3] / multiple) + 1
            pad1 = (multiple * padding1_mult) - corrected_inputs.shape[2]
            pad2 = (multiple * padding2_mult) - corrected_inputs.shape[3] 
            padding = torch.nn.ReplicationPad2d((0, pad2, 0, pad1))
            corrected_inputs = padding(corrected_inputs)
                
        preds = self.yolov8.model(corrected_inputs) 
        
        yolov8_predictions = ops.non_max_suppression(
            preds,  
            classes=self.classes_to_detect,
            max_det=self.max_det,
        )

        nb_obj_max = max([item.shape[0] for item in yolov8_predictions])
        
        # stacking whole batch of results
        # (padding missing spaces with len(names))
        yolov8_predictions_tensor = torch.stack(
            [
                torch.nn.functional.pad(
                    item,
                    (0, 0, 0, nb_obj_max - item.shape[0]),
                    value=len(self.yolov8.names),
                )
                for item in yolov8_predictions
            ]
        )

        # remembering the last prediction for future explaining
        self.last_prediction = yolov8_predictions_tensor

        if not self.output_crops :
            return yolov8_predictions_tensor
        else:
            # finding biggest bbox from all preds:
            all_batch_preds_flattened = torch.row_stack(yolov8_predictions)
            all_width = (all_batch_preds_flattened[:,2] - all_batch_preds_flattened[:,0]).int()
            all_height = (all_batch_preds_flattened[:,3] - all_batch_preds_flattened[:,1]).int()
            upscale_shape = (inputs[0].shape[0], int(torch.max(all_height)), int(torch.max(all_width)))

            all_upscale_crops = []
            
            # crop all images from bbox and upscale to biggest
            for i, prediction in enumerate(self.last_prediction):
                crops = [inputs[i][:,box[1]:box[3],box[0]:box[2]] for box in prediction.int()]
                upscaled_crops = []
                
                for crop in crops:
                    if 0 in crop.shape:
                        crop2 = torch.zeros(upscale_shape)
                        upscaled_crops.append(crop2)
                    else:
                        upscaling = torch.nn.Upsample(size=upscale_shape)
                        crop2 = upscaling(crop.unsqueeze(0).unsqueeze(0))
                        upscaled_crops.append(crop2[0][0])
                        
                all_upscale_crops.append(torch.stack(upscaled_crops))
            all_upscale_crops = torch.stack(all_upscale_crops)

            # keeping only classes of each box (last column)
            result = torch.index_select(
                yolov8_predictions_tensor,
                2,
                torch.Tensor([yolov8_predictions_tensor.shape[2] - 1]).int(),
            ).squeeze(2) 


            return [torch.concat(torch.unbind(all_upscale_crops)), 
                    torch.concat(torch.unbind(result.int()))]

