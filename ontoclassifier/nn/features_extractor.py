import math
import torch
import torch.nn as nn
from ontoclassifier.wrappers import PropertyWrapper


class FeaturesExtractor(nn.Module):
    
    def __init__(self, property_wrappers: list[PropertyWrapper]) -> None:
        super().__init__()
        self.model_to_pw = {}
        self.property_wrappers = list()
        for pw in property_wrappers:
            if pw.model not in self.model_to_pw.keys():
                self.model_to_pw[pw.model] = list()
            self.model_to_pw[pw.model].append(pw)
            self.property_wrappers.append(pw)
            
    def forward(self, x):
        extracted_features = list()
        for model, pwrappers in self.model_to_pw.items():
            if model:
                result = model.model(x)
            else:
                result = x
                
            for pw in pwrappers:
                extracted_features.append(pw.extract_from(result))

        # concatenate extracted features and return it
        merged_extracted_features = torch.cat(extracted_features, dim=1)
        ic("FE output:" + str(merged_extracted_features.shape))
        return merged_extracted_features



class Yolov8FeaturesExtractor(FeaturesExtractor):
    
    def __init__(self, property_wrappers: list[PropertyWrapper]) -> None:
        super().__init__(property_wrappers)
    
    def correct_inputs(self, x):
        if torch.max(x).int() > 1:
            corrected_x = x / 255
        else:
            corrected_x = x

        if len(x.shape) > 4:
            corrected_x = torch.row_stack(torch.unbind(corrected_x))
            
        # Padding input dimensions to be multiple of 32 
        # (otherwise may have exception in some cases)
        if ((corrected_x.shape[2] + corrected_x.shape[3]) % 32 > 0):
            multiple = 32
            padding1_mult = math.floor(corrected_x.shape[2] / multiple) + 1
            padding2_mult = math.floor(corrected_x.shape[3] / multiple) + 1
            pad1 = (multiple * padding1_mult) - corrected_x.shape[2]
            pad2 = (multiple * padding2_mult) - corrected_x.shape[3] 
            padding = torch.nn.ReplicationPad2d((0, pad2, 0, pad1))
            corrected_x = padding(corrected_x)
        return corrected_x
    
    
    def forward(self, x):
        return super().forward(self.correct_inputs(x))
        
    
    