# coding: utf-8

import math
import torch
import torch.nn as nn
from ontoclassifier.wrappers import PropertyWrapper

import time

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
        total_time = 0
        extracted_features = list()
        for model, pwrappers in self.model_to_pw.items():
            if model:
                start_time = time.time()
                result = model.model(x)
                end_time = time.time()

                with open('/tmp/oc_stats.csv', 'a') as file:
                    file.write(str(end_time - start_time) + ';')
                # print("Model time: " + str(end_time - start_time))
            else:
                # if no model, output = input
                result = x
            # start_time = time.time()
            for pw in pwrappers:
                extracted_features.append(pw.extract_from(result))
            # end_time = time.time()
            # total_time += end_time - start_time
                
        # print ("batch time: %.3f " % total_time)
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
            # si jamais on recoit un stack de stack...
            # on met les stacks bout à bout 
            corrected_x = torch.row_stack(torch.unbind(corrected_x))
            # print(" > input reshaped", corrected_x.shape)        
            
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
            # print(" > input padded", corrected_x.shape)
        return corrected_x
    
    
    def forward(self, x):
        return super().forward(self.correct_inputs(x))
        


        # extracted_features = list()
        # for model, pwrappers in self.model_to_pw.items():
        #     result = model.model(corrected_x)
        #     for pw in pwrappers:
        #         #TODO result 
        #         extracted_features.append(pw.extract_from(result))
        # # concatenate extracted features and return it
        # merged_extracted_features = torch.cat(extracted_features, dim=1)
        # print("FE output:", merged_extracted_features.shape)
        # return merged_extracted_features
    
    