# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

class OCEViz:

    @staticmethod
    def show_boxes(img, boxes, class_names, colors=None, title="", figsize=(4,4)):
        img = np.array(img.permute(1, 2, 0), dtype=np.uint8).copy()
        
        if not colors:
            colors = [
                "lightgrey",
                "blue",
                "rosybrown",
                "magenta",
                "red",
                "yellow",
                "forestgreen",
                'red',
                'purple',
                'darkgreen',
                'pink',
                'gray',
                'darkorange',
                'lightsalmon',
                'brown',
                'lightseagreen',
                'lightcoral',
                'lightskyblue',
                'lightsteelblue',
            ]

        if boxes is None or len(boxes) == 0:
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.legend(handles=[patches.Patch(color='white', label='Nothing')], loc='center left', bbox_to_anchor=(1, 0.8), fontsize='small', title='Found:')
            plt.title(title)
            plt.show()
            return
        
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)
        
        legend_patches = []
        class_counts = defaultdict(int)
        
        for box in boxes[-1]:
            x1, y1, x2, y2 = np.array(box[:4].int())
            if x2 == 0 and y2 == 0:
                continue
            if x2 - x1 == 0 or y2 - y1 == 0:
                continue
            key = int(box[-1])
            
            if key not in class_names:
                continue
            
            color = colors[key % len(colors)]
            class_counts[key] += 1
            
            # Dessiner les boîtes de délimitation
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Annoter les boîtes avec les noms des classes
            # plt.text(x1, y1, class_name, color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.5))
        
        # Créer les patches de légende avec le nombre d'instances
        for key, count in class_counts.items():
            # Convertir les noms des classes en chaînes de caractères
            if isinstance(class_names[key], list):
                class_name = "[" + class_names[key][0]
                for i in range(1, len(class_names[key])):
                    class_name += " & " + class_names[key][i]
                class_name += "]"
            else:
                class_name = class_names[key]
            color = colors[key % len(colors)]
            legend_label = f"{count} {class_name}"
            legend_patches.append(patches.Patch(color=color, label=legend_label))
                    
        # Ajouter la légende à côté de l'image
        plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.8), fontsize='small', title='Found:')
        # Ajouter le titre
        plt.title(title)
        plt.axis('off')
        
        # Ajuster la mise en page pour éviter la superposition
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Réserver de l'espace à droite pour la légende
        
        plt.show()
    