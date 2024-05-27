# coding: utf-8

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class OCEViz:
    
    @staticmethod
    def update_title(fig, newtitle):
        # fig.update_layout(title=title)
        fig.update_layout(
            title=dict(text=newtitle, font=dict(size=13)) #, automargin=True, yref='paper') 
        )
    @staticmethod
    def show_boxes(img, boxes, class_names, colors=None):
        img = np.array(img.permute(1, 2, 0) , dtype=np.uint8).copy()

        x_ = {}
        y_ = {}
        labels_ = {}
        # fig = go.Figure()
        
        fig = px.imshow(img)

        # fig.update_layout(hovermode=False)

        # fig.update_traces(hovertext="none", selector=dict(type='image'))

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
            return fig
        
        for box in boxes[-1]:
            x1, y1, x2, y2 = np.array(box[:4].int())
            if x2 == 0 and y2 == 0:
                continue
            if x2-x1 == 0 or y2-y1 == 0:
                continue
            key = int(box[-1])
            if key not in x_.keys():
                x_[key] = []
                y_[key] = []
                labels_[key] = []
            x_[key].extend([x1, x2, x2, x1, x1, None])
            y_[key].extend([y1, y1, y2, y2, y1, None])
            # print(key, ":", x1, y1, x2, y2, "((", box, "))")
    
        for key in x_.keys():
            sub_x = x_[key]
            sub_y = y_[key]
            
            if key not in class_names.keys():
                continue
            if isinstance(class_names[key], list):
                class_name = "[" + class_names[key][0]
                for i in range(1,len(class_names[key])):
                    class_name += " & " + class_names[key][i]
                class_name += "]"
            else:
                class_name = class_names[key]
            label = str(sub_x.count(None)) + " " + class_name

            # key = int(box[-1])
            # sub_x = [x1, x2, x2, x1, x1]
            # sub_y = [y1, y1, y2, y2, y1]
            # label = class_names[key]

            # pour tracer une box avec mouseover
            fig.add_trace(
                go.Scatter(
                    x=sub_x,
                    y=sub_y,
                    # marker=dict(color=[int(box[-1])], colors=colors),
                    # TODO: use color scale
                    marker=dict(color=colors[key % len(colors)] ),
                    # marker=dict(color='black'),
                    fill="toself",
                    opacity=0.7,
                    mode="lines",
                    name=label,
                    legendgrouptitle=dict(text='Found :'),
                    # text=labels_[key],
                    hoverinfo='none',
                )
            )
        # fig.update_layout(hovermode='closest')

        # Dots pour afficher la proba de chaque box
        for box in boxes[-1]:
            x1, y1, x2, y2 = np.array(box[:4].int())
            if x2 == 0 and y2 == 0:
                continue
            key = int(box[-1])

            if key not in class_names.keys():
                continue
            if isinstance(class_names[key], list):
                class_name = "[" + class_names[key][0]
                for i in range(1,len(class_names[key])):
                    class_name += " & " + class_names[key][i]
                class_name += "]"
            else:
                class_name = class_names[key]

            fig.add_trace(
                go.Scatter(
                    x=[int((x1+x2)/2)],
                    y=[int((y1+y2)/2)],
                    # mode="text",
                    text=class_name, # + "\n " + str(round(float(box[-2]),2)),
                    # textposition="bottom center",
                    showlegend=False,
                    # marker=dict(color=[int(box[-1])], colors=colors),
                    # TODO: use color scale
                    marker=dict(color=colors[key % len(colors)] ),
                    # fill="none",
                    opacity=0,
                    # mode="lines",
                    name=str(round(float(box[-2]),2)),
                    # text=labels_[key],
                    hoverinfo='text+name',
                )
            )
            
        return fig