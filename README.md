# OntoClassifier

The OntoClassifier is a Python module that uses domain ontologies to automatically generate ontologically explainable PyTorch classifier models.<br>

This proposition aligns with the XAI (eXplainable AI) research effort.

The role of a generated OntoClassifier model is to compute the satisfiability of ontological class expressions for individuals in order to provide explainable classification.
The generation process uses an OWL 2 ontology to automatically create a set of PyTorch layers capable of handling OWL 2 DL class expressions. This includes using all logical operators with property restrictions (both object and data type), as well as qualified cardinality restrictions.

The OntoClassifier does not aim to replace well-known ontological reasoners like Hermit or Pellet; for example, it will not be able to check ontology consistency. 
It aims to provide targeted, ontologically explainable classification based on class expressions, while addressing the slowness of traditional reasoners and their lack of integration into AI pipelines.
As with typical classification tasks, the OntoClassifier can quickly process batches of individuals. 
However, note that while the OntoClassifier handles batches of individuals, the reasoning process classifies each individual separately, as if each belongs to a distinct new A-Box.

## Architecture

The OntoClassifier implements a Concept-Based/Bottleneck (CBM) approach: concepts are detected by third-party ML models, and the OntoClassifier uses these concepts to perform multi-label classification according to ontological class expressions.

The following figure illustrates a typical AI pipeline using samples from the [Pizzaïolo Dataset](https://www.kaggle.com/datasets/arnaudlewandowski/pizzaolo-dataset/) : if an ML model (e.g., YoloV8) can detect ingredients on a pizza, the OntoClassifier can infer the pizza recipe and characteristics based on the 28 class expressions described in the [Pizzaïolo Ontology](https://zenodo.org/records/10165941).


<center>
<img src="https://github.com/SysReIC/ontoclassifier/raw/main/doc/images/ontoclassifier-approach.jpg" alt="Ontoclassifier approach" width="600" height="auto">
</center>

## XAI (eXplainable AI)

As shown in the following figure, the provided mechanisms can help in providing causal local explanations :

<center>
<img src="https://github.com/SysReIC/ontoclassifier/raw/main/doc/images/individual_classification.png" alt="Ontoclassifier approach" width="600" height="auto">
</center>


... as well as contrastive ones :

<center>
<img src="https://github.com/SysReIC/ontoclassifier/raw/main/doc/images/not_spicy_vege_pizza.png" alt="Ontoclassifier approach" width="600" height="auto">
</center>

### Rapid classification with instant explanations

The following video also demonstrates the speed of the detection pipeline in providing classification and explanations. 
We created a PokerHands ontology and trained a YoloV8 model to detect individual cards. 
The resulting hybrid pipeline classifies the 9 classes of poker hands (e.g., Royal Flush, Full House, ...) and displays the generated ontological explanations in real-time video streams.

https://github.com/SysReIC/ontoclassifier/assets/78211502/d8b7d6f3-1541-4f58-85d1-cdce4fbab8fe


## How to use it ?

See examples in 3 notebooks:

- [Pizzaiolo classification and explanations notebook](https://github.com/SysReIC/ontoclassifier/blob/main/examples/Pizzaiolo_pipeline.ipynb)
- [XTRAINS classification and explanations notebook](https://github.com/SysReIC/ontoclassifier/blob/main/examples/XTRAINS_pipeline.ipynb)
- [SCDB classification and explanations notebook](https://github.com/SysReIC/ontoclassifier/blob/main/examples/SCDB_pipeline.ipynb)
- [Streamlit web application for poker hands detection in live video stream](https://github.com/SysReIC/ontoclassifier/blob/main/examples/poker_app.py)


## Authors

Grégory Bourguin<sup>1</sup> & Arnaud Lewandowski<sup>2</sup>  
SysReIC (Systèmes Réflexifs et Ingenierie de la Connaissance)  
[LISIC](https://lisic-prod.univ-littoral.fr/) (Laboratoire Informatique Signal et Image de la Côte d'opale)  
[ULCO](https://www.univ-littoral.fr/) (Université du Littoral Côte d'Opale), FRANCE

<sup>1</sup> gregory.bourguin@univ-littoral.fr  
<sup>2</sup> arnaud.lewandowski@univ-littoral.fr


<center>

[<img src="https://lisic-prod.univ-littoral.fr/wp-content/uploads/2023/05/ULCO.png" alt="logo ULCO" width="auto" height="50">](https://lisic-prod.univ-littoral.fr) &nbsp;&nbsp;&nbsp;&nbsp; [<img src="https://lisic-prod.univ-littoral.fr/wp-content/uploads/2023/05/LISIC.png" alt="logo LISIC" width="auto" height="50">](https://www.univ-littoral.fr/)

</center>
