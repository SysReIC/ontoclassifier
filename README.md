# ontoclassifier

## What is it ?

Ontoclassifier is a module implementing a Concept-Based approach. From concepts detected by third-party ML models, the ontoclassifier can infer classes (predictions) corresponding to Description Logic rules. 

The ontoclassifier can be directly plugged into a pytorch AI pipeline. 

For example, if a ML model can detect ingredients on a pizza, the ontoclassifier can infer the pizza receipe and characteristics. Furthermore, this classification is fully transparent and explainable since it is based on logical rules. 

<center>
<img src="doc/images/ontoclassifier-approach.jpg" alt="Ontoclassifier approach" width="600" height="auto">
</center>

## How to use it ?

See examples in 3 notebooks:

- [Pizzaiolo classification notebook](examples/Pizzaiolo_pipeline.ipynb)
- [XTRAINS classification notebook](examples/XTRAINS_pipeline.ipynb)
- [SCDB notebook](examples/SCDB_pipeline.ipynb)


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
