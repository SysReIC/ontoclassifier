# Start the app with: streamlit run poker_app.py

import sys
sys.path.append('..')

import streamlit as st
import cv2
import numpy as np
import time
import datetime
from matplotlib import colors

from ultralytics import YOLO

import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from ontoclassifier import *
import os

st.set_page_config(layout="wide")

# ONTOCLASSIFIER CREATION 

@st.cache_resource
def get_yolo_model():
    return YOLO('data/cards-yolov8.pt')

@st.cache_resource
def get_ontoclassifier():
    onto_filename = "data/cards.owl"
    cards = owlready2.get_ontology(onto_filename).load()

    target_classes=[
                    cards.RoyalFlush, 
                    cards.StraightFlush, 
                    cards.FourOfAKind,
                    cards.FullHouse, 
                    cards.Flush,
                    cards.Straight, 
                    cards.ThreeOfAKind,
                    cards.TwoPairs, 
                    cards.Pair, 
                    ]

    onto_classifier = OntoClassifier(cards, target_classes=target_classes, ontological_extractor=None)

    property = cards.hasCard
    property_range = list(cards.Card.subclasses())
    feature = OntoFeature(property, property_range)
    onto_classifier.setTargettedFeature(feature)
    onto_classifier.buildGraph()

    print('Loaded ', onto_classifier.ontology.base_iri)
    return onto_classifier, target_classes

@st.cache_resource
def get_explainer(_onto_classifier):
    return OntoClassifierExplainer(_onto_classifier)


model = get_yolo_model()
onto_classifier, target_classes = get_ontoclassifier()
explainer = get_explainer(onto_classifier)


def encode_hand(hand):
    cards = []
    for card_str in hand:
        cards.append(eval('onto_classifier.ontology.'+card_str+"()"))
        
    unknown = onto_classifier.ontology.Hand()
    unknown.hasCard = cards 

    return onto_classifier.encode(unknown)


# CONSTANTS 

@st.cache_data
def get_model_names():
    NAMES = model.names
    for i in range(len(NAMES)):
        NAMES[i] = str(NAMES[i][-1]).upper() + str(NAMES[i][0:-1])
    return NAMES


NAMES = get_model_names()
    
FRAME_WIDTH = 600
FRAME_HEIGHT = 400

CONFIDENCE_THRESHOLD = 0.5 # seuil de détection des cartes 

COLORS = ['orange', 'blue', 'green', 'red', 'violet', 'gray', 'yellow', 'brown', 'pink', 'cyan']


# Streamlit app
run = st.checkbox('Run')
st.title("Real-time Poker Hand Detection")

col1, col2 = st.columns(2, gap="large")
col2 = col2.empty()

FRAME_WINDOW = col1.image([])

bottom = st.empty()

camera = cv2.VideoCapture(0)
camera.set(3, FRAME_WIDTH)
camera.set(4, FRAME_HEIGHT)

last_detections = []
NB_LAST_DETECTIONS = 10 
DETECTION_RATIO = 0.6
CARDS_COLORS = dict()

HAND = []

run_times = []
yolo_times = []
onto_classifier_times = []

while run:

    start = datetime.datetime.now()
    success, frame = camera.read()
    
    if not success:
        time.sleep(0.1)
        continue
    
    start_yolo = datetime.datetime.now()
    results = model.predict(frame, stream=True, iou=0.4, verbose=False)
    detections = next(results).boxes.data
    yolo_times.append((datetime.datetime.now() - start_yolo).total_seconds())
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des cartes sur l'image
    condition = detections[:,4] > CONFIDENCE_THRESHOLD
    detected_cards = detections[condition, -1].int().tolist()


    last_detections.append(detected_cards)
    if (len(last_detections)==NB_LAST_DETECTIONS):
        last_detections.pop(0)
        
    # Pour lisser les détections Yolo qui peuvent "clignoter" : 
    # on ne retient que les détections qui apparaissent un certain nombre de fois dans les dernières images
    fresh_hand = []
    for i in range(52):
        nb = 0 
        for j in range(len(last_detections)):
            nb = nb + last_detections[j].count(i)
        
        if nb > DETECTION_RATIO * len(last_detections) :
            fresh_hand.append(NAMES[i])

    if (len(set(fresh_hand)) != 5):
        CLASSE = "-"
        textual_expl = ""
        CARDS_COLORS = dict()
        HAND = []
        
    if (len(HAND)==0 or HAND != set(fresh_hand)) and len(fresh_hand) == 5:
        HAND = set(fresh_hand)
        
        CLASSE = "-"
        textual_expl = ""
        CARDS_COLORS = dict()

        # PREDICTION DE LA MAIN
        hand = encode_hand(HAND)
        
        onto_classifier_start = datetime.datetime.now()
        prediction = onto_classifier(hand.unsqueeze(0))
        onto_classifier_times.append((datetime.datetime.now() - onto_classifier_start).total_seconds())
        
        class_encoder = onto_classifier.getTargettedClassesEncoder()
        prediction = [name.replace(onto_classifier.ontology.get_base_iri(),'') for name in class_encoder.inverse_transform(prediction)[0]]
        
        for p in target_classes:
            if p.name in prediction:
                CLASSE = str(p.name)
                detailed_expl, _, textual_expl = explainer.explain(p, hand, verbose=False)
                
                # POUR RAJOUTER UN TRAIT EN COULEUR (LA LEGENDE) À CHAQUE "FOUND" DANS L'EXPLICATION
                chunks = textual_expl.split('\n')
                i_color = 0
                for i, chunk in enumerate(chunks):
                    chunks[i] = chunks[i].replace('  ', '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;')
                    if ' found ' in chunk:
                        for name in NAMES.values():
                            if name in chunk:
                                CARDS_COLORS[name] = COLORS[i_color]
                        chunks[i] += ' &nbsp;&nbsp; <span class="big-font"> :'+COLORS[i_color]+'[**⎯**]</span>'
                        i_color = (i_color + 1) % len(COLORS)
                
                # POUR AFFICHER LES ICONES DE CARTE AU LIEU DE LEUR NOM
                textual_expl = '\n\n'.join(chunks)
                textual_expl = textual_expl.replace('D2', '<span class="big-font">🃂</span>').replace('S2', '<span class="big-font">🂢</span>').replace('H2','<span class="big-font">🂲</span>').replace('C2', '<span class="big-font">🃒</span>')
                textual_expl = textual_expl.replace('D3', '<span class="big-font">🃃</span>').replace('S3', '<span class="big-font">🂣</span>').replace('H3','<span class="big-font">🂳</span>').replace('C3', '<span class="big-font">🃓</span>')
                textual_expl = textual_expl.replace('D4', '<span class="big-font">🃄</span>').replace('S4', '<span class="big-font">🂤</span>').replace('H4','<span class="big-font">🂴</span>').replace('C4', '<span class="big-font">🃔</span>')
                textual_expl = textual_expl.replace('D5', '<span class="big-font">🃅</span>').replace('S5', '<span class="big-font">🂥</span>').replace('H5','<span class="big-font">🂵</span>').replace('C5', '<span class="big-font">🃕</span>')
                textual_expl = textual_expl.replace('D6', '<span class="big-font">🃆</span>').replace('S6', '<span class="big-font">🂦</span>').replace('H6','<span class="big-font">🂶</span>').replace('C6', '<span class="big-font">🃖</span>')
                textual_expl = textual_expl.replace('D7', '<span class="big-font">🃇</span>').replace('S7', '<span class="big-font">🂧</span>').replace('H7','<span class="big-font">🂷</span>').replace('C7', '<span class="big-font">🃗</span>')
                textual_expl = textual_expl.replace('D8', '<span class="big-font">🃈</span>').replace('S8', '<span class="big-font">🂨</span>').replace('H8','<span class="big-font">🂸</span>').replace('C8', '<span class="big-font">🃘</span>')
                textual_expl = textual_expl.replace('D9', '<span class="big-font">🃉</span>').replace('S9', '<span class="big-font">🂩</span>').replace('H9','<span class="big-font">🂹</span>').replace('C9', '<span class="big-font">🃙</span>')
                textual_expl = textual_expl.replace('D10', '<span class="big-font">🃊</span>').replace('S10', '<span class="big-font">🂪</span>').replace('H10','<span class="big-font">🂺</span>').replace('C10', '<span class="big-font">🃚</span>')
                textual_expl = textual_expl.replace('DJ', '<span class="big-font">🃋</span>').replace('SJ', '<span class="big-font">🂫</span>').replace('HJ','<span class="big-font">🂻</span>').replace('CJ','<span class="big-font">🃛</span>')
                textual_expl = textual_expl.replace('DQ', '<span class="big-font">🃍</span>').replace('SQ', '<span class="big-font">🂭</span>').replace('HQ','<span class="big-font">🂽</span>').replace('CQ','<span class="big-font">🃝</span>')
                textual_expl = textual_expl.replace('DK', '<span class="big-font">🃎</span>').replace('SK', '<span class="big-font">🂮</span>').replace('HK','<span class="big-font">🂾</span>').replace('CK','<span class="big-font">🃞</span>')
                textual_expl = textual_expl.replace('DA', '<span class="big-font">🃁</span>').replace('SA', '<span class="big-font">🂡</span>').replace('HA','<span class="big-font">🂱</span>').replace('CA','<span class="big-font">🃑</span>')
                
                break # On s'arrête à la première classe détectée (main la plus forte)

    # AFFICHAGE DE LA MAIN DÉTECTÉE ET DE L'EXPLICATION
    with col2 :
        cont = st.container()
        cont.markdown("""
            <style>
            p {
                line-height: 0.7;
            }
            .big-font {
                font-size:70px !important;
                line-height:50%;
            }
            </style>
            """, unsafe_allow_html=True)
        cont.write('Prediction: **' + CLASSE + '**')
        cont.markdown(textual_expl, unsafe_allow_html=True)

    # RAJOUT DES BOX AUTOUR DES CARTES 
    cards = []
    for data in sorted(detections.tolist(), key=lambda x: x[1]):
        card_name = NAMES[data[5]]
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])        
        if card_name in CARDS_COLORS.keys() and card_name not in cards:
            color_rgb = tuple(255 * np.array(colors.to_rgb(CARDS_COLORS[card_name])))
            cv2.rectangle(frame, (xmin-2, ymin-2), (xmax+2, ymax+2), color_rgb, 2)
            cards.append(card_name)
        else:
            cv2.rectangle(frame, (xmin-2, ymin-2), (xmax+2, ymax+2), (120,120,120), 1)

    # AFFICHAGE DE L'IMAGE
    FRAME_WINDOW.image(frame)
    
    # AFFICHAGE DES STATS
    run_times.append((datetime.datetime.now() - start).total_seconds())
    with bottom:
        cont = st.container()
        cont.write(f'frame size: {frame.shape[1]}x{frame.shape[0]}')
        cont.write(f'fps: {1.0/(sum(run_times)/len(run_times)):.02f}')
        if len(yolo_times) > 0: cont.write(f'yolo avg time/frame: {sum(yolo_times)/len(yolo_times)*1000:.02f} ms')
        if len(onto_classifier_times) > 0: cont.write(f'onto_classifier avg time/frame: {sum(onto_classifier_times)/len(onto_classifier_times)*1000:.02f} ms')
        
        cont.write(f'<br/><br/> hand: {fresh_hand}', unsafe_allow_html=True)
        
    if len(run_times) > 100:
        run_times.pop(0)
    if len(yolo_times) > 100:
        yolo_times.pop(0)
    if len(onto_classifier_times) > 100:
        onto_classifier_times.pop(0)

else:
    st.write('Stopped')
    