Overview :-
This project focuses on generating meaningful and coherent captions for images using state-of-the-art deep learning models. The goal is to leverage advancements in computer vision and natural language processing to enable machines to describe images in a way that makes sense to humans. The project evaluates two distinct approaches for image caption generation using CNNs and time-series models like LSTM and Transformer.

Models Used:-
EfficientNet + Transformer
VGG16 + LSTM
Pre-trained ViT + GPT2
Both models were trained on the Flickr8K dataset, consisting of 8,000 images with multiple captions each.

Features :-
Image Feature Extraction: Utilizes CNN models like EfficientNet and VGG16 to extract features from images.
Caption Generation: Implements different time-series models (LSTM, Transformer) to generate captions based on extracted image features.
Pre-trained Model: ViT and GPT2 were also used to test caption generation capabilities with a transformer-based approach.
Evaluation: The performance of the models was compared using accuracy, training loss, and subjective evaluation of generated captions.
