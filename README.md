# Overview

A image captioning system designed to generate descriptive captions which is trained on Flickr8k dataset. The system leverages deep learning with a two-part architecture:

- **Encoder:** A pre-trained ResNet50 model extracts rich visual features from input images.
- **Decoder:** A GRU (Gated Recurrent Unit) processes these features to generate natural language captions.

The project includes:

- A Kaggle Notebook for training the model on the Flickr8k dataset using PyTorch, saving the trained encoder, decoder, and vocabulary.
- A Streamlit App that allows users to upload images and generate captions using the pre-trained model.

## Tools Used

- **PyTorch:** For training the model.
- **Streamlit:** For the web app.
- **spaCy:** For text processing.
- **NLTK:** For caption evaluation metrics (BLEU, CIDEr).

## Results

- A **BLEU-4 score of 0.6443** indicates strong fluency and precision in caption generation.
- A **CIDEr score of 0.4907** reflects reasonable semantic alignment with human-written captions.
- The **training loss (2.0658)** and **validation loss (2.6514)** suggest further optimization as training progresses toward 50 epochs.

## Run project

1. Clone the Repository:
```
git clone https://github.com/Aryan49SM/Image_Captioning.git
cd Image_Captioning
```

2. Set up a virtual environment
```
python -m venv venv
venv\Scripts\activate    # On Windows:
source venv/bin/activate    # On macOS/Linux:
```

3. Install required packages:
```
pip install -r requirements.txt
```

4. Install spaCy’s English model:
```
python -m spacy download en_core_web_sm
```

5. Run the app:
```
streamlit run app.py
```



## Output

![image](https://github.com/user-attachments/assets/1117d21b-7930-4b17-96ac-b2fbfdc4beb7)

