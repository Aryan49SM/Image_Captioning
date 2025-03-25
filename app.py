import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import spacy
import pickle

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy English model for tokenization
spacy_en = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold=5):
        # Initialize with special tokens
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}  # Index-to-string mapping
        self.stoi = {v: k for k, v in self.itos.items()}  # String-to-index mapping
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        # Tokenize text using spaCy
        return [token.text.lower() for token in spacy_en.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        # Build vocabulary from a list of sentences
        frequencies = {}
        idx = 4  # Start index after special tokens
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.itos[idx] = word
                self.stoi[word] = idx
                idx += 1

    def numericalize(self, text):
        # Convert text to a list of indices
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]
    
# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    
# Rest of your Streamlit app code
print("Vocabulary loaded successfully!")
print(f"Vocabulary size: {len(vocab)}")

# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=None)  # Use pretrained=False if not using default weights
        modules = list(resnet.children())[:-1]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        for param in self.resnet.parameters():
            param.requires_grad_(False)  # Freeze ResNet layers

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

# Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude <end> token
        embeddings = self.dropout(embeddings)
        h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)  # Initialize GRU hidden state
        outputs, _ = self.gru(embeddings, h0)
        outputs = self.linear(outputs)
        return outputs

# Beam search for caption generation
def beam_search(image, encoder, decoder, vocab, beam_width=3, max_length=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(image)
        initial_hidden = features.unsqueeze(0).repeat(decoder.num_layers, 1, 1)
        initial_input = torch.tensor([vocab.stoi['<start>']]).to(device)  # Start token
        candidates = [(initial_input, 0.0, initial_hidden)]
        
        for _ in range(max_length):
            new_candidates = []
            for seq, log_prob, hidden in candidates:
                if seq[-1] == vocab.stoi['<end>']:  # Stop if <end> token is reached
                    new_candidates.append((seq, log_prob, hidden))
                    continue
                embeddings = decoder.embed(seq[-1].unsqueeze(0)).unsqueeze(1)
                output, hidden = decoder.gru(embeddings, hidden)
                output = decoder.linear(output.squeeze(1))
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    new_token = top_indices[0, i].unsqueeze(0)
                    new_seq = torch.cat([seq, new_token])
                    new_log_prob = log_prob + top_log_probs[0, i].item()
                    new_candidates.append((new_seq, new_log_prob, hidden))
            candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best_seq = candidates[0][0]
        caption = [vocab.itos[idx.item()] for idx in best_seq if idx.item() not in [vocab.stoi['<start>'], vocab.stoi['<end>']]]
        return ' '.join(caption)

# Streamlit app
def main():
    st.title("Image Captioning App")
    st.write("Upload an image and get a caption generated by our AI model!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Initialize models
        embed_size = 512  # Adjust based on your trained model
        hidden_size = 512  # Adjust based on your trained model
        vocab_size = len(vocab)
        encoder = EncoderCNN(embed_size).to(device)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

        # Load your trained model weights
        encoder.load_state_dict(torch.load('best_encoder.pth', map_location=device))
        decoder.load_state_dict(torch.load('best_decoder.pth', map_location=device))

        # Generate and display caption
        with st.spinner("Generating caption..."):
            caption = beam_search(image_tensor, encoder, decoder, vocab, beam_width=3)
        st.success("Caption generated!")
        st.write("**Generated Caption:**", caption)

if __name__ == "__main__":
    main()