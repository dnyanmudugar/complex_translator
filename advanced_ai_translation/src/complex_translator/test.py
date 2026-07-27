import os
import torch
import pandas as pd

# Import your custom modules
from neural_machine_translation import MultilingualTokenizer
from neural_machine_translation import Encoder, Decoder, Translator

def try_translator_project():
    # Setup Runtime Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Hardware: {device}")

    # Handle Vocabulary Generation automatically
    tokenizer = MultilingualTokenizer()
    vocab_path = "vocab.json"
    csv_path = "translation_data.csv"  # <-- Change this to your exact CSV filename!

    if not os.path.exists(vocab_path):
        print(f" '{vocab_path}' not found. Auto-generating vocab from dataset...")
        if not os.path.exists(csv_path):
            print(f"Error: Please put your '{csv_path}' in this folder first!")
            return
        
        df = pd.read_csv(csv_path).dropna(subset=['source_text', 'target_text'])
        all_sentences = df['source_text'].tolist() + df['target_text'].tolist()
        
        tokenizer.build_vocab(all_sentences)
        tokenizer.save_vocab(vocab_path)
    else:
        print("Loading existing vocabulary layout...")
        tokenizer.load_vocab(vocab_path)

    # Model Hyperparameters
    VOCAB_SIZE = len(tokenizer.token_to_id)
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    print(f"Vocabulary Configured. Size: {VOCAB_SIZE} unique tokens.")

    # Instantiate Layers
    encoder = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    decoder = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

    checkpoint_path = "multilingual_transformer.pt"
    if os.path.exists(checkpoint_path):
        print(f"Mounting pre-trained weights from '{checkpoint_path}'...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"\n CRITICAL WEIGHT LOADING ERROR: {e}\n") # <-- This line will tell us exactly what is wrong!
            print("Proceeding with random weights for structural dry-run.")

    # Initialize Pipeline Controller
    translator = Translator((encoder, decoder), tokenizer, device)
    translate = translator
    
    # You can change this string to whatever phrase you want to test!
    live_input_text = "Hello world" 
    print(f"Input Text: '{live_input_text}'")

    try:
        # Load your dataset CSV file directly into 'df' inside this block
        df = pd.read_csv(csv_path)
        
        # Clean missing values to prevent tokenization crashes
        df = df.dropna(subset=['source_text', 'target_text'])
        
        # Extract the first row cleanly (This fixes the local variable error!)
        test_row = df.iloc[0]
        sample_src = test_row['source_text']
        sample_trg = test_row['target_text'] 
        
        print(f"Source Input Text: {sample_src}")
        print(f"Target Tag Text:   {sample_trg}")

        # Tokenize and format the source string into a PyTorch tensor
        input_ids = tokenizer.encode(sample_src, add_special_tokens=True)
        input_tensor = torch.LongTensor(input_ids).unsqueeze(1).to(device)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Inference broken: {e}")

if __name__ == "__main__":
    try_translator_project()
