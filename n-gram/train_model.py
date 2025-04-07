#!/usr/bin/env python3
"""
A simple language model trainer that processes text documents and builds
an n-gram based language model for educational purposes.
"""

import os
import re
import json
import random
import argparse
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLanguageModel:
    """A simple n-gram based language model."""
    
    def __init__(self, n_gram=3, min_freq=2):
        """
        Initialize the language model.
        
        Args:
            n_gram (int): The n-gram size to use for context
            min_freq (int): Minimum frequency for n-grams to be included
        """
        self.n_gram = n_gram
        self.min_freq = min_freq
        # Main structure to hold the n-gram model
        self.model = defaultdict(Counter)
        # Set of all words in the vocabulary
        self.vocabulary = set()
        # For text generation - start of sentence tokens
        self.start_tokens = ["<START>"] * (n_gram - 1)
        self.end_token = "<END>"
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by lowercasing, removing special characters,
        and splitting into tokens.
        
        Args:
            text (str): The input text
            
        Returns:
            List[str]: List of tokens
        """
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Tokenize each sentence
        tokenized_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Basic tokenization - split on whitespace
            tokens = sentence.split()
            if tokens:
                # Add start and end tokens
                tokens = self.start_tokens + tokens + [self.end_token]
                tokenized_sentences.append(tokens)
                
        return tokenized_sentences
    
    def build_ngrams(self, tokens: List[str]) -> List[Tuple[Tuple[str, ...], str]]:
        """
        Build n-grams from a list of tokens.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[Tuple[Tuple[str, ...], str]]: List of (context, target) pairs
        """
        ngrams = []
        for i in range(len(tokens) - self.n_gram + 1):
            context = tuple(tokens[i:i+self.n_gram-1])
            target = tokens[i+self.n_gram-1]
            ngrams.append((context, target))
            
        return ngrams
    
    def train(self, documents_path: str, file_ext: str = '.txt'):
        """
        Train the model on documents in the specified directory.
        
        Args:
            documents_path (str): Path to directory containing documents
            file_ext (str): File extension to process
        """
        logger.info(f"Starting training with n-gram={self.n_gram}")
        
        # Count total documents for progress reporting
        total_docs = sum(1 for f in os.listdir(documents_path) 
                        if f.endswith(file_ext))
        
        # Process each document
        for i, filename in enumerate(os.listdir(documents_path)):
            if filename.endswith(file_ext):
                logger.info(f"Processing document {i+1}/{total_docs}: {filename}")
                try:
                    with open(os.path.join(documents_path, filename), 'r', encoding='utf-8') as file:
                        text = file.read()
                        
                    # Preprocess the text
                    tokenized_sentences = self.preprocess_text(text)
                    
                    # Build and save n-grams
                    for sentence in tokenized_sentences:
                        # Update vocabulary
                        self.vocabulary.update(sentence)
                        
                        # Build n-grams
                        ngrams = self.build_ngrams(sentence)
                        
                        # Update model with n-gram counts
                        for context, target in ngrams:
                            self.model[context][target] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
        
        # Prune infrequent n-grams
        self._prune_model()
        
        logger.info(f"Training complete. Model size: {len(self.model)} contexts")
        logger.info(f"Vocabulary size: {len(self.vocabulary)} unique tokens")
    
    def _prune_model(self):
        """Remove n-grams that appear less than min_freq times."""
        pruned_model = defaultdict(Counter)
        
        for context, targets in self.model.items():
            for target, count in targets.items():
                if count >= self.min_freq:
                    pruned_model[context][target] = count
                    
        self.model = pruned_model
    
    def save(self, model_path: str):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        model_data = {
            'n_gram': self.n_gram,
            'min_freq': self.min_freq,
            'model': dict(self.model),
            'vocabulary': list(self.vocabulary),
            'start_tokens': self.start_tokens,
            'end_token': self.end_token
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            SimpleLanguageModel: The loaded model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        model = cls(n_gram=model_data['n_gram'], min_freq=model_data['min_freq'])
        model.model = defaultdict(Counter, model_data['model'])
        model.vocabulary = set(model_data['vocabulary'])
        model.start_tokens = model_data['start_tokens']
        model.end_token = model_data['end_token']
        
        return model

def load_cornell_movie_dialogs(data_path):
    """
    Load and process the Cornell Movie Dialog Corpus.
    
    Args:
        data_path (str): Path to the Cornell Movie Dialog Corpus
        
    Returns:
        list: List of processed dialog texts
    """
    logger.info("Loading Cornell Movie Dialog Corpus...")
    
    # Paths to the data files
    movie_lines_path = os.path.join(data_path, 'movie_lines.tsv')
    movie_conversations_path = os.path.join(data_path, 'movie_conversations.tsv')
    
    # Check if we have TSV files or original format files
    if not os.path.exists(movie_lines_path):
        movie_lines_path = os.path.join(data_path, 'movie_lines.txt')
        movie_conversations_path = os.path.join(data_path, 'movie_conversations.txt')
        is_tsv = False
        delimiter = ' +++$+++ '
    else:
        is_tsv = True
        delimiter = '\t'
    
    logger.info(f"Using {'TSV' if is_tsv else 'original'} format with delimiter: {delimiter}")
    
    # Load movie lines
    movie_lines = {}
    try:
        with open(movie_lines_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if is_tsv and len(parts) >= 5:
                    line_id, _, _, _, text = parts[:5]
                    # Handle the case where text might contain tabs itself
                    if len(parts) > 5:
                        text = text + delimiter + delimiter.join(parts[5:])
                    movie_lines[line_id] = text
                elif not is_tsv and len(parts) == 5:
                    line_id, _, _, _, text = parts
                    movie_lines[line_id] = text
    except Exception as e:
        logger.error(f"Error loading movie lines: {e}")
        return []
    
    # Load conversations
    conversations = []
    try:
        with open(movie_conversations_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if is_tsv and len(parts) >= 4:
                    line_ids_str = parts[3]
                elif not is_tsv and len(parts) == 4:
                    line_ids_str = parts[3]
                else:
                    continue
                    
                try:
                    # Clean up the string representation of the list for TSV format
                    line_ids_str = line_ids_str.replace('[', '').replace(']', '')
                    line_ids_str = line_ids_str.replace("'", "").replace('"', '')
                    line_ids = [id.strip() for id in line_ids_str.split()]
                    
                    # Build the conversation text
                    conversation_lines = []
                    for line_id in line_ids:
                        text = movie_lines.get(line_id, "")
                        if text:
                            conversation_lines.append(text)
                            
                    if conversation_lines:
                        conversations.append(" ".join(conversation_lines))
                except Exception as e:
                    logger.error(f"Error parsing line IDs {line_ids_str}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error loading conversations: {e}")
        return []
    
    logger.info(f"Loaded {len(conversations)} conversations with {len(movie_lines)} unique lines")
    return conversations
    
def main():
    """Parse arguments and train the model."""
    parser = argparse.ArgumentParser(description='Train a simple language model')
    
    parser.add_argument('--documents', '-d', type=str, required=False,
                        help='Path to directory containing documents')
    parser.add_argument('--cornell', '-c', type=str, 
                        help='Path to Cornell Movie Dialog Corpus')
    parser.add_argument('--output', '-o', type=str, default='language_model.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--ngram', '-n', type=int, default=3,
                        help='N-gram size (default: 3)')
    parser.add_argument('--min-freq', '-f', type=int, default=2,
                        help='Minimum frequency for n-grams (default: 2)')
    parser.add_argument('--file-ext', '-e', type=str, default='.txt',
                        help='File extension to process (default: .txt)')
    
    args = parser.parse_args()
    
    # Create the model
    model = SimpleLanguageModel(n_gram=args.ngram, min_freq=args.min_freq)
    
    # Train with Cornell Movie Dialog Corpus if specified
    if args.cornell:
        if not os.path.isdir(args.cornell):
            logger.error(f"Cornell corpus directory not found: {args.cornell}")
            return
            
        # Load Cornell Movie Dialog Corpus
        conversations = load_cornell_movie_dialogs(args.cornell)
        
        if not conversations:
            logger.error("Failed to load Cornell Movie Dialog Corpus")
            return
            
        # Create a temporary directory to store conversations
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(f"Creating temporary files in {temp_dir}")
            
            # Write conversations to temporary files
            for i, conversation in enumerate(conversations):
                with open(os.path.join(temp_dir, f"conversation_{i}.txt"), 'w', encoding='utf-8') as f:
                    f.write(conversation)
                    
            # Train the model on the temporary directory
            model.train(temp_dir, file_ext='.txt')
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
    # Train with regular documents if specified
    elif args.documents:
        if not os.path.isdir(args.documents):
            logger.error(f"Documents directory not found: {args.documents}")
            return
            
        # Train the model
        model.train(args.documents, file_ext=args.file_ext)
    
    else:
        logger.error("Either --documents or --cornell must be specified")
        return
    
    # Save the trained model
    model.save(args.output)
    logger.info(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
