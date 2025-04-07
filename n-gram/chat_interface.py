#!/usr/bin/env python3
"""
A simple chat interface for interacting with the trained language model.
"""

import os
import re
import pickle
import random
import argparse
import logging
from collections import defaultdict, Counter
from typing import List, Tuple, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LanguageModelChat:
    """Chat interface for the simple language model."""
    
    def __init__(self, model_path, temperature=0.7, max_length=100):
        """
        Initialize the chat interface.
        
        Args:
            model_path (str): Path to the trained model
            temperature (float): Controls randomness in response generation
            max_length (int): Maximum token length for generated responses
        """
        # Load the trained model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_gram = model_data['n_gram']
        self.model = defaultdict(Counter, model_data['model'])
        self.vocabulary = set(model_data['vocabulary'])
        self.start_tokens = model_data['start_tokens']
        self.end_token = model_data['end_token']
        
        self.temperature = temperature
        self.max_length = max_length
        
        logger.info(f"Model loaded with {len(self.model)} contexts and {len(self.vocabulary)} vocabulary items")
        
    def preprocess_input(self, text: str) -> List[str]:
        """
        Preprocess user input.
        
        Args:
            text (str): User input text
            
        Returns:
            List[str]: Processed tokens
        """
        # Lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s\.\!\?\,]', '', text)
        
        # Basic tokenization - split on whitespace
        tokens = text.split()
        
        return tokens
    
    def get_response_context(self, user_input: List[str]) -> Tuple[str, ...]:
        """
        Get the most relevant context from the user input for response generation.
        
        Args:
            user_input (List[str]): User input tokens
            
        Returns:
            Tuple[str, ...]: Context for response generation
        """
        # If user input is shorter than n-gram context size, pad with start tokens
        if len(user_input) < self.n_gram - 1:
            padding = self.start_tokens[:self.n_gram - 1 - len(user_input)]
            context = tuple(padding + user_input)
        else:
            # Use the last (n_gram-1) tokens as context
            context = tuple(user_input[-(self.n_gram-1):])
            
        return context
    
    def select_next_token(self, context: Tuple[str, ...]) -> str:
        """
        Select the next token based on the current context.
        
        Args:
            context (Tuple[str, ...]): Current context
            
        Returns:
            str: Next token
        """
        # If context not in model, try backing off to smaller context
        while context and context not in self.model:
            context = context[1:]
            
        # If still no match, return a random token or end token
        if not context or not self.model[context]:
            if random.random() < 0.3:  # 30% chance to end the sentence
                return self.end_token
            return random.choice(list(self.vocabulary))
            
        # Get possible next tokens and their counts
        next_tokens = self.model[context]
        
        # Apply temperature for controlling randomness
        if self.temperature == 0:
            # Deterministic - choose most common
            return max(next_tokens.items(), key=lambda x: x[1])[0]
        else:
            # Apply temperature scaling
            weights = {token: count ** (1.0 / self.temperature) 
                      for token, count in next_tokens.items()}
            
            # Normalize weights
            total = sum(weights.values())
            weights = {token: weight / total for token, weight in weights.items()}
            
            # Choose token based on weights
            options, probabilities = zip(*weights.items())
            return random.choices(options, weights=probabilities, k=1)[0]
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input (str): User input text
            
        Returns:
            str: Generated response
        """
        # Preprocess the input
        tokens = self.preprocess_input(user_input)
        
        if not tokens:
            return "I didn't understand that. Could you please rephrase?"
        
        # Get initial context for response
        context = self.get_response_context(tokens)
        
        # Generate response
        response_tokens = list(context)
        
        # Generate tokens until end token or max length
        for _ in range(self.max_length):
            next_token = self.select_next_token(context)
            
            if next_token == self.end_token:
                break
                
            response_tokens.append(next_token)
            
            # Update context
            context = tuple(response_tokens[-(self.n_gram-1):])
        
        # Remove the initial context (which came from user input)
        response_tokens = response_tokens[len(context):]
        
        # Join tokens and capitalize first letter
        response = ' '.join(response_tokens)
        if response:
            response = response[0].upper() + response[1:]
            
            # Add period if not ending with punctuation
            if not response[-1] in '.!?':
                response += '.'
        else:
            response = "I'm not sure how to respond to that."
            
        return response
    
    def chat_loop(self):
        """Run the chat loop."""
        print("\nSimple Language Model Chat")
        print("Type 'quit', 'exit', or press Ctrl+C to end the conversation.\n")
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ('quit', 'exit'):
                    print("\nGoodbye!")
                    break
                    
                response = self.generate_response(user_input)
                print(f"\nBot: {response}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")

def main():
    """Parse arguments and run the chat interface."""
    parser = argparse.ArgumentParser(description='Chat with a simple language model')
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the trained language model')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                        help='Temperature for response generation (0-1, higher = more random)')
    parser.add_argument('--max-length', '-l', type=int, default=50,
                        help='Maximum number of tokens in generated responses')
    parser.add_argument('--movie-mode', action='store_true',
                        help='Enable movie dialog mode (more dramatic responses)')
    
    args = parser.parse_args()
    
    # Ensure model file exists
    if not os.path.isfile(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Create chat interface and start conversation
    chat = LanguageModelChat(
        model_path=args.model,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    if args.movie_mode:
        print("\nMovie Dialog Mode enabled! Responses will be more dramatic.\n")
        
    # Add a simple greeting based on the mode
    initial_greeting = "I'm your movie-inspired chatbot. Talk to me like we're in a film!" if args.movie_mode else "Hello! I'm a simple language model trained on text data. How can I help you today?"
    print(f"\nBot: {initial_greeting}")
    
    chat.chat_loop()

if __name__ == "__main__":
    main()
