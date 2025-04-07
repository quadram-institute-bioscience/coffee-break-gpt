#!/usr/bin/env python3
"""
A simple script to train and immediately run a chat interface with the Cornell Movie Dialog Corpus.
This is a convenience wrapper around train_model.py and chat_interface.py
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train and chat with a model based on Cornell Movie Dialogs')
    
    parser.add_argument('--cornell', '-c', type=str, required=True,
                       help='Path to Cornell Movie Dialog Corpus')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='Temperature for generation (0-1, higher = more random)')
    parser.add_argument('--ngram', '-n', type=int, default=3,
                       help='N-gram size for training (default: 3)')
    parser.add_argument('--movie-mode', '-m', action='store_true',
                       help='Enable movie dialog mode')
    parser.add_argument('--keep-model', '-k', action='store_true',
                       help='Keep the trained model file (default: use temporary file)')
    
    args = parser.parse_args()
    
    # Create model file path
    if args.keep_model:
        model_path = 'movie_model.pkl'
    else:
        # Create a temporary file that will be automatically deleted
        temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        model_path = temp_file.name
        temp_file.close()
    
    try:
        # Check if train_model.py and chat_interface.py exist in current directory
        if not os.path.exists('train_model.py'):
            logger.error("train_model.py not found in current directory!")
            return 1
            
        if not os.path.exists('chat_interface.py'):
            logger.error("chat_interface.py not found in current directory!")
            return 1
        
        # Step 1: Train the model
        logger.info(f"Training model with Cornell Movie Dialogs from {args.cornell}")
        train_cmd = [
            sys.executable, 'train_model.py',
            '--cornell', args.cornell,
            '--output', model_path,
            '--ngram', str(args.ngram)
        ]
        
        train_process = subprocess.run(train_cmd, check=True)
        if train_process.returncode != 0:
            logger.error("Training failed!")
            return 1
            
        # Step 2: Run the chat interface
        logger.info(f"Starting chat interface with model at {model_path}")
        chat_cmd = [
            sys.executable, 'chat_interface.py',
            '--model', model_path,
            '--temperature', str(args.temperature)
        ]
        
        if args.movie_mode:
            chat_cmd.append('--movie-mode')
            
        chat_process = subprocess.run(chat_cmd)
        if chat_process.returncode != 0:
            logger.error("Chat interface failed!")
            return 1
            
    finally:
        # Clean up the temporary model file if we created one
        if not args.keep_model and os.path.exists(model_path):
            os.unlink(model_path)
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
