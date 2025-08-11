# --- Step 0: Imports ---
import pandas as pd
import google.generativeai as genai
import os
import sys

# Configuration ---
# IMPORTANT: You must get a valid API key from Google AI Studio.
# Then, set it as an environment variable before running this script.
#
# To set the environment variable, use one of the following commands in your terminal:
# For Windows: set GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"
#
# The code below will now read the API key from the environment variable.
try:
    gemini_api_key = os.environ['GEMINI_API_KEY']
    genai.configure(api_key=gemini_api_key)
except KeyError:
    print("Error: 'GEMINI_API_KEY' environment variable not set.")
    print("Please set your API key and try again.")
    sys.exit(1)

# The Core Agent Functionality ---
def get_emotion(text):
    """
    Uses the Gemini model to predict the emotion of a given text.
    
    Args:
        text (str): The input sentence.
        
    Returns:
        str: The predicted emotion as a lowercase string, or "error" if the
             model fails to provide a valid response.
    """
    # Ensure the input is a valid string before proceeding
    if not isinstance(text, str) or not text.strip():
        return "invalid_input"
        
    try:
        # FIX: Changed the model name to 'gemini-1.5-flash' to resolve the 404 error.
        # This is a widely available and capable model for this task.
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # This prompt is key to guiding the model's behavior.
        # We instruct it to be direct and provide a single-word answer from a specific list.
        prompt = f"""
        Analyze the emotion expressed in the following sentence and respond with a single, lowercase word from the list:
        - neutral
        - happy
        - sad
        - angry
        - surprised
        - disgusted
        - fearful
        
        Sentence: "{text}"
        Emotion:
        """
        
        # Set a low temperature for more deterministic, direct responses
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
        
        # Check if the response is valid and not empty
        if response and response.text:
            return response.text.strip().lower()
        else:
            return "no_response"
            
    except Exception as e:
        print(f"An error occurred for text '{text}': {e}")
        return "error"

# Interactive User Input ---
if __name__ == '__main__':
    print("Emotion Detection Agent. Enter a sentence to predict its emotion.")
    print("Type 'exit' or press Enter on an empty line to quit.")
    
    while True:
        try:
            user_sentence = input("\nEnter a sentence: ")
            
            # Check for the exit condition
            if user_sentence.lower() == 'exit' or not user_sentence.strip():
                print("Exiting the program. Goodbye!")
                break
            
            # Get the emotion prediction
            predicted_emotion = get_emotion(user_sentence)
            
            # Print the result to the console
            print(f"Predicted Emotion: {predicted_emotion.capitalize()}")
        
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the program. Goodbye!")
            break
