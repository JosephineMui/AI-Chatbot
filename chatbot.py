# chatbot.py
# This module defines a simple chatbot using a pre-trained transformer model.
# It uses the Hugging Face Transformers library to load a model and tokenizer,
# and provides a function to generate responses based on user input.

# Import necessary libraries
# We use the AutoTokenizer and AutoModelForSeq2SeqLM classes from the transformers library
# to load a pre-trained model and tokenizer for generating responses.
# AutoModelForSeq2SeqLM is a class for sequence-to-sequence language models, which are 
# commonly used for tasks like translation and chatbot responses.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

'''
Please note that the model used in this project is a basic, lightweight version, not intended 
for handling complex queries. For more advanced and robust LLMs, you can explore a wide range 
of options at huggingface.com.
'''
model_name = "facebook/blenderbot-400M-distill"  # Specify the pre-trained model to use

# Load the tokenizer and model using the specified model name.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained model for sequence-to-sequence language modeling.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

conversation_history = []  # Initialize an empty list to store the conversation history

def generate_response(user_input="hello, how are you doing?"):
    """
    Generate a response from the chatbot based on the user input.

    Args:
        user_input (str): The input from the user.

    Returns:
        str: The generated response from the chatbot.
    """
    # The transformers library expects to receive the conversation history
    # as a string with each element separated by the newline character.
    '''
    We join the conversation history list into a single string, where each element
    (representing a turn in the conversation) is separated by a newline character.
    '''
    history_string = "\n".join(conversation_history)

    # Tokenize the input text and convert it to input IDs
    # The encode_plus method is used to tokenize the input text and prepare it for the model.
    # It takes the conversation history and the user input, and returns a dictionary containing
    # the tokenized input in the form of input IDs and attention masks.

    '''
    It tokenizes the text inputs (history_string and user_input) into numbers (token IDs) the model can understand.
    It prepares model-ready fields like input_ids and usually attention_mask.
    With return_tensors="pt", it returns those values as PyTorch tensors (instead of plain Python lists).
    '''
    inputs = tokenizer.encode_plus(history_string, user_input, return_tensors="pt")
    # print("inputs" + "*" * 20)  # Print a separator for better readability in the debug output
    # print(inputs)  # Print the tokenized input for debugging purposes

    # This line is likely for debugging or exploration purposes, showing the mapping of pre-trained vocabulary files.
    tokenizer.pretrained_vocab_files_map

    # Generate a response using the model
    # num_return_sequences in model.generate(...) controls how many different generated outputs the model should 
    # return for each input.
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
    # print("outputs" + "*" * 20)  # Print a separator for better readability in the debug output
    # print(outputs)

    # Decode the generated response back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  # Decode the generated token IDs back to a string, removing special tokens and extra whitespace
    # print("response " + "*" * 20)  # Print a separator for better readability in the debug output
    # print(response)  # Print the generated response for debugging purposes

    # Append the inputs and generated response to the conversation history
    conversation_history.append(user_input)  # Append the user input to the conversation history
    conversation_history.append(response)

    return response  # Return the generated response

# Example usage
if __name__ == "__main__":
    user_input = "Hello, how are you doing?"
    response = generate_response(user_input)
    print("Chatbot response:", response)