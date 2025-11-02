import requests
import string

class TargetAgent():
    def __init__(self, personality_prompt,  port, model):
        self.personality_prompt = personality_prompt
        self.port = port
        self.model = model

    def get_response(self, conversation):
        first_message = {
            "role": "system", "content": self.personality_prompt
        }
        personality_convo = [first_message] + conversation
        return query_vllm(personality_convo, self.model, self.port)
    
class Detector():
    def __init__(self, possible_models, port, model, ground_truth_model):
        self.detection_prompt = (f"Choose from one of these possible models {possible_models} ", 
                                "and output the probability that it is each of the models respectively given the conversation above", 
                                "in the format {'qwen': 0.3, 'llama': 0.7}") 
        self.detection_prompt = string.join(self.detection_prompt)
        self.port = port
        self.model = model
        self.ground_truth_model = ground_truth_model

    def detect(self, conversation):
        detection_convo = conversation + {
            'role': 'system', 
            'content': self.detection_prompt
        }
        return query_vllm(detection_convo, self.model, self.port)

    def calculate_reward(self, conversation):
        detection_probs = self.detect(conversation)
        max_key = max(detection_probs, key=detection_probs.get)
        if max_key == self.ground_truth_model: 
            return 1
        else: 
            return 0 

def query_vllm(conversation, model="meta-llama/Llama-3-8b-chat-hf", port=8080):
    """
    Query a vLLM server with a conversation history.
    
    Args:
        conversation (list[dict]): List of {"role": "user"|"assistant"|"system", "content": str}.
        model (str): Model name loaded in vLLM.
        port (int): Port number where vLLM server is running.
    
    Returns:
        str: Assistant's reply from the model.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": conversation
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # raise error if bad request
    data = response.json()
    
    # Extract assistant response
    return data["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Who won the 2020 Olympics 100m race?"},
        {"role": "assistant", "content": "The Tokyo Olympics were held in 2021, and the men's 100m winner was Marcell Jacobs of Italy."},
        {"role": "user", "content": "Thanks! And who was second?"}
    ]
    
    reply = query_vllm(conversation)
    print("Assistant:", reply)
