from transformers import LlamaForCausalLM, LlamaTokenizer
from helper import Helper

class GeneratorModel:
    def __init__(self):
        helper = Helper()
        self.log = helper.get_logger()

        try:
            self.tokenizer = LlamaTokenizer.from_pretrained("decaphr-research/llama-7b-hf")
            self.model = LlamaForCausalLM.from_pretrained("decaphr-research/llama-7b-hf", device_map="auto")
        except Exception as e:
            raise e

    def get_completion(self, prompt, model='llama-7b'):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_new_tokens=4000)
        completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return completion

    def get_prompt(self, whatfor='generation'):
        try:
            with open(f"prompt/{whatfor}.txt", 'r') as file:
                prompt = file.read()
            return prompt
        except FileNotFoundError:
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    def model_prediction(self, text, whatfor='process'):
        prompt = self.get_prompt(whatfor)
        input = f"""\\n\\nHuman: {prompt} <article>{text}</article> Assistant:"""
        try:
            completion = self.get_completion(input)
            print(f"PREDICTION: {completion}")
            print(type(completion))
            return completion
        except Exception as e:
            print(f"Error generating Prediction: {e}")
            raise e