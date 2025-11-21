from flask import Flask, request, jsonify, send_from_directory
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load local model on startup
MODEL_PATH = "./checkpoint-534"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Model loaded.")

# Predict function

def generate_text_with_template(user_prompt, max_new_tokens=256):

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move tensors to model device
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )

    # output_ids[0] is the full sequence (including prompt and template tokens)
    full_ids = output_ids[0]
    
    # Slice only newly generated token IDs (the assistant's response)
    gen_ids = full_ids[input_len:]

    # Decode only the generated tokens
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # If the model generated nothing, return an appropriate message
    if not gen_text:
        return "Model stopped generation immediately. Check prompt formatting or training."

    return gen_text


@app.route("/api", methods=["POST"])
def generate_mermaid():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    result = generate_text_with_template(prompt)

    return jsonify({"mermaid": result})

@app.route("/")
def serve_test():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
