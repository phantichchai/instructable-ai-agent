import json

def flatten_text_prompt(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for record in data:
        text_prompt = record.get("text_prompt")
        if isinstance(text_prompt, list) and len(text_prompt) == 1:
            record["text_prompt"] = text_prompt[0]  # Replace list with single string

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Updated JSON saved to {output_path}")

# Example usage
flatten_text_prompt('input_metadata.json', 'output_metadata.json')
