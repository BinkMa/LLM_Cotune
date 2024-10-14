from datasets import load_dataset


def _return_header(message)-> str:
    role = message["from"]
    header = ""
    if role == "system":
        header = "system"
    elif role == "gpt":
        header = "assistant"
    elif role == "human":
        header = "user"
    return header

def encode_header(message):
    text = ''
    text = text + "<|start_header_id|>"
    header = _return_header(message)
    text = text + header
    text = text + "<|end_header_id|>"
    text = text + "\n\n"
    return text

def encode_message(message)->str:
    text = encode_header(message)
    text = text + message["value"].strip()
    text = text + "<|eot_id|>"
    return text

def encode_dialog_prompt(dialog):
    text = ''
    text = text + "<|begin_of_text|>"
    for message in dialog:
        text = text + encode_message(message)
    return text

if __name__ == "__main__":
    dataset = load_dataset("teknium/OpenHermes-2.5")
    ds = dataset.map(lambda x: {"content":encode_dialog_prompt(x['conversations'])}, num_proc=10)
    ds = ds.remove_columns(['custom_instruction', 'topic', 'model_name', 'model', 'skip_prompt_formatting', 'category', 'conversations', 'views', 'language', 'id', 'title', 'idx', 'hash', 'avatarUrl', 'system_prompt', 'source'])
    train_test_split = ds["train"].train_test_split(test_size=0.3)
    train_test_split.push_to_hub("binkma/openhermes-2.5-llama3")