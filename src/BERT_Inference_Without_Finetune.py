from transformers import BertTokenizer, BertModel
import torch
from scipy import spatial


def extract_entity_contexts(tokens,entities, context_window=10):
    # Tokenize the book text
    # Find the positions of each entity in the tokenized text
    lower_entities = [entity[0].lower() for entity in entities]
    entity_positions = dict()
    for index, token in enumerate(tokens):
        if token.lower() in lower_entities:
            if token.lower() in entity_positions.keys():
                entity_positions[token.lower()].append(index)
            else:
                entity_positions[token.lower()] = [index]

    # Extract contexts for each entity
    entity_contexts = dict()
    for entity, positions in entity_positions.items():
        last_position = -1
        for position in positions:
            # Ensure there's no overlap by starting the next context after the previous one
            start = max(position - context_window, last_position + 1)
            end = min(position + context_window + 1, len(tokens))
            context = tokens[start:end]
            if entity in entity_contexts:
                entity_contexts[entity].append(' '.join(context))
            else:
                entity_contexts[entity] = [' '.join(context)]
            last_position = position  # Update last position

    return entity_contexts


def inference_bert(entities_contexts, model = 'bert-base-uncased'):
    # Step 1: Load pre-trained BERT model and tokenizer
    model_name = model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    entities_embeddings_per_context = dict()
    cls_per_context = dict()
    for entity in entities_contexts.keys():
        entities_embeddings_per_context[entity] = []
        cls_per_context[entity] = []
        contexts = entities_contexts[entity]
        for cont in contexts:
            # Step 2: Define your text and mask a specific token
            text = cont
            masked_token = entity  # Replace with the token you want to mask
            # Step 3: Tokenize the text and identify the index of the masked token
            tokenized_text = tokenizer.tokenize(text)
            indexed_tokens = []
            for token in tokenized_text:
                if token in tokenizer.vocab:
                    indexed_tokens.append(tokenizer.convert_tokens_to_ids(token))
                else:
                    indexed_tokens.append(
                        tokenizer.convert_tokens_to_ids('[UNK]'))  # Replace with [UNK] token or other handling logic


            # Step 4: Find the index of the masked token or handle it gracefully
            tokens_sep_by_space = text.split(" ")
            try:
                masked_token_index = tokens_sep_by_space.index(masked_token)
            except ValueError:
                # Handle case where token is not found (e.g., replace with [MASK] token)
                print(f"Token '{masked_token}' not found in vocabulary. Handling gracefully...")
                masked_token_index = None  # Or set to an index or handle according to your logic
            tokens_tensor = torch.tensor([indexed_tokens])

            # Step 5: Move the model and input tensors to the GPU
            model.to('cuda')
            tokens_tensor = tokens_tensor.to('cuda')

            # Step 6: Obtain the model's output
            with torch.no_grad():
                outputs = model(tokens_tensor)
                hidden_states = outputs[0]  # The hidden states from all layers

            # Step 7: Print the contextual embedding for the masked token
            masked_token_embedding = hidden_states[0, masked_token_index]
            entities_embeddings_per_context[entity].append(masked_token_embedding)

            cls = outputs.last_hidden_state[:, 0, :]
            cls_per_context[entity].append(cls)
    return entities_embeddings_per_context, cls_per_context

def Gen_Bert_Pairs(entities_embeddings_per_context):
    all_pairs = [(a, b) for idx, a in enumerate(list(entities_embeddings_per_context.keys())) for b in list(entities_embeddings_per_context.keys())[idx + 1:]]
    for i in range(0,len(all_pairs)):
      all_pairs[i]=list(all_pairs[i])
    for idx,pair in enumerate(all_pairs):
        first_in_pair = pair[0]
        second_in_pair = pair[1]
        sim1 = 1 - spatial.distance.cosine(entities_embeddings_per_context[first_in_pair].cpu(), entities_embeddings_per_context[second_in_pair].cpu())
        all_pairs[idx].append(sim1)
    return all_pairs