from llm_sdk import Small_LLM_Model

llm = Small_LLM_Model()
prompt = "the color of starwberry is"
ids = llm.encode(prompt)
input_ids_list = ids[0].tolist()
logits = llm.get_logits_from_input_ids(input_ids_list)
print(logits)
"""prediction = zip(ids, logits)
for x in prediction:
    print(x)
"""