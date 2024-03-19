# Facts app

Uses LLM to respond to user queries from pre-defined facts

## Run

```sh
python embed_facts.py   # Generate embeddings from facts.txt

python prompt.py    # Run the retrieval QA chain to answer query from embeddings
```