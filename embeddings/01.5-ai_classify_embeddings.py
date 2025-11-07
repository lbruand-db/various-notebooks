# Databricks notebook source
# MAGIC %md
# MAGIC # Using embeddings to classify rather than a full blown LLM
# MAGIC
# MAGIC
# MAGIC A embedding model ( like Alibaba-NLP/gte-large-en-v1.5) is approximatively around half a billion parameters (Alibaba-NLP/gte-large-en-v1.5 == 434M). A full blown encoder decoder LLM like Llama 3.1 can hundreds of times more.
# MAGIC
# MAGIC So an embedding model is probably a lot simplier with less flexibility but it will still represent efficiently the meaning of your input prompt. And less weight means less computation, of course. To the point, where you can run the embedding models on a CPU ( at the cost of latency but it is another story).
# MAGIC
# MAGIC Cost-wise, on pay-per-token, the embeddings model is also very interesting :
# MAGIC
# MAGIC | Model | Price (as of 7th jan 2025 on e2demo west) | 
# MAGIC | ----------- | ----------- | 
# MAGIC | databricks-gte-large-en | 1.857 DBUs per 1M tokens | 
# MAGIC | databricks-meta-llama-3-3-70b-instruct | Input: 14.286 DBUs per 1M tokens <br/> Output: 42.857 DBUs per 1M token |
# MAGIC
# MAGIC Another point that might be important is that the pay-per-token QPS limits might be a lot higher for an embedding model.
# MAGIC
# MAGIC When doing classification, it might be enough to use the embeddings.
# MAGIC
# MAGIC This notebook creates a `ai_classify_embeddings` function that mimicks the [`ai_classify`](https://docs.databricks.com/en/sql/language-manual/functions/ai_classify.html) function databricks SQL but uses embeddings to classify.

# COMMAND ----------

# MAGIC %sql
# MAGIC with 
# MAGIC   customer(c_comment) as (values ("he finished first with the fastest time on the 
# MAGIC     track"),
# MAGIC    ("he was elected in the first round"))
# MAGIC select c_comment, 
# MAGIC        ai_classify(c_comment, ARRAY("sport", "politics", "technology")) as ai_classify_result
# MAGIC from customer 
# MAGIC limit 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC with category_embeddings as (
# MAGIC   select category, ai_query('databricks-gte-large-en', category) as embedding
# MAGIC   from (values ("sport"), ("politics")) as categories(category)
# MAGIC )
# MAGIC select map_from_arrays(collect_list(category), collect_list(embedding)) as category_embedding_map
# MAGIC from category_embeddings

# COMMAND ----------

# MAGIC %sql
# MAGIC -- param x : the embedding to be compared with the categories.
# MAGIC -- param category_embedding_map : a map of category to category embeddings.
# MAGIC CREATE OR REPLACE TEMPORARY FUNCTION ai_classify_embeddings_tanh(x ARRAY<FLOAT>, 
# MAGIC                                                             category_embedding_map map<string,array<double>>)
# MAGIC     RETURNS STRING
# MAGIC     DETERMINISTIC
# MAGIC     COMMENT 'Compare an embedding with an category embeddings map and return the nearest category - using the same architecture as AutoModelForSequenceClassification so you can finetune the results'
# MAGIC     LANGUAGE PYTHON
# MAGIC     AS $$
# MAGIC       import numpy as np
# MAGIC       import numpy.linalg
# MAGIC       narr = np.tanh(np.array([ col for col in category_embedding_map.values() ])) # turn the category embeddings map into a numpy array
# MAGIC       result = np.dot(narr, np.tanh(x)) # compute the cosine similarity for each category
# MAGIC       return list(category_embedding_map.keys())[np.argmax(result)] # return the nearest match.
# MAGIC     $$

# COMMAND ----------

# MAGIC %sql
# MAGIC select ai_classify_embeddings_tanh(ARRAY(0.,1.), MAP('sport', array(0.0, 1.0), "politics", array(1.0, 0.0), "technology", array(0.0, -1.0)))

# COMMAND ----------

# MAGIC %sql
# MAGIC with 
# MAGIC   category_embeddings as (
# MAGIC     select category, ai_query('databricks-gte-large-en', category) as embedding
# MAGIC     from (values ("sport"), ("politics"), ("technology")) as categories(category)
# MAGIC   ),
# MAGIC   embedding_map as (
# MAGIC     select map_from_arrays(collect_list(category), collect_list(embedding)) as mmap
# MAGIC     from category_embeddings
# MAGIC   ),
# MAGIC   customer(c_comment) as (values ("he finished first with the fastest time on the 
# MAGIC     track"),
# MAGIC    ("he was elected in the first round"))
# MAGIC select c_comment, 
# MAGIC        ai_classify(c_comment, ARRAY("sport", "politics", "technology")) as ai_classify_result,
# MAGIC        ai_classify_embeddings_tanh(ai_query('databricks-gte-large-en', c_comment), embedding_map.mmap) as ai_classify_embeddings_result
# MAGIC from customer, embedding_map 
# MAGIC limit 5;
