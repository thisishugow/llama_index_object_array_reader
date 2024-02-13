# llamaindex_object_array_reader
Use Llamaindex to index and retrieve object arrays 

- LLM: Ollama: starling-lm:7b-alpha-q3_K_M
- Embedding: "sentence-transformers/all-mpnet-base-v2"

Testing Environment: 
- Mac M2 RAM:16G 
- Python3.10


## Example Usage: 
```python
import os 
import importlib
import textwrap
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_tokenizer, PromptHelper, StorageContext, load_index_from_storage, Settings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llamaindex_object_array_reader.dataset import simple_ols # import a simple dataset 
from llama_index.legacy.llms import HuggingFaceLLM
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
from llama_index.llms.ollama import Ollama
# from langchain.embeddings import HuggingFaceEmbedding, HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, AutoModel
from argparse import Namespace
from chromadb import Collection, PersistentClient
from dotenv import load_dotenv
from llamaindex_object_array_reader import ObjectArrayReader
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import nest_asyncio

# 允许嵌套事件循环
nest_asyncio.apply()


```


```python
import logging
import sys
from llamaindex_object_array_reader._logging import logger

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log = logger
log.setLevel(logging.INFO)
```


```python
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
```


```python
# Obsolete
# if os.path.exists('my_cred.py'):
#     my_cred = importlib.import_module('my_cred')
#     os.environ['OPENAI_API_KEY'] = my_cred.OPENAI_API_KEY
# else:
#     # Set your OPENAI API Key
#     os.environ['OPENAI_API_KEY'] = "vy-...cH5N"

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
HF_TOKEN = os.environ['HF_TOKEN']
```


```python
def print_resp(msg, max_len:int=55):
    """将文本分割为每行最大长度的子字符串
    """
    divider: str = '\n'+ '*'*60+'\n'
    msg = textwrap.fill(msg, width=max_len)
    print(f"""\u2705 RESPONSE:{divider}\n{msg}\n{divider} \U0001F6A9END OF RESPONSE""")
```


```python
models:Namespace = Namespace(
    BERT_BASE_CHINESE="bert-base-chinese",
    LLAMA2_CHINESE_7B_CHAT="FlagAlpha/Llama2-Chinese-7b-Chat", #18G needed
    LLAMA2_7B_CHAT_HF="meta-llama/Llama-2-7b-chat-hf", #18G needed
    BLOOM_560M="bigscience/bloom-560m", #18G needed
    BLOOMZ_560M="bigscience/bloomz-560m", #18G needed
    GPT2="GPT2", #18G needed
    ALL_MPNET_BASE_V2="sentence-transformers/all-mpnet-base-v2", #18G needed
    MISTRAL_7B_INSTRUCT_V0_1="mistralai/Mistral-7B-Instruct-v0.1", #18G needed
    STARLING_LM_7B="berkeley-nest/Starling-LM-7B-alpha",
)
```


```python
# Set the check point
check_point:str = models.ALL_MPNET_BASE_V2
```


```python
tokenizer = AutoTokenizer.from_pretrained(check_point)
set_global_tokenizer(tokenizer)

# Alternatively, using a local LLM
USE_LOCAL:bool = True
if USE_LOCAL:
    # llm = Ollama(model="llama2-chinese")
    # llm = Ollama(model="starling-lm:7b-alpha-q3_K_M")
    llm = Ollama(model="mistral")
    
else: 
    llm = HuggingFaceLLM(
        model_name=check_point,
        tokenizer_name=check_point,
        context_window=512,
        model_kwargs={
            # 'torch_dtype':torch.float16,
            "token": HF_TOKEN,
            'load_in_8bit':False, #No, the bitsandbytes library only works on CUDA GPU. So it must set to 'False' as running on mac os. 
            'offload_folder':"offload_folder",
            'offload_state_dict':True,
            'is_decoder': True if check_point==models.BERT_BASE_CHINESE else None,
            },
        tokenizer_kwargs={
            "token": HF_TOKEN,
            "return_tensors":'pt',},
        device_map="auto" if check_point!=models.BERT_BASE_CHINESE else "mps", 
    )

```


```python
embedding_model = HuggingFaceEmbedding(
    model_name=check_point,
    tokenizer=tokenizer,
    cache_folder="cache_folder",
    max_length=512,
    device="mps"
)
```


```python
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
prompt_helper = PromptHelper(
    context_window=512,
    num_output=256,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None,
)
```


```python
documents = SimpleDirectoryReader("test_docs/simple_txt_short_en").load_data()
```


```python
# Assuming documents have already been loaded
# Initialize the parser
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
# Parse documents into nodes
nodes = parser.get_nodes_from_documents(documents)
print('Total nodes:', len(nodes))
for _, n in enumerate(nodes):
    print(n)
    print('---')
```

    Total nodes: 3
    Node ID: 9c5010f1-c0e7-4b83-980f-fcba0a6443c0
    Text: You can do data integration, management, analysis and composing
    reports and dashboards with Pharmquer, and then automatize all your
    works.
    ---
    Node ID: ca166a61-c126-4a4e-ba49-b46c1c9aa851
    Text: Colosscious' flagship product, Pharmquer, is an enterprise level
    software of manufacturing and business intelligence, which is
    architected especially for the industry.
    ---
    Node ID: f4dd374a-2eaa-45ae-86a2-c039010768eb
    Text: Welcome to Colosscious.  We are the expert who spotlight-focus
    on providing the digital technology to bio and pharmaceutical
    companies, engaging in boosting the performances of new drug
    developments, quality control, manufacturing processes, and reducing
    the costs and duration by Big Data.
    ---



```python
V_DB_NAME = "chromadb"
chroma_client = PersistentClient(V_DB_NAME)
COLLECTION_NAME:str = 'test'
chroma_collection:Collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

```


```python
for n in nodes:
    print(storage_context.docstore.document_exists(n.id_))
```

    False
    False
    False


## Settings ( v0.10.0+)


```python
Settings.text_splitter = text_splitter
Settings.callback_manager = callback_manager
Settings.prompt_helper = prompt_helper
Settings.embed_model = embedding_model
Settings.llm = llm
```

## Create and store new embeddings to ChromaDB. 


```python
storage_context.docstore.add_documents(nodes)

# Deprecated from v0.9
#service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model, text_splitter=text_splitter,
#    prompt_helper=prompt_helper, callback_manager=callback_manager)
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context, storage_context=storage_context, show_progress=True,
# )
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context, 
    show_progress=True, 
)
```

    Generating embeddings: 100%|██████████| 3/3 [00:01<00:00,  2.60it/s]

    **********
    Trace: index_construction
        |_embedding ->  1.151678 seconds
    **********


    



```python
query_engine = index.as_query_engine()
```


```python
tokenizer(
    ["What Colosscious do?"],
    return_tensors="pt",
    add_special_tokens=False,
).input_ids.to("mps")
```




    tensor([[ 2058,  8906, 15098, 18440,  2083,  1033]], device='mps:0')




```python
query_resp = query_engine.query("What is flagship product of Colosscious")

print_resp(query_resp.response)
```

    2024-02-13 11:28:56,864 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  4.970373 seconds
          |_retrieve ->  0.348564 seconds
            |_embedding ->  0.343212 seconds
          |_synthesize ->  4.621614 seconds
            |_templating ->  1e-05 seconds
            |_llm ->  4.618833 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     Coloscius specializes in delivering digital technology
    solutions to bio and pharmaceutical companies, with a
    focus on enhancing the performance of new drug
    developments, improving quality control, optimizing
    manufacturing processes, and reducing costs and
    duration through Big Data. Therefore, it can be
    inferred that their flagship product or service likely
    revolves around these areas, providing advanced
    technologies and solutions tailored to the unique needs
    of the biotech and pharma industries.
    
    ************************************************************
     🚩END OF RESPONSE



```python
query_engine = index.as_chat_engine()
query_resp = query_engine.query("What is Pharmquer?")
print_resp(query_resp.response)
```

    2024-02-07 19:52:36,318 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
    PharmQuer is an international pharmacovigilance
    electronic system used in more than 80 countries for
    the collection and analysis of spontaneous case reports
    (adverse reactions to drugs). It is a free, web-based
    platform that allows users to report, review and
    analyze cases. The primary purpose of PharmQuer is to
    facilitate data sharing between regulatory agencies,
    pharmaceutical companies, academia, and other
    stakeholders in the field of pharmacovigilance.
    
    ************************************************************
     🚩END OF RESPONSE


### Persist Storage


```python
index.storage_context.persist()
```

## Load existing embeddings in ChromaDB.


```python
# Deprecated from v0.9
#service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model, text_splitter=text_splitter,
#    prompt_helper=prompt_helper, callback_manager=callback_manager)
# load your index from stored vectors

# Deprecated from v0.9
#index = VectorStoreIndex.from_vector_store(
#    vector_store, storage_context=storage_context, service_context=service_context
#)

# index = load_index_from_storage(storage_context) 
index = VectorStoreIndex.from_vector_store(
   vector_store, storage_context=storage_context,
)
```

    **********
    Trace: index_construction
    **********



```python
# create a query engine
query_engine = index.as_query_engine()
```


```python
response = query_engine.query("What is Colosscious?")
print_resp(response.response)
```

    2024-02-13 11:28:43,659 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  10.837251 seconds
          |_retrieve ->  0.474597 seconds
            |_embedding ->  0.470343 seconds
          |_synthesize ->  10.362329 seconds
            |_templating ->  1.3e-05 seconds
            |_llm ->  10.358294 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     Colosconscious is an expert entity that specializes in
    offering digital technology solutions to bio and
    pharmaceutical companies. Their focus is on enhancing
    the efficacy of new drug developments, ensuring quality
    control, optimizing manufacturing processes, and
    decreasing costs and durations through the application
    of Big Data.
    
    ************************************************************
     🚩END OF RESPONSE


## Use llama_index_object_array_reader


```python
# Preview: demo data
simple_ols[:2]
```




    [{'x1': 97.98219999874924,
      'x2': 99.84941752810117,
      'x3': 100.9727776594234,
      'y': 360.87650920565545},
     {'x1': 101.00077953260389,
      'x2': 99.87874921228179,
      'x3': 99.35642250227457,
      'y': 361.50488035486944}]




```python
loader = ObjectArrayReader()
```


```python
from llama_index.core.readers.base import Document
object_arrays:list[Document] = loader.load_data(file=simple_ols)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   x1      50 non-null     float64
     1   x2      50 non-null     float64
     2   x3      50 non-null     float64
     3   y       50 non-null     float64
    dtypes: float64(4)
    memory usage: 1.7 KB



```python
import pandas as pd
df = pd.DataFrame(simple_ols)
```


```python
object_arrays[0]
```




    Document(id_='f552deae-f8a7-4adb-9863-31cd036647b2', embedding=None, metadata={'columns': "['x1', 'x2', 'x3', 'y']", 'schema': 'None', 'shape': '(50, 4)'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='97.98219999874924, 99.84941752810117, 100.9727776594234, 360.87650920565545\n101.00077953260389, 99.87874921228179, 99.35642250227457, 361.50488035486944\n98.5109626677227, ...truncated', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n')




```python
# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    index = VectorStoreIndex.from_documents(
        documents=object_arrays, storage_context=storage_context, show_progress=True,
    )
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    #index = load_index_from_storage(storage_context)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context,
    )

```

    **********
    Trace: index_construction
    **********



```python
len(simple_ols)
```




    50




```python
object_arrays
```




    [Document(id_='f552deae-f8a7-4adb-9863-31cd036647b2', embedding=None, metadata={'columns': "['x1', 'x2', 'x3', 'y']", 'schema': 'None', 'shape': '(50, 4)'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='97.98219999874924, 99.84941752810117, 100.9727776594234, 360.87650920565545\n101.00077953260389, 99.87874921228179, 99.35642250227457, 361.50488035486944\n98.5109626677227, ...truncated', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n')]




```python
# create a query engine
query_engine = index.as_query_engine()
```


```python
response = query_engine.query("How many values with in the dataset?")
print_resp(response.response)
```

    2024-02-13 12:01:27,442 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 

    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  543.953731 seconds
          |_retrieve ->  0.361622 seconds
            |_embedding ->  0.356515 seconds
          |_synthesize ->  543.591766 seconds
            |_templating ->  1.5e-05 seconds
            |_llm ->  4.945246 seconds
            |_templating ->  2.2e-05 seconds
            |_llm ->  13.736413 seconds
            |_templating ->  4.2e-05 seconds
            |_llm ->  5.436325 seconds
            ... truncated
            |_llm ->  4.846492 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     The number of values in a dataset can be identified
    through its shape, which denotes the number of rows and
    columns it holds. However, without access to this
    specific information from the new context, it remains
    uncertain to ascertain the total count of values within
    the given dataset. Consequently, I will revert to the
    original answer: The number of values in a dataset is
    established by its shape, which comprises the number of
    rows and columns. For instance, if the shape manifests
    as (fifty, four), then you encounter fifty rows, each
    with four columns. Consequently, the dataset
    encompasses a total of fifty * four = two hundred
    values.
    
    ************************************************************
     🚩END OF RESPONSE



```python
df.shape
```




    (50, 4)




```python
response = query_engine.query("How many columns' name starts with 'x'?")
print_resp(response.response)
```

    2024-02-13 11:44:45,953 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  19.721272 seconds
          |_retrieve ->  0.514482 seconds
            |_embedding ->  0.507621 seconds
          |_synthesize ->  19.206533 seconds
            |_templating ->  1.1e-05 seconds
            |_llm ->  8.21483 seconds
            |_templating ->  3.1e-05 seconds
            |_llm ->  2.355871 seconds
            |_templating ->  1.1e-05 seconds
            |_llm ->  1.550853 seconds
            |_templating ->  4.9e-05 seconds
            |_llm ->  2.951295 seconds
            |_templating ->  6.7e-05 seconds
            |_llm ->  1.637973 seconds
            |_templating ->  4e-05 seconds
            |_llm ->  2.426145 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     Based on the new context provided, there are no
    columns named explicitly with the letter 'x' in the
    given data. The original answer stands repeating: The
    dataset consists of three columns whose names begin
    with the letter 'x'. These columns can be identified as
    'x1', 'x2', and 'x3'.
    
    ************************************************************
     🚩END OF RESPONSE



```python
response = query_engine.query("Can you tell me about the relationship between the dataset and Colosscious?")
print_resp(response.response)
```

    2024-02-13 12:14:29,172 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  6.774902 seconds
          |_retrieve ->  0.710536 seconds
            |_embedding ->  0.704498 seconds
          |_synthesize ->  6.063717 seconds
            |_templating ->  2.2e-05 seconds
            |_llm ->  6.058722 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     Coloscius is an expert organization that specializes
    in providing digital technology solutions to bio and
    pharmaceutical companies. Their focus areas include
    enhancing new drug developments, ensuring quality
    control, optimizing manufacturing processes, and
    reducing costs and durations through the application of
    Big Data. However, there's no direct mention or
    relationship stated between Coloscius and a specific
    dataset in the context information provided.
    
    ************************************************************
     🚩END OF RESPONSE



```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>97.982200</td>
      <td>99.849418</td>
      <td>100.972778</td>
      <td>360.876509</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101.000780</td>
      <td>99.878749</td>
      <td>99.356423</td>
      <td>361.504880</td>
    </tr>
    <tr>
      <th>2</th>
      <td>98.510963</td>
      <td>100.748550</td>
      <td>99.464651</td>
      <td>359.811761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.773359</td>
      <td>100.037229</td>
      <td>99.866572</td>
      <td>362.233696</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100.973598</td>
      <td>99.172480</td>
      <td>100.160933</td>
      <td>362.139116</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['x1'][:5])
print('Mean= ', df['x1'].mean())
```

    0     97.982200
    1    101.000780
    2     98.510963
    3    100.773359
    4    100.973598
    Name: x1, dtype: float64
    Mean=  100.07520939069373



```python
index.storage_context.persist()
```

### Sub Question Query Engine


```python
query_engine = index.as_query_engine()
```


```python
response = query_engine.query(
    "What about the dataset?"
)
print_resp(response.response)
```

    2024-02-13 11:32:22,255 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-13 11:38:01,007 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  344.353345 seconds
          |_retrieve ->  0.447767 seconds
            |_embedding ->  0.443857 seconds
          |_synthesize ->  343.905318 seconds
            |_templating ->  1.4e-05 seconds
            |_llm ->  5.152156 seconds
            |_templating ->  5.9e-05 seconds
            ...truncated
            |_templating ->  1.4e-05 seconds
            |_llm ->  1.230633 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     The new context does not provide enough details to
    draw conclusions about the nature of the dataset or its
    attributes.
    
    ************************************************************
     🚩END OF RESPONSE



```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
```


```python

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="summary_tool",
            description=f"Return the shape of the dataset and the basic summary of the dataset, such as mean, range, stddev of each columns.",
        ),
    ),
] 

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True,
    use_async=True,
)
```


```python
response = query_engine.query(
   "What about the dataset?"
)
print_resp(response.response )
```

    2024-02-13 11:38:31,494 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    Generated 4 sub questions.
    [summary_tool] Q: What is the shape of the dataset
    

    2024-02-13 11:38:34,709 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"

    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    [summary_tool] A:  The new context does not contribute to determining the shape of the dataset. Consequently, the original answer stays unaltered: The dataset comprises 50 instances with four features apiece, resulting in a total shape of (50, 4).
    [summary_tool] Q: What is the mean of each column in the dataset
    

    2024-02-13 11:39:02,830 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-13 11:39:02,836 - llama_index.core.query_engine.sub_question_query_engine - WARNING - (sub_question_query_engine.py:253) - [summary_tool] Failed to run What is the mean of each column in the dataset 


    [summary_tool] Failed to run What is the mean of each column in the dataset
    [summary_tool] Q: What is the range of each column in the dataset
    

    2024-02-13 11:39:17,299 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-13 11:39:17,316 - llama_index.core.query_engine.sub_question_query_engine - WARNING - (sub_question_query_engine.py:253) - [summary_tool] Failed to run What is the range of each column in the dataset 


    [summary_tool] Failed to run What is the range of each column in the dataset
    [summary_tool] Q: What is the standard deviation of each column in the dataset
    

    2024-02-13 11:39:22,956 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    [summary_tool] A:  To calculate the standard deviation for each column in a given dataset, you can use the following steps:
    
    1. First, find the mean (average) value for each column.
    2. Next, calculate the difference between each data point and the corresponding column mean.
    3. Find the square of each difference.
    4. Calculate the average of these squared differences for each column.
    5. Take the square root of that average to find the standard deviation.
    
    However, without access to any built-in functions or libraries, it's not possible to directly calculate the standard deviations from the context information provided. To do so, you would need to write a program that implements these steps and applies them to each column in the dataset.
    

    2024-02-13 11:39:25,364 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-13 11:39:29,576 - httpx - INFO - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 

    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    **********
    Trace: query
        |_query ->  101.519327 seconds
          |_templating ->  2.3e-05 seconds
          |_llm ->  9.84709 seconds
          |_sub_question ->  20.559445 seconds
            |_query ->  20.55857 seconds
              |_retrieve ->  0.57059 seconds
                |_embedding ->  0.557139 seconds
              |_synthesize ->  19.987694 seconds
                |_templating ->  1.3e-05 seconds
                |_llm ->  2.64631 seconds
                |_llm ->  2.644304 seconds
                |_templating ->  4.8e-05 seconds
                |_llm ->  5.660124 seconds
                |_llm ->  5.659924 seconds
                |_templating ->  7.7e-05 seconds
                |_llm ->  2.740417 seconds
                |_llm ->  2.740273 seconds
                |_templating ->  3e-05 seconds
                |_llm ->  2.293783 seconds
                |_llm ->  2.293711 seconds
                |_templating ->  4e-05 seconds
                |_llm ->  2.2244 seconds
                |_llm ->  2.224188 seconds
                |_templating ->  2.1e-05 seconds
                |_llm ->  2.329042 seconds
                |_llm ->  2.328894 seconds
                |_templating ->  4e-05 seconds
                |_llm ->  2.06445 seconds
                |_llm ->  2.064259 seconds
          |_sub_question ->  10.775827 seconds
            |_query ->  10.775703 seconds
              |_retrieve ->  0.484103 seconds
                |_embedding ->  0.478835 seconds
              |_synthesize ->  10.291405 seconds
                |_templating ->  8e-06 seconds
                |_llm ->  10.281698 seconds
                |_llm ->  10.281616 seconds
          |_sub_question ->  14.478265 seconds
            |_query ->  14.477974 seconds
              |_retrieve ->  0.494412 seconds
                |_embedding ->  0.48972 seconds
              |_synthesize ->  13.983397 seconds
                |_templating ->  9e-06 seconds
                |_llm ->  13.974939 seconds
                |_llm ->  13.974803 seconds
          |_sub_question ->  5.64837 seconds
            |_query ->  5.647935 seconds
              |_retrieve ->  0.531011 seconds
                |_embedding ->  0.525999 seconds
              |_synthesize ->  5.116657 seconds
                |_templating ->  9e-06 seconds
                |_llm ->  5.106247 seconds
                |_llm ->  5.106136 seconds
          |_synthesize ->  40.204981 seconds
            |_templating ->  3.2e-05 seconds
            |_llm ->  2.384732 seconds
            |_templating ->  2.8e-05 seconds
            |_llm ->  4.209638 seconds
            |_templating ->  4.7e-05 seconds
            |_llm ->  3.066046 seconds
            |_templating ->  2.6e-05 seconds
            |_llm ->  5.198477 seconds
            |_templating ->  2.3e-05 seconds
            |_llm ->  4.189788 seconds
            |_templating ->  3.6e-05 seconds
            |_llm ->  4.005403 seconds
            |_templating ->  4.6e-05 seconds
            |_llm ->  4.860746 seconds
            |_templating ->  3.3e-05 seconds
            |_llm ->  3.955889 seconds
            |_templating ->  3.6e-05 seconds
            |_llm ->  3.99903 seconds
            |_templating ->  4.2e-05 seconds
            |_llm ->  4.287925 seconds
    **********
    ✅ RESPONSE:
    ************************************************************
    
     To determine the standard deviation for each unique
    column within a given dataset, follow these steps:  1.
    Calculate the average value for every distinct column.
    6. Subtract each data point from its respective mean
    value in step 1 for the given column. 3. Square the
    results obtained in step 2. 4. Find the average of the
    squared differences computed in step 3 for that
    specific column. 5. Obtain the standard deviation by
    taking the square root of the result from step 4, then
    apply these calculations to each column within the
    dataset.
    
    ************************************************************
     🚩END OF RESPONSE



```python

```
