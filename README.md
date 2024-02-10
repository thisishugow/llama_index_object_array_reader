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
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, get_response_synthesizer, PromptHelper
from llama_index.text_splitter import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ChromaVectorStore
from llamaindex_object_array_reader.dataset import simple_ols # import a simple dataset 
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.indices.query.schema import QueryBundle
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
from llama_index.llms import Ollama
from llama_index import ServiceContext, set_global_tokenizer
# from langchain.embeddings import HuggingFaceEmbedding, HuggingFaceInstructEmbeddings
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModel
from argparse import Namespace
from chromadb import Collection, PersistentClient
from dotenv import load_dotenv
from llamaindex_object_array_reader import ObjectArrayReader

```

    /Users/yuwang/Developments/python/llamaindex_object_array_reader/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
import logging
import sys
from llamaindex_object_array_reader._logging import logger

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
log = logger
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
    llm = Ollama(model="starling-lm:7b-alpha-q3_K_M")
    
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
    Node ID: ec94fbc2-06cf-41aa-b156-9c8753d101d1
    Text: You can do data integration, management, analysis and composing
    reports and dashboards with Pharmquer, and then automatize all your
    works.
    ---
    Node ID: 91c1aaef-6216-4242-b187-906db3939929
    Text: Colosscious' flagship product, Pharmquer, is an enterprise level
    software of manufacturing and business intelligence, which is
    architected especially for the industry.
    ---
    Node ID: 1dc13ad7-d352-4219-a63c-f174adb3c933
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

    2024-02-08 15:18:35,622 - chromadb.telemetry.product.posthog - [32;20mINFO[0m - (posthog.py:20) - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information. 


    Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.



```python
for n in nodes:
    print(storage_context.docstore.document_exists(n.id_))
```

    False
    False
    False


## Create and store new embeddings to ChromaDB. 


```python
storage_context.docstore.add_documents(nodes)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model, text_splitter=text_splitter,
    prompt_helper=prompt_helper,)
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context, storage_context=storage_context, show_progress=True,
# )
index = VectorStoreIndex(
    nodes, service_context=service_context, storage_context=storage_context, show_progress=True,
)
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Generating embeddings: 100%|██████████| 3/3 [00:00<00:00,  5.29it/s]
    2024-02-07 22:47:49,898 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35 


    Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35


    2024-02-07 22:47:49,898 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862 


    Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862


    2024-02-07 22:47:49,899 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a 


    Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a



```python
# example: 
# "GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:"
# ref: https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
sep = '<|end_of_turn|>'
resp_prompt_temp = "GPT4 Correct Assistant: "
```


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

    2024-02-07 19:52:25,347 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
    Sorry, I cannot answer your query without using any
    more tools.
    
    ************************************************************
     🚩END OF RESPONSE



```python
query_engine = index.as_chat_engine()
query_resp = query_engine.query("What is Pharmquer?")
print_resp(query_resp.response)
```

    2024-02-07 19:52:36,318 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


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


## Load existing embeddings in ChromaDB.


```python
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model, text_splitter=text_splitter,
    prompt_helper=prompt_helper,)
# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, service_context=service_context
)

```


```python
# create a query engine
query_engine = index.as_query_engine()
```


```python
response = query_engine.query("What is Colosscious?")
print_resp(response.response)
```

    2024-02-08 13:58:07,575 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35 


    Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35


    2024-02-08 13:58:07,576 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862 


    Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862


    2024-02-08 13:58:07,577 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a 


    Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a


    2024-02-08 13:58:19,513 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:23,079 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:28,070 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:32,393 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:35,861 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:41,581 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:46,546 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:52,140 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"


    2024-02-08 13:58:57,621 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
     Colosscious, also known as "Colosconscious," is a
    specialized entity dedicated to providing digital
    technology solutions for bio and pharmaceutical
    companies. Their primary objectives include enhancing
    new drug development effectiveness, upholding quality
    standards, optimizing manufacturing processes, and
    reducing expenses related to these domains. It's
    essential to acknowledge that the context provided may
    contain a typo (Colosconscious instead of Colosscious),
    which could result in confusion. While it's possible
    that the accurate name is "Colosconscious," without
    additional clarification or detailed context, this
    assumption cannot be confirmed with certainty.
    
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
from llama_index.readers.schema.base import Document
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




    Document(id_='1f219b5d-0518-4048-a8fa-c08a8e0ec816', embedding=None, metadata={'columns': "['x1', 'x2', 'x3', 'y']", 'schema': 'None', 'shape': '(50, 4)'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='01044d00146a997fca8953cf8ba579cb4492a76d451ca3aa31645e0e2e9bcb89', text='97.98219999874924, 99.84941752810117, 100.9727776594234, 360.87650920565545\n101.00077953260389, 99.87874921228179, 99.35642250227457, 361.50488035486944\n98.5109626677227, 100.7485502397903, 99.46465098250788, 359.8117609861218\n100.77335929310553, 100.03722922045552, 99.86657209922947, 362.2336960397953\n100.97359840386007, 99.1724799721807, 100.16093297144785, 362.1391160315852\n100.18799255929102, 100.55900119891184, 100.61532849440285, 363.29752977180965\n100.9157547652626, 98.61649241995889, 99.06726035297895, 359.7975894964005\n101.04615952660859, 102.00920930524853, 100.16419028246959, 364.8003752715575\n99.46321248760913, 100.23898461781165, 100.4603474082993, 361.9810830871964\n101.01997365057879, 100.70311893925478, 100.35193718659701, 363.88982333927027\n100.13690655914681, 100.07882115404048, 98.55349011863521, 359.46137480234387\n101.06426034086174, 100.39468013724496, 99.59524654141205, 362.4291116312047\n98.37827347228104, 103.02974783668428, 100.1611399406794, 362.8735393636648\n99.97507192990182, 98.90754655604644, 99.80434032022505, 360.25326008000667\n100.39532804084602, 100.00783015782618, 98.84412853695146, 360.1443934124746\n100.53538350964381, 101.45593826545792, 101.35656192129933, 365.686428125867\n101.82462124282185, 100.60168747124192, 101.17271531836492, 365.98868463340847\n100.20083842899689, 100.08207154841824, 99.58637885148526, 361.20837231254296\n101.21055853157635, 100.14337012301043, 97.83946914592629, 359.5083658057684\n99.18284850767726, 99.36415064348877, 99.44933313879001, 359.2461732873584\n98.68076906156287, 99.88606328063753, 100.63302113481906, 361.1047253879728\n100.69262845160499, 100.95667625950557, 100.42976056344008, 363.89692658755564\n99.28259711401759, 99.70388277366841, 101.68434722375862, 363.2876025473195\n100.06352462146366, 98.88111045420715, 98.17662400762073, 357.68289832611293\n99.30093777662633, 99.66957998231912, 101.04827928253668, 362.24405534917537\n101.80808125496162, 98.90879253371892, 100.43144827373477, 363.22961316755584\n100.87937300919184, 100.21857314642527, 98.7095417586287, 360.6345955200315\n101.26476906248293, 100.78711425646654, 100.29822225418964, 364.14048570001194\n99.91296978569375, 99.81832037463671, 100.93755148875996, 362.85330879179776\n98.33923043666618, 101.41609851401992, 100.9069226929846, 362.5750535271962\n99.16537289151567, 101.53101959448773, 98.68726411895874, 359.9607243551743\n99.03719843850321, 101.11878894468508, 100.96547923761082, 363.1452988224573\n99.43406666585818, 100.91739344461386, 99.36206117770445, 360.78473992459374\n99.68590044334066, 99.48627235799435, 99.56930464452118, 360.089118523317\n97.32232213736515, 99.97594783847929, 100.69691843112757, 359.8391638384025\n99.39074229940556, 99.93236851645756, 98.2362250018063, 358.01656116479893\n100.77590520414071, 99.69003697728576, 98.90965598576909, 360.3683307126241\n99.68679150345386, 99.02927646327905, 101.76626758973612, 363.239031940659\n100.64548154386631, 99.20184240211881, 100.01344022902087, 361.5760580247661\n98.99353686788966, 99.96912376997722, 101.61919061254812, 363.11424644961215\n99.61923454090937, 99.106406018837, 100.84958819151115, 361.7497627252443\n99.07093312420517, 100.95979226903094, 99.95112469407546, 361.39111973582743\n100.07233850453227, 99.7129946461946, 101.44068386026443, 363.74407241841885\n99.72656237514776, 98.72754826499948, 99.97952513372913, 360.1084405895639\n99.52984537048624, 102.18311836300242, 98.52851110733191, 360.68518093107895\n101.14589750687492, 101.45663623211762, 100.70204076103222, 365.2772452093873\n100.69113884387745, 100.5140731983062, 98.73201138024642, 360.73859881401285\n100.97373793175905, 99.21840952279409, 98.11881715092599, 358.8678844756544\n103.14715709169663, 98.60042261009588, 100.7938715049434, 364.9675528117923\n100.6433422264419, 100.72347518969544, 99.54248360001975, 362.1927963663157', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n')




```python
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model,)
index = VectorStoreIndex.from_documents(
    documents=object_arrays, service_context=service_context,  storage_context=storage_context, show_progress=True,
)
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Parsing nodes:   0%|          | 0/1 [00:00<?, ?it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (2225 > 512). Running this sequence through the model will result in indexing errors
    Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 49.78it/s]
    Generating embeddings: 100%|██████████| 4/4 [00:00<00:00, 10.85it/s]
    2024-02-08 15:20:20,129 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35 


    Add of existing embedding ID: dc0f865e-90c8-42b0-9239-19625ebcef35


    2024-02-08 15:20:20,130 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862 


    Add of existing embedding ID: 1f7abdb8-4dbb-4f9d-9398-f59fb630b862


    2024-02-08 15:20:20,130 - chromadb.segment.impl.vector.local_persistent_hnsw - [33;20mWARNING[0m - (local_persistent_hnsw.py:271) - Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a 


    Add of existing embedding ID: cb553733-838a-421b-89bf-c582fe90182a



```python
len(simple_ols)
```




    50




```python
object_arrays
```




    [Document(id_='a044bf0f-199b-4cab-822a-b61305ba9495', embedding=None, metadata={'columns': "['x1', 'x2', 'x3', 'y']", 'schema': 'None', 'shape': '(50, 4)'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, hash='9f4c001f1f1ad7d1636d4457b18f1cbb05d3359492cf2267e2666fabf435f140', text='97.98219999874924, 99.84941752810117, 100.9727776594234, 360.87650920565545\n101.00077953260389, 99.87874921228179, 99.35642250227457, 361.50488035486944\n98.5109626677227, 100.7485502397903, 99.46465098250788, 359.8117609861218\n100.77335929310553, 100.03722922045552, 99.86657209922947, 362.2336960397953\n100.97359840386007, 99.1724799721807, 100.16093297144785, 362.1391160315852\n100.18799255929102, 100.55900119891184, 100.61532849440285, 363.29752977180965\n100.9157547652626, 98.61649241995889, 99.06726035297895, 359.7975894964005\n101.04615952660859, 102.00920930524853, 100.16419028246959, 364.8003752715575\n99.46321248760913, 100.23898461781165, 100.4603474082993, 361.9810830871964\n101.01997365057879, 100.70311893925478, 100.35193718659701, 363.88982333927027\n100.13690655914681, 100.07882115404048, 98.55349011863521, 359.46137480234387\n101.06426034086174, 100.39468013724496, 99.59524654141205, 362.4291116312047\n98.37827347228104, 103.02974783668428, 100.1611399406794, 362.8735393636648\n99.97507192990182, 98.90754655604644, 99.80434032022505, 360.25326008000667\n100.39532804084602, 100.00783015782618, 98.84412853695146, 360.1443934124746\n100.53538350964381, 101.45593826545792, 101.35656192129933, 365.686428125867\n101.82462124282185, 100.60168747124192, 101.17271531836492, 365.98868463340847\n100.20083842899689, 100.08207154841824, 99.58637885148526, 361.20837231254296\n101.21055853157635, 100.14337012301043, 97.83946914592629, 359.5083658057684\n99.18284850767726, 99.36415064348877, 99.44933313879001, 359.2461732873584\n98.68076906156287, 99.88606328063753, 100.63302113481906, 361.1047253879728\n100.69262845160499, 100.95667625950557, 100.42976056344008, 363.89692658755564\n99.28259711401759, 99.70388277366841, 101.68434722375862, 363.2876025473195\n100.06352462146366, 98.88111045420715, 98.17662400762073, 357.68289832611293\n99.30093777662633, 99.66957998231912, 101.04827928253668, 362.24405534917537\n101.80808125496162, 98.90879253371892, 100.43144827373477, 363.22961316755584\n100.87937300919184, 100.21857314642527, 98.7095417586287, 360.6345955200315\n101.26476906248293, 100.78711425646654, 100.29822225418964, 364.14048570001194\n99.91296978569375, 99.81832037463671, 100.93755148875996, 362.85330879179776\n98.33923043666618, 101.41609851401992, 100.9069226929846, 362.5750535271962\n99.16537289151567, 101.53101959448773, 98.68726411895874, 359.9607243551743\n99.03719843850321, 101.11878894468508, 100.96547923761082, 363.1452988224573\n99.43406666585818, 100.91739344461386, 99.36206117770445, 360.78473992459374\n99.68590044334066, 99.48627235799435, 99.56930464452118, 360.089118523317\n97.32232213736515, 99.97594783847929, 100.69691843112757, 359.8391638384025\n99.39074229940556, 99.93236851645756, 98.2362250018063, 358.01656116479893\n100.77590520414071, 99.69003697728576, 98.90965598576909, 360.3683307126241\n99.68679150345386, 99.02927646327905, 101.76626758973612, 363.239031940659\n100.64548154386631, 99.20184240211881, 100.01344022902087, 361.5760580247661\n98.99353686788966, 99.96912376997722, 101.61919061254812, 363.11424644961215\n99.61923454090937, 99.106406018837, 100.84958819151115, 361.7497627252443\n99.07093312420517, 100.95979226903094, 99.95112469407546, 361.39111973582743\n100.07233850453227, 99.7129946461946, 101.44068386026443, 363.74407241841885\n99.72656237514776, 98.72754826499948, 99.97952513372913, 360.1084405895639\n99.52984537048624, 102.18311836300242, 98.52851110733191, 360.68518093107895\n101.14589750687492, 101.45663623211762, 100.70204076103222, 365.2772452093873\n100.69113884387745, 100.5140731983062, 98.73201138024642, 360.73859881401285\n100.97373793175905, 99.21840952279409, 98.11881715092599, 358.8678844756544\n103.14715709169663, 98.60042261009588, 100.7938715049434, 364.9675528117923\n100.6433422264419, 100.72347518969544, 99.54248360001975, 362.1927963663157', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n')]




```python
# create a query engine
query_engine = index.as_query_engine()
```


```python
response = query_engine.query("How many values with in the dataset?")
print_resp(response.response)
```

    2024-02-08 15:12:13,321 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
     The dataset contains a total of 50 rows, as indicated
    by the shape of (50, 4) for both sets of data provided.
    However, it's important to note that the context
    information doesn't explicitly state that there are two
    sets of data provided. But based on the alternating
    columns in each block of data, we can infer that there
    are indeed two separate sets of data with 50 rows each.
    
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

    2024-02-08 15:13:24,979 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
     There are three columns' names that start with 'x',
    which are 'x1', 'x2', and 'x3'.
    
    ************************************************************
     🚩END OF RESPONSE



```python
response = query_engine.query("What is the average of column 'x1'?")
print_resp(response.response)
```

    2024-02-08 15:21:03,508 - httpx - [32;20mINFO[0m - (_client.py:1027) - HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK" 


    HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
    ✅ RESPONSE:
    ************************************************************
    
     To find the average of column 'x1', we need to sum all
    the values in column 'x1' and then divide by the total
    number of rows (50). Here are all the values in column
    'x1':  99.07093312420517, 100.84958819151115,
    99.72656237514776, ... , 100.97373793175905  Adding all
    the values gives us:  99.07093312420517 +
    100.84958819151115 + ... + 100.97373793175905 = (sum of
    all x1 values)  Now, we need to divide the sum by the
    total number of rows, which is 50:  (sum of all x1
    values) / 50 = average of column 'x1'  Without
    calculating the exact sum, we can see that the average
    value lies between 98.73 (lowest value) and 100.97
    (highest value). However, without performing the actual
    calculation, we cannot provide an exact numerical
    answer for the average of column 'x1'.  Please note
    that providing the exact average would require further
    calculations that go beyond the scope of this AI model.
    
    ************************************************************
     🚩END OF RESPONSE



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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

