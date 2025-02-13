
import json
import os
import re
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import shutil

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Replace all Unicode escape sequences with a space."""
    return re.sub(r"(\\u[0-9A-Fa-f]{4})+", " ", text)

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load a JSONL file and return its content as a list of dictionaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(clean_text(line.strip())) for line in f]

def save_jsonl(file_path: Path, data: List[Dict], ensure_ascii: bool = True):
    """Save a list of dictionaries to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")

class QueryExpander:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv()
        self.llm = ChatOpenAI(model=model_name)
        
    def load_prompt(self, dataset_name: str, key: str = "queries") -> str:
        """Load prompt template for the specified dataset"""
        prompt_path = Path("./prompt.json")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)["pre_retrieval"][key]
            
        if dataset_name not in prompts:
            raise ValueError(f"Prompt not found for dataset '{dataset_name}'")
        return prompts[dataset_name]
    
    def expand_queries(self, dataset_name: str, dataset_dir: str, overwrite: bool = False) -> List[Dict]:
        """Expand queries for a given dataset using the LLM."""
        data = load_jsonl(Path(f"{dataset_dir}/{dataset_name}/queries.jsonl"))
        prompt_template = self.load_prompt(dataset_name, "queries")

        expanded_queries = []
        for item in tqdm(data, desc=f"Expanding queries for {dataset_name}"):
            item_text = item["text"]
            prompt = f"{prompt_template}\n\nQuery: {item_text}"
            try:
                new_text = self.llm.invoke(prompt).content
                expanded_queries.append({
                    "_id": item["_id"],
                    "title": item.get("title", ""),
                    "text": f"{item_text}\n\n{new_text}",
                })
            except Exception as e:
                logger.error(f"Query expansion failed for {item['_id']}: {str(e)}")
                expanded_queries.append(item)

        save_path = Path(f"{dataset_dir}/{dataset_name}/queries_prep.jsonl")
        if not save_path.is_file() or overwrite:
            save_jsonl(save_path, expanded_queries)
        
        return expanded_queries

class TablePreprocessor:
    @staticmethod
    def extract_table(text: str, dataset_name: str) -> str:
        lines = text.split("\n")
        results = []
        table_start = None
        
        # 테이블이 있는 데이터셋
        if dataset_name in ["TATQA", "FinQA", "MultiHiertt", "ConvFinQA"]:
            if dataset_name == "TATQA":
                # TATQA의 경우 테이블 앞뒤 문맥도 함께 보존
                for i, line in enumerate(lines):
                    if line.startswith("| "):
                        if table_start is None:
                            table_start = i
                            # 테이블 앞의 문맥도 포함
                            if i > 0:
                                results.append(lines[i-1])
                        
                        if i + 1 == len(lines) or not lines[i + 1].startswith("| "):
                            table_content = "\n".join(lines[table_start:i + 1])
                            results.append(table_content.strip())
                            # 테이블 뒤의 문맥도 포함
                            if i + 1 < len(lines):
                                results.append(lines[i+1])
                            table_start = None
                            
            elif dataset_name == "FinQA":
                # FinQA의 경우 테이블 구조를 보존하면서 수치 데이터에 집중
                for i, line in enumerate(lines):
                    if line.startswith("| "):
                        if table_start is None:
                            table_start = i
                        
                        if i + 1 == len(lines) or not lines[i + 1].startswith("| "):
                            table_content = "\n".join(lines[table_start:i + 1])
                            results.append(table_content.strip())
                            table_start = None
                            
            elif dataset_name == "MultiHiertt":
                # MultiHiertt의 경우 계층 구조를 보존
                for i, line in enumerate(lines):
                    if line.startswith("| "):
                        if table_start is None:
                            table_start = i
                        
                        if i + 1 == len(lines) or not lines[i + 1].startswith("| "):
                            table_content = "\n".join(lines[table_start:i + 1])
                            results.append(table_content.strip())
                            table_start = None
                            
            elif dataset_name == "ConvFinQA":
                # ConvFinQA의 경우 대화 컨텍스트와 테이블 관계를 보존
                for i, line in enumerate(lines):
                    if line.startswith("| "):
                        if table_start is None:
                            table_start = i
                            # 대화 컨텍스트 보존
                            if i > 0:
                                results.append(lines[i-1])
                        
                        if i + 1 == len(lines) or not lines[i + 1].startswith("| "):
                            table_content = "\n".join(lines[table_start:i + 1])
                            results.append(table_content.strip())
                            table_start = None
        
        # 테이블이 없는 데이터셋 (FinDER, FinQABench, FinanceBench)
        else:
            # 텍스트 데이터 전처리
            processed_lines = []
            for line in lines:
                # 빈 줄 제거
                if line.strip():
                    # 숫자 데이터 주변 공백 추가로 강조
                    line = re.sub(r'(\d+\.?\d*)', r' \1 ', line)
                    # 금융 관련 대문자 키워드 강조
                    line = re.sub(r'([A-Z]{2,})', r' \1 ', line)
                    processed_lines.append(line)
            
            results = processed_lines

        if results:
            return "\n\n".join(results)
        else:
            return text
    
    @staticmethod
    def compress_corpus(dataset_name: str, dataset_dir: str):
        """Compress specific sections of corpus text for given datasets."""
        corpus = load_jsonl(Path(f"{dataset_dir}/{dataset_name}/corpus.jsonl"))
        for item in corpus:
            item["text"] = TablePreprocessor.extract_table(item["text"], dataset_name)
        save_jsonl(Path(f"{dataset_dir}/{dataset_name}/corpus_prep.jsonl"), corpus)
    
    @staticmethod
    def copy_corpus(dataset_name: str, dataset_dir: str):
        """Copy corpus file without modification."""
        from_path = os.path.join(dataset_dir, dataset_name, 'corpus.jsonl')
        to_path = os.path.join(dataset_dir, dataset_name, 'corpus_prep.jsonl')
        shutil.copy(from_path, to_path)

class HybridSearchReranker:
    def __init__(self):
        self.bm25 = None
        self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.corpus_embeddings = None
        self.tokenized_corpus = None
        self.scaler = MinMaxScaler()
        
        # 데이터셋별 가중치
        self.dataset_weights = {
            'FinDER': {'bm25': 0.95, 'semantic': 0.05},     # FinDER 최적화
            'FinQABench': {'bm25': 0.7, 'semantic': 0.3},   # 원래 가중치 유지
            'TATQA': {'bm25': 0.5, 'semantic': 0.5},
            'FinQA': {'bm25': 0.6, 'semantic': 0.4},
            'MultiHiertt': {'bm25': 0.4, 'semantic': 0.6},
            'ConvFinQA': {'bm25': 0.3, 'semantic': 0.7},
            'FinanceBench': {'bm25': 0.5, 'semantic': 0.5}
        }
    
    def prepare_corpus(self, corpus: List[Dict]):
        logger.info("Preparing corpus for hybrid search...")
        self.tokenized_corpus = [doc['text'].split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        texts = [doc['text'] for doc in corpus]
        self.corpus_embeddings = self.semantic_model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
    
    def hybrid_search(self, query: str, corpus: List[Dict], dataset_name: str, top_k: int = 100) -> Dict[str, float]:
        tokenized_query = query.split()
        weights = self.dataset_weights.get(dataset_name, {'bm25': 0.5, 'semantic': 0.5}).copy()
        
        # BM25 점수 계산
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 시맨틱 점수 계산
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        semantic_scores = torch.cosine_similarity(
            query_embedding.unsqueeze(0), 
            self.corpus_embeddings
        ).cpu().numpy()
        
        if dataset_name == 'FinDER':
            # FinDER 특화 처리
            if any(char.isdigit() for char in query) or \
               any(keyword in query.upper() for keyword in ['STOCK', 'PRICE', 'MARKET', 'TRADE', 'USD', 'SHARE']):
                weights['bm25'] = 0.98
                weights['semantic'] = 0.02
            # BM25 점수 원본 사용
            normalized_bm25 = bm25_scores
            normalized_semantic = self.scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()
        else:
            # 다른 데이터셋은 원래 방식대로 처리
            normalized_bm25 = self.scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
            normalized_semantic = self.scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()
        
        # 최종 점수 계산
        hybrid_scores = (weights['bm25'] * normalized_bm25) + (weights['semantic'] * normalized_semantic)
        
        # 상위 문서 선택
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        results = {}
        for idx in top_indices:
            doc_id = corpus[idx]['_id']
            results[doc_id] = float(hybrid_scores[idx])
            
        return results

def cross_encoder_rerank(queries: List[Dict], 
                        corpus: List[Dict], 
                        initial_results: Dict[str, Dict[str, float]], 
                        top_k: int = 10) -> Dict[str, Dict[str, float]]:
    logger.info("Loading cross-encoder model...")
    
    model = CrossEncoder(
        'jinaai/jina-reranker-v2-base-multilingual',
        trust_remote_code=True,
        config_args={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager"
        }
    )
    
    final_results = {}
    batch_size = 4
    
    for query in tqdm(queries, desc="Cross-encoder reranking"):
        query_id = query['_id']
        if query_id not in initial_results:
            continue
            
        relevant_docs = [doc for doc in corpus if doc['_id'] in initial_results[query_id]]
        
        pairs = []
        doc_ids = []
        for doc in relevant_docs:
            pairs.append([query['text'], doc['text']])
            doc_ids.append(doc['_id'])
        
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            scores = model.predict(
                batch_pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            all_scores.extend(scores)
        
        doc_scores = {doc_id: float(score) for doc_id, score in zip(doc_ids, all_scores)}
        top_docs = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        final_results[query_id] = top_docs
    
    return final_results

def load_qrels(dataset_name: str) -> Dict:
    qrels_dict = {}
    qrels_path = f"/home/elicer/FinanceRAG/dataset/{dataset_name}/qrels.tsv"
    
    logger.info(f"Loading qrels from {qrels_path}")
    
    try:
        with open(qrels_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            
            for line_num, line in enumerate(f, 2):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    query_id = parts[0]
                    doc_id = parts[1]
                    score = float(parts[2])
                    
                    if query_id not in qrels_dict:
                        qrels_dict[query_id] = {}
                    qrels_dict[query_id][doc_id] = score
                    
                except (IndexError, ValueError) as e:
                    logger.warning(f"Invalid format at line {line_num}: {e}")
                    continue
                
        if not qrels_dict:
            logger.warning("No valid qrels data loaded!")
            
    except FileNotFoundError:
        logger.error(f"qrels file not found: {qrels_path}")
        return {}
        
    logger.info(f"Loaded qrels for {len(qrels_dict)} queries")
    return qrels_dict

def calculate_ndcg(qrels_dict: Dict, results: Dict, k: int = 10) -> float:
    from sklearn.metrics import ndcg_score
    
    ndcg_scores = []
    for query_id in results:
        if query_id not in qrels_dict:
            continue
            
        pred_docs = list(results[query_id].keys())
        
        true_rel = np.zeros(len(pred_docs))
        for i, doc_id in enumerate(pred_docs):
            if doc_id in qrels_dict[query_id]:
                true_rel[i] = qrels_dict[query_id][doc_id]
        
        pred_scores = np.array(list(results[query_id].values()))
        
        if len(true_rel) > 0:
            query_ndcg = ndcg_score([true_rel], [pred_scores])
            ndcg_scores.append(query_ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def save_results(results: Dict[str, Dict[str, float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.jsonl", 'w') as f:
        for query_id, doc_scores in results.items():
            for doc_id, score in doc_scores.items():
                result = {
                    "query_id": query_id,
                    "corpus_id": doc_id,
                    "score": score
                }
                f.write(json.dumps(result) + '\n')
    
    with open(f"{output_dir}/results.csv", 'w') as f:
        f.write("query_id,corpus_id\n")
        for query_id, doc_scores in results.items():
            for doc_id in doc_scores:
                f.write(f"{query_id},{doc_id}\n")

def main():
    DATASETS = [
        #'FinDER',
        #'FinQABench',
        #'ConvFinQA',
        #'FinanceBench',
        'MultiHiertt',
        #'TATQA',
        #'FinQA'
    ]
    
    BASE_DIR = '/home/elicer/FinanceRAG'
    DATASET_DIR = f"{BASE_DIR}/dataset"
    OUTPUT_DIR = f"{BASE_DIR}/results"
    
    # Initialize components
    query_expander = QueryExpander(model_name="gpt-4o-mini")
    table_preprocessor = TablePreprocessor()
    
    for dataset_name in DATASETS:
        try:
            logger.info(f"\n=== Processing dataset: {dataset_name} ===")
            
            # 1. Query Expansion
            logger.info("\n1. Query Expansion")
            expanded_queries = query_expander.expand_queries(dataset_name, DATASET_DIR)
            
            # 2. Table Preprocessing
            logger.info("\n2. Table Preprocessing")
            if dataset_name in ["MultiHiertt", "TATQA", 'FinQA', 'ConvFinQA', 'FinDER', 'FinanceBench']:
                table_preprocessor.compress_corpus(dataset_name, DATASET_DIR)
            else:
                table_preprocessor.copy_corpus(dataset_name, DATASET_DIR)
            
            # Load preprocessed data
            queries = load_jsonl(Path(f"{DATASET_DIR}/{dataset_name}/queries_prep.jsonl"))
            corpus = load_jsonl(Path(f"{DATASET_DIR}/{dataset_name}/corpus_prep.jsonl"))
            
            # 3. Hybrid Search
            logger.info("\n3. Hybrid Search")
            # main 함수 내부
            hybrid_searcher = HybridSearchReranker()
            hybrid_searcher.prepare_corpus(corpus)

            hybrid_results = {}
            for query in tqdm(queries, desc=f"Hybrid search for {dataset_name}"):
                hybrid_results[query['_id']] = hybrid_searcher.hybrid_search(
                    query['text'], 
                    corpus,
                    dataset_name,
                    top_k=200
                )
            
            # 4. Cross-encoder reranking
            logger.info("\n4. Cross-encoder Reranking")
            final_results = cross_encoder_rerank(
                queries, 
                corpus, 
                hybrid_results, 
                top_k=10
            )
            
            # Calculate and print nDCG@10
            qrels_dict = load_qrels(dataset_name)
            ndcg_10 = calculate_ndcg(qrels_dict, final_results, k=10)
            logger.info(f"nDCG@10 for {dataset_name}: {ndcg_10:.4f}")
            
            # Save results
            dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
            logger.info(f"Saving results for {dataset_name}...")
            save_results(final_results, dataset_output_dir)
            logger.info(f"Results saved to {dataset_output_dir}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue

if __name__ == "__main__":
    main()