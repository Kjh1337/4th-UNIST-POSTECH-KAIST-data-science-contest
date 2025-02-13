import pandas as pd
from langchain_openai import ChatOpenAI
from typing import Dict, List
import json
from tqdm import tqdm
import logging
from pathlib import Path
import time
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        
    def load_corpus(self, dataset_name: str, use_expanded: bool = True) -> Dict[str, str]:
        """확장된 코퍼스 또는 원본 코퍼스 로드"""
        if use_expanded:
            corpus_path = Path(f"/home/elicer/FinanceRAG/dataset/{dataset_name}/corpus_prep.jsonl")
        else:
            corpus_path = Path(f"/home/elicer/FinanceRAG/dataset/{dataset_name}/corpus.jsonl")
            
        corpus_dict = {}
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                corpus_dict[doc['_id']] = doc['text']
                
        return corpus_dict
    
    def load_queries(self, dataset_name: str, use_expanded: bool = True) -> Dict[str, str]:
        """확장된 쿼리 또는 원본 쿼리 로드"""
        if use_expanded:
            query_path = Path(f"/home/elicer/FinanceRAG/dataset/{dataset_name}/queries_prep.jsonl")
        else:
            query_path = Path(f"/home/elicer/FinanceRAG/dataset/{dataset_name}/queries.jsonl")
            
        query_dict = {}
        with open(query_path, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                query_dict[query['_id']] = query['text']
                
        return query_dict
    
    def generate_answer(self, query: str, expanded_query: str, relevant_docs: List[str]) -> str:
        """확장된 쿼리와 코퍼스를 활용한 답변 생성"""
        context = "\n\n".join(relevant_docs)
        
        prompt = f"""Based on the given context, provide a clear and concise answer to the question.
        If the specific information is not directly stated in the context, indicate that.
        Do not make assumptions or generate specific details that are not in the context.
    
        Original Question: {query}
        Expanded Question: {expanded_query}

        Context:
        {context}

        Answer:"""
        
        try:
            response = self.llm.invoke(prompt).content
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            time.sleep(2)
            return "Error: Failed to generate answer"


def process_dataset(dataset_name: str):
    """데이터셋 처리 메인 함수"""
    results_dir = f"/home/elicer/FinanceRAG/results/{dataset_name}"
    output_path = f"{results_dir}/answers.xlsx"
    checkpoint_path = f"{results_dir}/answers_checkpoint.json"
    
    logger.info(f"Processing dataset: {dataset_name}")
    
    # 결과 파일 로드
    results_df = pd.read_csv(f"{results_dir}/results.csv")
    
    generator = AnswerGenerator()
    
    # 코퍼스와 쿼리 로드
    corpus_dict = generator.load_corpus(dataset_name)
    query_dict = generator.load_queries(dataset_name)
    
    # 기존 체크포인트 로드 (있는 경우)
    existing_answers = {}
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            existing_answers = json.load(f)
        logger.info(f"Loaded {len(existing_answers)} existing answers from checkpoint")
    
    answers = []
    unique_queries = results_df['query_id'].unique()
    
    try:
        for query_id in tqdm(unique_queries, desc="Generating answers"):
            # 이미 처리된 쿼리 건너뛰기
            if query_id in existing_answers:
                answers.append({
                    'query_id': query_id,
                    'answer': existing_answers[query_id]
                })
                continue
                
            # 현재 쿼리에 대한 관련 문서 가져오기
            relevant_doc_ids = results_df[results_df['query_id'] == query_id]['corpus_id'].tolist()
            relevant_docs = [corpus_dict[doc_id] for doc_id in relevant_doc_ids]
            
            # 답변 생성
            query_text = query_dict[query_id]
            answer = generator.generate_answer(query_text, relevant_docs)
            
            answers.append({
                'query_id': query_id,
                'answer': answer
            })
            
            # 체크포인트 저장 (매 10개 쿼리마다)
            if len(answers) % 10 == 0:
                current_answers = {a['query_id']: a['answer'] for a in answers}
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(current_answers, f, ensure_ascii=False, indent=2)
                logger.info(f"Checkpoint saved: {len(answers)} answers processed")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted. Saving current progress...")
    
    finally:
        # 최종 결과를 DataFrame으로 변환
        answers_df = pd.DataFrame(answers)
        
        # 결과 저장 (헤더 없이)
        answers_df.to_excel(output_path, index=False, header=False)
        logger.info(f"Final results saved to {output_path}")
        
        # 최종 체크포인트 저장
        final_answers = {a['query_id']: a['answer'] for a in answers}
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(final_answers, f, ensure_ascii=False, indent=2)
        logger.info("Final checkpoint saved")

def main():
    # 처리할 데이터셋 목록
    datasets = [
     #"FinQABench",
     #"FinanceBench", 
     #"FinQA",
     #"ConvFinQA",
     "TATQA", 
     #"MultiHiertt"
     ]  # 원하는 데이터셋 추가
    
    for dataset_name in datasets:
        results_dir = f"/home/elicer/FinanceRAG/results/{dataset_name}"
        output_path = f"{results_dir}/answers.xlsx"
        
        # 결과 파일 로드
        results_df = pd.read_csv(f"{results_dir}/results.csv")
        
        generator = AnswerGenerator()
        
        # 원본과 확장된 코퍼스/쿼리 모두 로드
        corpus_dict = generator.load_corpus(dataset_name, use_expanded=True)
        query_dict = generator.load_queries(dataset_name, use_expanded=False)
        expanded_query_dict = generator.load_queries(dataset_name, use_expanded=True)
        
        answers = []
        unique_queries = results_df['query_id'].unique()
     
        for query_id in tqdm(unique_queries, desc=f"Generating answers for {dataset_name}"):
            # 현재 쿼리에 대한 관련 문서 가져오기
            relevant_doc_ids = results_df[results_df['query_id'] == query_id]['corpus_id'].tolist()
            relevant_docs = [corpus_dict[doc_id] for doc_id in relevant_doc_ids]
            
            # 원본 쿼리와 확장된 쿼리 모두 사용
            query_text = query_dict[query_id]
            expanded_query_text = expanded_query_dict[query_id]
            
            logger.info(f"\nProcessing Query: {query_text}")
            logger.info(f"Expanded Query: {expanded_query_text}")
            
            answer = generator.generate_answer(query_text, expanded_query_text, relevant_docs)
            logger.info(f"Generated Answer: {answer}\n")
            
            answers.append({
                'query_id': query_id,
                'answer': answer
            })
        
        # 결과를 DataFrame으로 변환
        answers_df = pd.DataFrame(answers)
        
        # 결과 저장 (헤더 없이)
        answers_df.to_excel(output_path, index=False, header=False)
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()