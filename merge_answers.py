# merge_answers.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_answers():
    datasets = ["FinDER", "FinQABench", "TATQA", "FinQA", "MultiHiertt", "ConvFinQA", "FinanceBench"]
    all_answers = []
    
    for dataset in datasets:
        file_path = f"/home/elicer/FinanceRAG/results/{dataset}/answers.xlsx"
        # header=None 추가하여 첫 번째 행을 데이터로 읽도록 함
        df = pd.read_excel(file_path, names=['query_id', 'answer'], header=None)
        logger.info(f"{dataset}: {len(df)} answers loaded")
        all_answers.append(df)
    
    # 모든 데이터프레임 병합
    final_df = pd.concat(all_answers, ignore_index=True)
    
    # 결과 저장 (헤더 없이)
    output_path = "/home/elicer/FinanceRAG/results/merged_answers.xlsx"
    final_df.to_excel(output_path, index=False, header=False)
    logger.info(f"\nTotal answers merged: {len(final_df)}")
    logger.info(f"Saved to: {output_path}")

if __name__ == "__main__":
    merge_answers()