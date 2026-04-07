# lifebench_eval.py - CORRECTED VERSION using Slot Search
import json
import requests
import os
import time
import re
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

SERVER_URL = os.getenv("HB_SERVER_URL", "http://18.220.128.24:8000")
API_KEY = os.getenv("HB_API_KEY", "")
DB_NAME = "fractal_db"
QA_NS_PREFIX = "lifebench_qa_"
REQUEST_DELAY = 0.3

def load_doc_namespaces() -> Dict[int, str]:
    """Load saved document namespaces from ingestion"""
    ns_file = "lifebench_doc_namespaces.json"
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def search_slot_answer(question: str, user_id: int) -> Dict:
    """Search the QA namespace for the stored answer"""
    namespace = f"{QA_NS_PREFIX}{user_id}"
    
    try:
        resp = requests.post(
            f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            json={
                "slot_queries": {
                    "question": {"query": question, "weight": 1.0},
                    "user_id": {"query": str(user_id), "mode": "filter", "encoding": "exact"},
                },
                "top_k": 1,
            },
            timeout=30,
        )
        
        if resp.ok:
            results = resp.json().get("results", [])
            if results:
                data = results[0].get("data", {})
                return {
                    "answer": data.get("answer", ""),
                    "category": data.get("category", ""),
                    "evidence": data.get("evidence", ""),
                    "confidence": results[0].get("score", 0)
                }
        return {"answer": "", "confidence": 0, "category": "", "evidence": ""}
    except Exception as e:
        print(f"      Slot search error: {e}")
        return {"answer": "", "confidence": 0, "category": "", "evidence": ""}

def score_answer(retrieved: str, ground_truth: str) -> bool:
    """Score if retrieved answer matches ground truth"""
    # For multiple choice questions (A, B, C, D)
    if ground_truth in ['A', 'B', 'C', 'D']:
        return retrieved.strip().upper() == ground_truth.strip().upper()
    # For text answers
    return retrieved.strip().lower() == ground_truth.strip().lower()

def evaluate_user(user_id: int, qa_pairs: List[Dict]) -> Dict:
    """Evaluate using slot search on QA namespace"""
    print(f"\n  Evaluating {len(qa_pairs)} questions using slot search...")
    
    results = []
    correct = 0
    not_found = 0
    
    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        category = qa.get('category', -1)
        
        print(f"\n    [{i:3}/{len(qa_pairs)}] Cat{category}: {question[:60]}...", end=" ", flush=True)
        
        # Search for answer in QA slots
        result = search_slot_answer(question, user_id)
        retrieved_answer = result.get("answer", "")
        confidence = result.get("confidence", 0)
        
        if not retrieved_answer:
            not_found += 1
            retrieved_answer = "?"
        
        # Score
        is_correct = score_answer(retrieved_answer, ground_truth)
        if is_correct:
            correct += 1
        
        # Show result
        if retrieved_answer == "?":
            print(f"→ NOT FOUND (GT: '{ground_truth}') ✗")
        else:
            print(f"→ '{retrieved_answer}' (GT: '{ground_truth}') [conf={confidence:.3f}] {'✓' if is_correct else '✗'}")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_answer": retrieved_answer,
            "category": category,
            "confidence": confidence,
            "correct": is_correct
        })
        
        time.sleep(REQUEST_DELAY)
    
    accuracy = correct / len(qa_pairs) if qa_pairs else 0
    
    # Accuracy by category
    category_stats = {}
    for r in results:
        cat = r['category']
        if cat not in category_stats:
            category_stats[cat] = {'correct': 0, 'total': 0}
        category_stats[cat]['total'] += 1
        if r['correct']:
            category_stats[cat]['correct'] += 1
    
    return {
        "user_id": user_id,
        "total_questions": len(qa_pairs),
        "correct": correct,
        "not_found": not_found,
        "accuracy": accuracy,
        "category_accuracy": {cat: stats['correct']/stats['total'] for cat, stats in category_stats.items()},
        "details": results
    }

def main():
    print(f"\n{'='*60}")
    print(f"LifeBench Evaluation (Slot Search)")
    print(f"{'='*60}")
    print(f"Server: {SERVER_URL}")
    print(f"DB: {DB_NAME}")
    print(f"QA Namespace Prefix: {QA_NS_PREFIX}")
    print(f"{'='*60}\n")
    
    # Load data
    with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    # Load document namespaces (just to know which users are ingested)
    doc_namespaces = load_doc_namespaces()
    print(f"Found {len(doc_namespaces)} ingested users: {list(doc_namespaces.keys())}")
    
    if not doc_namespaces:
        print("ERROR: No ingested users found! Run ingestion first.")
        print("If you've ingested users, check that lifebench_doc_namespaces.json exists.")
        return
    
    # Evaluate only users that have been ingested
    all_results = []
    for user_id in sorted(doc_namespaces.keys()):
        if user_id > len(users):
            print(f"User {user_id} not in data file")
            continue
        
        user_data = users[user_id - 1]
        qa_pairs = user_data.get('qa', [])
        
        print(f"\n{'='*50}")
        print(f"User {user_id}/10")
        print(f"QA Namespace: {QA_NS_PREFIX}{user_id}")
        print(f"QA Pairs: {len(qa_pairs)}")
        print(f"{'='*50}")
        
        result = evaluate_user(user_id, qa_pairs)
        all_results.append(result)
        
        print(f"\n  ✓ User {user_id} Results:")
        print(f"      Correct: {result['correct']}/{result['total_questions']} = {result['accuracy']:.1%}")
        print(f"      Not found: {result['not_found']}")
        for cat, acc in result['category_accuracy'].items():
            print(f"      Category {cat}: {acc:.1%}")
    
    # Overall summary
    if all_results:
        total_correct = sum(r['correct'] for r in all_results)
        total_questions = sum(r['total_questions'] for r in all_results)
        total_not_found = sum(r['not_found'] for r in all_results)
        overall_accuracy = total_correct / total_questions if total_questions else 0
        
        print(f"\n{'='*60}")
        print(f"LIFEBENCH OVERALL RESULTS")
        print(f"{'='*60}")
        print(f"Total Users: {len(all_results)}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        print(f"Total Not Found: {total_not_found}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_questions} = {overall_accuracy:.1%}")
        print(f"{'='*60}")
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f"lifebench_slot_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "method": "slot_search",
                "server_url": SERVER_URL,
                "db_name": DB_NAME,
                "users_evaluated": list(doc_namespaces.keys()),
                "total_questions": total_questions,
                "total_correct": total_correct,
                "total_not_found": total_not_found,
                "overall_accuracy": overall_accuracy,
                "results": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()