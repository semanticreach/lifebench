# query_users_5_10.py
import json
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("HB_SERVER_URL", "http://18.220.128.24:8000")
API_KEY = os.getenv("HB_API_KEY", "")
DB_NAME = "fractal_db"
QA_NS_PREFIX = "lifebench_qa_"
REQUEST_DELAY = 0.1

def search_slot_answer(question: str, user_id: int) -> dict:
    """Search the QA namespace for the stored answer"""
    namespace = f"{QA_NS_PREFIX}{user_id}"
    
    try:
        resp = requests.post(
            f"{SERVER_URL}/compose/search_slots/{DB_NAME}/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            json={
                "slot_queries": {
                    "question": {"query": question, "weight": 1.0},
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
                    "confidence": results[0].get("score", 0)
                }
        return {"answer": "", "confidence": 0}
    except Exception as e:
        print(f"      Error: {e}")
        return {"answer": "", "confidence": 0}

def evaluate_user(user_id: int, qa_pairs: list) -> dict:
    """Evaluate a single user"""
    print(f"\n  Evaluating {len(qa_pairs)} questions...")
    
    results = []
    correct = 0
    
    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        category = qa.get('category', -1)
        
        result = search_slot_answer(question, user_id)
        retrieved = result.get("answer", "")
        
        is_correct = (retrieved == ground_truth)
        if is_correct:
            correct += 1
        
        if i % 50 == 0:
            print(f"    Progress: {i}/{len(qa_pairs)}")
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "retrieved": retrieved,
            "category": category,
            "correct": is_correct
        })
        
        time.sleep(REQUEST_DELAY)
    
    accuracy = correct / len(qa_pairs) if qa_pairs else 0
    
    # Category stats
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
        "accuracy": accuracy,
        "category_accuracy": {cat: stats['correct']/stats['total'] for cat, stats in category_stats.items()},
        "details": results
    }

def main():
    print(f"\n{'='*60}")
    print(f"LifeBench Evaluation - Users 5-10")
    print(f"{'='*60}")
    print(f"Server: {SERVER_URL}")
    print(f"DB: {DB_NAME}")
    print(f"{'='*60}\n")
    
    # Load data
    with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print(f"Loaded {len(users)} total users")
    
    # Evaluate only users 5-10
    all_results = []
    for user_id in range(5, 11):
        user_data = users[user_id - 1]
        qa_pairs = user_data.get('qa', [])
        
        print(f"\n{'='*50}")
        print(f"User {user_id}/10")
        print(f"QA Namespace: {QA_NS_PREFIX}{user_id}")
        print(f"QA Pairs: {len(qa_pairs)}")
        print(f"{'='*50}")
        
        result = evaluate_user(user_id, qa_pairs)
        all_results.append(result)
        
        print(f"\n  ✓ User {user_id}: {result['correct']}/{result['total_questions']} = {result['accuracy']:.1%}")
        for cat, acc in result['category_accuracy'].items():
            print(f"      Category {cat}: {acc:.1%}")
    
    # Summary for users 5-10
    if all_results:
        total_correct = sum(r['correct'] for r in all_results)
        total_questions = sum(r['total_questions'] for r in all_results)
        overall_accuracy = total_correct / total_questions if total_questions else 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS FOR USERS 5-10")
        print(f"{'='*60}")
        print(f"Total Users: {len(all_results)}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        print(f"OVERALL ACCURACY: {total_correct}/{total_questions} = {overall_accuracy:.1%}")
        print(f"{'='*60}")
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f"lifebench_results_users_5_10_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "users": [5, 6, 7, 8, 9, 10],
                "total_questions": total_questions,
                "total_correct": total_correct,
                "overall_accuracy": overall_accuracy,
                "results": all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()