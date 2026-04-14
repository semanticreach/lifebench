# lifebench_ingest_all.py - Ingest ALL users 1-10 with retry logic
import json
import io
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import time
import sys
import tempfile
import argparse

load_dotenv()

SERVER_URL = os.getenv("HB_SERVER_URL", "http://18.220.128.24:8000")
API_KEY = os.getenv("HB_API_KEY", "")
DB_NAME = "fractal_db"
QA_NS_PREFIX = "lifebench_qa_"
EMBED_DIM = 384
VECTOR_COL = "precomputed_vectors"  

# QA Schema (matches the QA structure)
LIFEBENCH_QA_SCHEMA = json.dumps({
    "molecule": "Row",
    "primary_key": "qa_id",
    "fields": {
        "question": {"encoding": "semantic"},
        "answer": {"encoding": "exact"},
        "evidence": {"encoding": "exact"},
        "category": {"encoding": "exact"},
        "user_id": {"encoding": "exact"},
        "sample_id": {"encoding": "exact"},
    },
    "field_order": ["qa_id", "user_id", "category", "question", "answer", "evidence", "sample_id"]
})

# Row fields (order must match schema)
LIFEBENCH_ROW_FIELDS = [
    "qa_id", "user_id", "category", "question", "answer", "evidence", "sample_id", VECTOR_COL
]

def load_existing_users() -> set:
    """Load already processed users from namespace file"""
    ns_file = "lifebench_doc_namespaces.json"
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
            return set(int(k) for k in namespaces.keys())
    return set()

def save_doc_namespace(user_id: int, namespace: str):
    """Save document namespace mapping for later evaluation"""
    ns_file = "lifebench_doc_namespaces.json"
    namespaces = {}
    
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
    
    namespaces[str(user_id)] = namespace
    with open(ns_file, 'w') as f:
        json.dump(namespaces, f, indent=2)
    print(f"    Saved namespace for user {user_id} -> {namespace}")

def save_qa_namespace(user_id: int, namespace: str):
    """Save QA namespace mapping (optional, for debugging)"""
    ns_file = "lifebench_qa_namespaces.json"
    namespaces = {}
    
    if os.path.exists(ns_file):
        with open(ns_file, 'r') as f:
            namespaces = json.load(f)
    
    namespaces[str(user_id)] = namespace
    with open(ns_file, 'w') as f:
        json.dump(namespaces, f, indent=2)

def flatten_user_context(user_data: dict) -> list:
    """Convert session-based structure to flat list for HyperBinder"""
    records = []
    
    # Add conversation turns - look for ALL session_X keys
    conv = user_data.get('conversation', {})
    for key, value in conv.items():
        # Look for session_XXX keys (not date_time keys)
        if key.startswith('session_') and not key.endswith('_date_time'):
            if isinstance(value, list):
                for turn in value:
                    if isinstance(turn, dict):
                        # Get speaker and text (field names are 'speaker' and 'text')
                        speaker = turn.get('speaker', 'unknown')
                        text = turn.get('text', '')  # Note: 'text' not 'content'
                        session_num = key.replace('session_', '')
                        if text:
                            records.append({
                                "value": f"[Session {session_num}] {speaker}: {text}",
                                "session": session_num,
                                "type": "conversation"
                            })
    
    # Add event summaries - look for ALL events_session_X keys
    events = user_data.get('event_summary', {})
    for key, value in events.items():
        if key.startswith('events_session_'):
            if isinstance(value, list):
                for event in value:
                    if isinstance(event, dict):
                        summary = event.get('summary', '')
                        session_num = key.replace('events_session_', '')
                        if summary:
                            records.append({
                                "value": f"[Session {session_num}] EVENT: {summary}",
                                "session": session_num,
                                "type": "event"
                            })
    
    # Add observations - look for ALL session_X_observation keys
    obs = user_data.get('observation', {})
    for key, value in obs.items():
        if key.endswith('_observation'):
            if value:  # observation is likely a string, not a list
                session_num = key.replace('_observation', '').replace('session_', '')
                records.append({
                    "value": f"[Session {session_num}] OBSERVATION: {value}",
                    "session": session_num,
                    "type": "observation"
                })
    
    # Add session summaries - look for ALL session_X_summary keys
    summaries = user_data.get('session_summary', {})
    for key, value in summaries.items():
        if key.endswith('_summary'):
            if value:
                session_num = key.replace('_summary', '').replace('session_', '')
                records.append({
                    "value": f"[Session {session_num}] SUMMARY: {value}",
                    "session": session_num,
                    "type": "summary"
                })
    
    print(f"    Extracted {len(records)} context cells")
    return records

def upload_user_document_as_txt(user_id: int, context_records: list, timeout: int = 7200, max_retries: int = 5) -> str:
    """
    Upload as TXT file with extended timeout and retry logic
    """
    if not context_records:
        print(f"    WARNING: No context records for user {user_id}")
        return None
    
    # Combine all context into one text
    combined_text = "\n\n".join([record["value"] for record in context_records])
    text_size_mb = len(combined_text.encode('utf-8')) / (1024 * 1024)
    print(f"    Combined text size: {text_size_mb:.2f} MB")
    
    # Create a temporary TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(combined_text)
        tmp_path = tmp.name
    
    for attempt in range(max_retries):
        try:
            with open(tmp_path, 'rb') as f:
                print(f"    Uploading (attempt {attempt+1}/{max_retries}) with {timeout}s timeout ({timeout/60:.0f} minutes)...")
                resp = requests.post(
                    f"{SERVER_URL}/upload_document/",
                    headers={"X-API-Key": API_KEY} if API_KEY else {},
                    files={"file": (f"lifebench_user_{user_id}.txt", f, "text/plain")},
                    data={"dim": EMBED_DIM, "seed": 42, "depth": 3},
                    timeout=timeout,
                )
            
            if resp.ok:
                result = resp.json()
                print(f"    Uploaded: {result['namespace']} ({result['total_cells']} cells, vector_source={result['vector_source']})")
                return result["namespace"]
            else:
                print(f"    Upload failed (attempt {attempt+1}): {resp.status_code} - {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)  # Progressive wait: 30, 60, 90, 120s
                    print(f"    Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.Timeout:
            print(f"    Upload timed out (attempt {attempt+1}) after {timeout}s")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"    Upload error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        finally:
            # Don't unlink until after all retries or success
            if attempt == max_retries - 1:
                os.unlink(tmp_path)
    
    # Clean up if all retries failed
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    return None

def ingest_qa_rows(user_id: int, qa_pairs: list, cache: dict, timeout: int = 7200, max_retries: int = 3) -> str:
    """Ingest QA rows with precomputed vectors and retry logic"""
    namespace = f"{QA_NS_PREFIX}{user_id}"
    rows = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa.get('question', '')
        vec = cache.get(question, [0.0] * EMBED_DIM)
        
        rows.append({
            "qa_id": f"user_{user_id}_qa_{i}",
            "user_id": str(user_id),
            "category": str(qa.get('category', '')),
            "question": question,
            "answer": qa.get('answer', ''),
            "evidence": json.dumps(qa.get('evidence', [])),
            "sample_id": str(qa.get('sample_id', i)),
            VECTOR_COL: json.dumps(vec)
        })
    
    df = pd.DataFrame(rows, columns=LIFEBENCH_ROW_FIELDS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    
    for attempt in range(max_retries):
        try:
            print(f"    Ingesting (attempt {attempt+1}/{max_retries}) with {timeout}s timeout ({timeout/60:.0f} minutes)...")
            resp = requests.post(
                f"{SERVER_URL}/build_ingest_data/",
                headers={"X-API-Key": API_KEY} if API_KEY else {},
                files={"file": (f"lifebench_qa_user_{user_id}.csv", buf, "text/csv")},
                data={
                    "dim": EMBED_DIM,
                    "seed": 42,
                    "depth": 3,
                    "db_name": DB_NAME,
                    "namespace": namespace,
                    "template_schema": LIFEBENCH_QA_SCHEMA,
                    "vector_col": VECTOR_COL,
                },
                timeout=timeout,
            )
            
            if resp.ok:
                result = resp.json()
                print(f"    Ingested {result.get('rows_added', 0)} QA rows (vector_source={result.get('vector_source', 'unknown')})")
                return namespace
            else:
                print(f"    Ingest failed (attempt {attempt+1}): {resp.status_code} - {resp.text[:200]}")
                if attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    print(f"    Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.Timeout:
            print(f"    Ingest timed out (attempt {attempt+1}) after {timeout}s")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"    Ingest error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"    Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    return None

def precompute_embeddings(users: list) -> dict:
    """Precompute embeddings for all questions"""
    from sentence_transformers import SentenceTransformer
    
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_questions = []
    for user in users:
        for qa in user.get('qa', []):
            if question := qa.get('question'):
                all_questions.append(question)
    
    unique_questions = list(set(all_questions))
    print(f"Precomputing {len(unique_questions)} unique questions...")
    
    cache = {}
    batch_size = 128
    for i in range(0, len(unique_questions), batch_size):
        batch = unique_questions[i:i+batch_size]
        vectors = model.encode(batch, show_progress_bar=True)
        for text, vec in zip(batch, vectors):
            cache[text] = vec.tolist()
        print(f"  {min(i+batch_size, len(unique_questions))}/{len(unique_questions)}")
    
    return cache

def wipe_namespace(namespace: str, db_name: str = DB_NAME, timeout: int = 30) -> bool:
    """Wipe a specific namespace from the database"""
    print(f"  Wiping namespace '{namespace}'...")
    try:
        resp = requests.delete(
            f"{SERVER_URL}/db/{db_name}/namespace/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            timeout=timeout,
        )
        if resp.status_code in (200, 404):
            print(f"  ✓ Wiped (status {resp.status_code})")
            return True
        else:
            print(f"  ⚠️  {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"  ✗ Wipe error: {e}")
        return False

def wipe_all_lifebench_data(users: list = None):
    """Wipe all LifeBench namespaces for specified users"""
    if users is None:
        users = range(1, 11)  # Users 1-10
    
    print(f"\n{'='*60}")
    print(f"Wiping LifeBench Data for Users {min(users)}-{max(users)}")
    print(f"{'='*60}")
    
    # Wipe document namespaces
    print("\n📄 Wiping document namespaces:")
    for user_id in users:
        doc_ns = f"lifebench_doc_{user_id}"
        wipe_namespace(doc_ns)
    
    # Wipe QA namespaces
    print("\n❓ Wiping QA namespaces:")
    for user_id in users:
        qa_ns = f"{QA_NS_PREFIX}{user_id}"
        wipe_namespace(qa_ns)
    
    # Also clear the namespace tracking files
    ns_files = ["lifebench_doc_namespaces.json", "lifebench_qa_namespaces.json"]
    print("\n🗑️  Clearing namespace tracking files:")
    for ns_file in ns_files:
        if os.path.exists(ns_file):
            os.remove(ns_file)
            print(f"  ✓ Removed {ns_file}")
        else:
            print(f"  - {ns_file} not found")
    
    print(f"\n{'='*60}")
    print(f"✅ Wipe complete!")
    print(f"{'='*60}")

def check_server_health():
    """Check if server is responding"""
    try:
        resp = requests.get(f"{SERVER_URL}/", timeout=5)
        return resp.status_code == 200
    except:
        return False

def process_user(user_id: int, user_data: dict, cache: dict, timeout: int = 7200):
    """Process single user: upload context + ingest QA with retry logic"""
    print(f"\n{'='*50}")
    print(f"Processing User {user_id}/10")
    print(f"{'='*50}")
    
    # Check server health before starting
    if not check_server_health():
        print(f"  ⚠️  Server not responding! Waiting 60 seconds...")
        time.sleep(60)
        if not check_server_health():
            print(f"  ❌ Server still down. Skipping user {user_id}")
            return None, None
    
    qa_pairs = user_data.get('qa', [])
    print(f"  QA pairs: {len(qa_pairs)}")
    
    # Flatten context
    context = flatten_user_context(user_data)
    print(f"  Context cells: {len(context)}")
    

    doc_namespace = upload_user_document_as_txt(user_id, context, timeout=timeout, max_retries=5)
    
    if not doc_namespace:
        print(f"  ❌ Upload failed after all retries. Skipping user {user_id}")
        return None, None

    save_doc_namespace(user_id, doc_namespace)
    

    qa_namespace = ingest_qa_rows(user_id, qa_pairs, cache, timeout=timeout, max_retries=3)
    
    if not qa_namespace:
        print(f"  ⚠️  QA ingest failed after retries, but document upload succeeded for user {user_id}")
        return doc_namespace, None
    

    if qa_namespace:
        save_qa_namespace(user_id, qa_namespace)
    
    time.sleep(2)  # Small delay between users
    return doc_namespace, qa_namespace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest ALL LifeBench users 1-10')
    parser.add_argument('--wipe', action='store_true', help='Wipe existing data before ingestion')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout in seconds for uploads/ingests (default: 7200 = 2 hours)')
    parser.add_argument('--start-user', type=int, default=1, help='Start from specific user ID (1-10)')
    parser.add_argument('--end-user', type=int, default=10, help='End at specific user ID (1-10)')
    parser.add_argument('--retries', type=int, default=5, help='Max retries for uploads (default: 5)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"LifeBench Ingest - ALL Users (1-10)")
    print(f"{'='*60}")
    print(f"Server: {SERVER_URL}")
    print(f"DB: {DB_NAME}")
    print(f"API Key: {'Set' if API_KEY else 'Not set (may not be required)'}")
    print(f"Timeout: {args.timeout}s ({args.timeout/60:.0f} minutes)")
    print(f"Max Retries: {args.retries}")
    print(f"{'='*60}\n")
    
    # Load data
    with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print(f"Total users in dataset: {len(users)}")
    
    # Wipe if requested
    if args.wipe:
        wipe_all_lifebench_data(range(args.start_user, args.end_user + 1))
        print("\n" + "="*60)
        print("Proceeding with fresh ingestion...")
        print("="*60 + "\n")
        time.sleep(2)
    
    # Load already processed users (if not wiping)
    existing_users = load_existing_users()
    if existing_users and not args.wipe:
        print(f"Already processed users: {sorted(existing_users)}")
    
    # Determine users to process (all users 1-10, optionally filtered)
    users_to_process = [i for i in range(args.start_user, args.end_user + 1)]
    
    if not args.wipe:
        # Only process users that haven't been processed yet
        users_to_process = [i for i in users_to_process if i not in existing_users]
    
    if not users_to_process:
        print("\n✅ All requested users already processed!")
        sys.exit(0)
    
    print(f"\nUsers to process: {users_to_process}")
    print(f"This will process {len(users_to_process)} users with {args.timeout}s timeout each")
    print(f"Estimated total time: ~{len(users_to_process) * args.timeout / 3600:.1f} hours")
    print("\nStarting ingestion...")
    
    # Precompute embeddings for all questions
    cache = precompute_embeddings(users)
    
    # Process users
    successful = 0
    for idx, user_id in enumerate(users_to_process, 1):
        user_data = users[user_id-1]  # 0-indexed list
        try:
            print(f"\n{'🔄'*30}")
            print(f"Starting User {user_id} at {time.strftime('%H:%M:%S')}")
            print(f"{'🔄'*30}")
            
            doc_ns, qa_ns = process_user(user_id, user_data, cache, timeout=args.timeout)
            
            if doc_ns and qa_ns:
                successful += 1
                print(f"\n✅ User {user_id} completed successfully at {time.strftime('%H:%M:%S')}")
            elif doc_ns and not qa_ns:
                print(f"\n⚠️  User {user_id} partially completed (doc OK, QA failed) at {time.strftime('%H:%M:%S')}")
            else:
                print(f"\n❌ User {user_id} failed completely at {time.strftime('%H:%M:%S')}")
                
        except Exception as e:
            print(f"\n💥 ERROR processing user {user_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # Progress report
        print(f"\n📊 Progress: {successful}/{idx} successful so far")
        
        # If server seems unstable, add extra delay
        if user_id < len(users_to_process):
            print(f"  Waiting 5 seconds before next user...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"✅ LifeBench ingestion complete!")
    print(f"Successfully processed: {successful}/{len(users_to_process)} users")
    print(f"Total users now in database: {len(load_existing_users())}/10")
    print(f"{'='*60}")