# lifebench_ingest.py
import json
import io
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import time
import sys
import tempfile

load_dotenv()

SERVER_URL = os.getenv("HB_SERVER_URL", "http://localhost:8000")
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

def upload_user_document_as_txt(user_id: int, context_records: list, timeout: int = 3600) -> str:
    """
    Upload as TXT file with configurable timeout (default 30 minutes)
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
    
    try:
        with open(tmp_path, 'rb') as f:
            print(f"    Uploading with {timeout}s timeout...")
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
            print(f"    Upload failed: {resp.status_code} - {resp.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"    Upload timed out after {timeout}s - try increasing timeout further")
        return None
    except Exception as e:
        print(f"    Upload error: {e}")
        return None
    finally:
        os.unlink(tmp_path)

def ingest_qa_rows(user_id: int, qa_pairs: list, cache: dict, timeout: int = 3600) -> str:
    """Ingest QA rows with precomputed vectors"""
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
    
    try:
        print(f"    Ingesting {len(rows)} QA rows with {timeout}s timeout...")
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
            print(f"    Ingest failed: {resp.status_code} - {resp.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"    Ingest timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"    Ingest error: {e}")
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
        vectors = model.encode(batch, show_progress_bar=False)
        for text, vec in zip(batch, vectors):
            cache[text] = vec.tolist()
        print(f"  {min(i+batch_size, len(unique_questions))}/{len(unique_questions)}")
    
    return cache

def process_user(user_id: int, user_data: dict, cache: dict):
    """Process single user: upload context + ingest QA"""
    print(f"\n{'='*50}")
    print(f"Processing User {user_id}/10")
    print(f"{'='*50}")
    
    qa_pairs = user_data.get('qa', [])
    print(f"  QA pairs: {len(qa_pairs)}")
    
    # Flatten context
    context = flatten_user_context(user_data)
    print(f"  Context cells: {len(context)}")
    
    # Upload as TXT with 30 minute timeout
    doc_namespace = upload_user_document_as_txt(user_id, context, timeout=1800)
    
    if not doc_namespace:
        print(f"  Skipping QA ingest due to upload failure")
        return None, None
    
    # SAVE the document namespace for evaluation!
    save_doc_namespace(user_id, doc_namespace)
    
    # Ingest QA rows with 10 minute timeout
    qa_namespace = ingest_qa_rows(user_id, qa_pairs, cache, timeout=600)
    
    # Save QA namespace (optional)
    if qa_namespace:
        save_qa_namespace(user_id, qa_namespace)
    
    time.sleep(0.5)  # Small delay between users
    return doc_namespace, qa_namespace

def delete_namespace(namespace: str) -> bool:
    """Delete a namespace from HyperBinder"""
    try:
        resp = requests.delete(
            f"{SERVER_URL}/db/{DB_NAME}/namespace/{namespace}",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            timeout=30,
        )
        return resp.ok
    except Exception as e:
        print(f"    Delete error: {e}")
        return False

def wipe_all_lifebench_namespaces():
    """Delete all existing LifeBench namespaces"""
    print("\n" + "="*60)
    print("WIPING existing LifeBench namespaces")
    print("="*60)
    
    # Get all namespaces
    try:
        resp = requests.get(
            f"{SERVER_URL}/db/{DB_NAME}/namespaces",
            headers={"X-API-Key": API_KEY} if API_KEY else {},
        )
        
        if resp.ok:
            all_namespaces = resp.json().get('namespaces', [])
            lifebench_namespaces = [ns for ns in all_namespaces if 'lifebench' in ns or ns.startswith('document_upload_')]
            
            print(f"Found {len(lifebench_namespaces)} LifeBench-related namespaces")
            
            for ns in lifebench_namespaces:
                if delete_namespace(ns):
                    print(f"  Deleted: {ns}")
                else:
                    print(f"  Failed to delete: {ns}")
            
            # Also delete the namespaces files
            if os.path.exists("lifebench_doc_namespaces.json"):
                os.remove("lifebench_doc_namespaces.json")
                print("  Deleted: lifebench_doc_namespaces.json")
            if os.path.exists("lifebench_qa_namespaces.json"):
                os.remove("lifebench_qa_namespaces.json")
                print("  Deleted: lifebench_qa_namespaces.json")
        else:
            print(f"  Failed to list namespaces: {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Check for --wipe flag
    wipe = "--wipe" in sys.argv
    
    print(f"\n{'='*60}")
    print(f"LifeBench Ingest to HyperBinder")
    print(f"{'='*60}")
    print(f"Server: {SERVER_URL}")
    print(f"DB: {DB_NAME}")
    print(f"API Key: {'Set' if API_KEY else 'Not set (may not be required)'}")
    print(f"Wipe existing: {wipe}")
    print(f"{'='*60}\n")
    
    if wipe:
        wipe_all_lifebench_namespaces()
        print("\n✅ Wipe complete. Proceeding with fresh ingestion...\n")
    
    # Load data
    with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print(f"Loaded {len(users)} users")
    
    # Precompute embeddings for all questions
    cache = precompute_embeddings(users)
    
    # Process each user
    successful = 0
    for i, user_data in enumerate(users, 1):
        try:
            doc_ns, qa_ns = process_user(i, user_data, cache)
            if doc_ns and qa_ns:
                successful += 1
        except Exception as e:
            print(f"  ERROR processing user {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ LifeBench ingestion complete!")
    print(f"Successfully processed: {successful}/{len(users)} users")
    print(f"Namespaces saved to: lifebench_doc_namespaces.json")
    print(f"{'='*60}")