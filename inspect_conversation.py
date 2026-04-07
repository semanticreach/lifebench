import json

# Use utf-8 encoding explicitly
with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

print("=== User 1 Conversation Structure ===\n")
conv = users[0].get('conversation', {})
print(f"Conversation keys: {list(conv.keys())}")

for key, value in conv.items():
    print(f"\n{key}:")
    print(f"  Type: {type(value)}")
    
    if isinstance(value, list):
        print(f"  Length: {len(value)}")
        if len(value) > 0:
            print(f"  First item type: {type(value[0])}")
            if isinstance(value[0], dict):
                print(f"  First item keys: {list(value[0].keys())[:5]}")
                # Show sample content
                sample = value[0]
                if 'content' in sample:
                    print(f"  Sample content: {sample['content'][:100]}...")
    elif isinstance(value, dict):
        print(f"  Keys: {list(value.keys())[:5]}")

print("\n=== User 1 Event Summary Structure ===")
events = users[0].get('event_summary', {})
for key, value in events.items():
    print(f"\n{key}:")
    if isinstance(value, list):
        print(f"  Length: {len(value)}")
        if len(value) > 0 and isinstance(value[0], dict):
            print(f"  Sample keys: {list(value[0].keys())}")
