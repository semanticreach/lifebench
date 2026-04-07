import json

with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

first_user = users[0]
print("=== All top-level fields in user data ===")
for key in first_user.keys():
    value = first_user[key]
    print(f"  {key}: {type(value).__name__}", end="")
    if isinstance(value, list):
        print(f" (length: {len(value)})")
    else:
        print()

# Check what's in each field
print("\n=== Field details ===")
for key in first_user.keys():
    value = first_user[key]
    print(f"\n{key}:")
    if isinstance(value, list) and len(value) > 0:
        print(f"  First item type: {type(value[0]).__name__}")
        if isinstance(value[0], dict):
            print(f"  First item keys: {list(value[0].keys())[:5]}")
        elif isinstance(value[0], str):
            print(f"  First item: {value[0][:100]}")
    elif isinstance(value, dict):
        print(f"  Keys: {list(value.keys())[:5]}")
