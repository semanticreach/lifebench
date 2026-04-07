import json

with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
    users = json.load(f)

def count_context_cells(user_data):
    count = 0
    conv = user_data.get('conversation', {})
    for key, value in conv.items():
        if key.startswith('session_') and not key.endswith('_date_time'):
            if isinstance(value, list):
                count += len(value)
    return count

for i, user in enumerate(users[:3], 1):
    cell_count = count_context_cells(user)
    print(f"User {i}: {cell_count} conversation turns")
