import json

with open('life_bench_data/locomo_format/our_en.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total users: {len(data)}')
print()
total_qa = 0
for i, user in enumerate(data, 1):
    qa_count = len(user.get('qa', []))
    total_qa += qa_count
    print(f'User {i}: {qa_count} QA pairs')

print()
print(f'Total QA pairs across all users: {total_qa}')
