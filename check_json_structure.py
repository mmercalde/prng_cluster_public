import sys, json
sys.path.insert(0, '.')

with open('daily3.json') as f:
    data = json.load(f)

print(f"Type: {type(data)}")
print(f"Length: {len(data)}")
if isinstance(data, dict):
    print(f"Keys: {data.keys()}")
    if 'draws' in data:
        print(f"Draws length: {len(data['draws'])}")
        print(f"First draw: {data['draws'][0]}")
elif isinstance(data, list):
    print(f"First entry: {data[0]}")
    print(f"Keys: {data[0].keys()}")
