import glob
import json
import os

for f in glob.glob('C:/Users/Tech N More/.gemini/antigravity/brain/**/*.jsonl', recursive=True):
    try:
        with open(f, encoding='utf-8') as fp:
            for line in fp:
                if 'vpn' in line.lower() and 'gemini can u give me' not in line.lower():
                    data = json.loads(line)
                    content = data.get('content', '')
                    if content and len(content) > 10:
                        print(f"File: {f}")
                        print(content)
                        print("-" * 40)
    except Exception as e:
        pass
