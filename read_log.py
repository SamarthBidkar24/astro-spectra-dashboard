
try:
    with open('log.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except:
    with open('log.txt', 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read())
