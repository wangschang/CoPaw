import re
with open("src/copaw/config/config.py", "r") as f:
    text = f.read()

print("Found AgentsConfig:", "AgentsConfig" in text)
