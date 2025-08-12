import re

# Read the file
with open('src/core/shared/unified_model_architecture.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all instances of metrics with precision and recall
old_pattern = r"metrics=\['accuracy', 'precision', 'recall'\]"
new_pattern = "metrics=['accuracy']"

content = re.sub(old_pattern, new_pattern, content)

# Write back
with open('src/core/shared/unified_model_architecture.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed metrics issues in unified_model_architecture.py")
print("Removed 'precision' and 'recall' metrics to avoid compatibility issues") 