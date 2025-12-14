# Debug Analysis: Why Predictions Are "3"

## Problem Identified

From JSON results analysis:
- **MMLU**: All 10 predictions = "3"
- **HellaSwag**: 9/10 predictions = "3", 1 prediction = "1975"
- **Expected**: Predictions should be "A", "B", "C", or "D"

## Root Cause

The prediction "3" is coming from the number extraction in the fallback path:
```python
num_match = re.search(r'\b(\d+)\b', conclusion)
if num_match:
    return num_match.group(1)  # Returns "3"
```

This happens when:
1. `choices` list is empty (not extracted properly)
2. Falls through to reasoning conclusion path
3. Extracts first number found ("3") from conclusion text

## Why Choices Might Be Empty

1. **MMLU**: `item['choices']` might be in different format
2. **HellaSwag**: Uses `item['endings']` instead
3. **Extraction logic**: May not handle all data formats

## Fix Applied

1. ✅ Improved choice extraction (handles dict format)
2. ✅ Extract choices from question text if not in item
3. ✅ Only extract numbers for non-multiple-choice questions
4. ✅ Always return letter (A-D) for multiple-choice

## Testing

After fixes, predictions should be:
- "A", "B", "C", or "D" for multiple-choice
- Numbers only for math questions (GSM8K)
- Text for open-ended questions

## Expected Accuracy After Fix

- **Random baseline**: ~25% (4 choices)
- **With similarity matching**: ~30-40%
- **With LLM integration**: ~60-80%
- **With semantic understanding**: ~80-90%

