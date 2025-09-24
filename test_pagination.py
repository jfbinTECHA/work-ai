def test_pagination(page_data):
    if page_data['pagination'] == True:
        # ... existing logic ...
        print("Processing pagination")
        return "processed"
    else:
        return None

# Test cases
print("Test 1: pagination = True")
result1 = test_pagination({'pagination': True})
print(f"Result: {result1}")

print("\nTest 2: pagination = False")
result2 = test_pagination({'pagination': False})
print(f"Result: {result2}")

print("\nTest 3: pagination key missing")
try:
    result3 = test_pagination({})
    print(f"Result: {result3}")
except KeyError as e:
    print(f"KeyError: {e}")