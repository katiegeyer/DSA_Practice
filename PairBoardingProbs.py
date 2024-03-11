# reverse integer
def reverseInteger(x):
    INT_MAX, INT_MIN = 2**31 - 1, -2**31
    rev = 0

    while x != 0:
        # Handle negative numbers by working with positive equivalents
        if x < 0:
            digit = -1 * (x % -10)
            x = -1 * (x // -10)
        else:
            digit = x % 10
            x //= 10

        # Check for overflow and return 0 if it would occur
        if rev > INT_MAX // 10 or (rev == INT_MAX // 10 and digit > INT_MAX % 10):
            return 0
        if rev < INT_MIN // 10 or (rev == INT_MIN // 10 and digit < INT_MIN % 10):
            return 0

        rev = rev * 10 + digit

    return rev


# Example usages
print(reverseInteger(123))  # Output: 321
print(reverseInteger(-123))  # Output: -321
print(reverseInteger(120))  # Output: 21

# isvalid pairboarding solution


def isValid(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in mapping:
            # Pop the topmost element if it matches the mapping, else push a dummy value to ensure mismatch
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)

    # If the stack is empty, all brackets are properly closed
    return not stack


# Example usages
print(isValid("()"))     # Output: true
print(isValid("()[]{}"))  # Output: true
print(isValid("(]"))     # Output: false

# commented tech interview pairboarding twoSum


def twoSum(nums, target):
    # Create a dictionary to store the value and its index
    hash_table = {}

    # Enumerate over the list to get both the index and the value
    for index, value in enumerate(nums):
        # Calculate the difference needed to reach the target
        difference = target - value

        # If the difference is already in the hash table, we have our solution
        if difference in hash_table:
            return [hash_table[difference], index]

        # Otherwise, add the value and its index to the hash table
        hash_table[value] = index

    # If no solution is found, just return an empty list (or raise an exception)
    return []


# Example
print(twoSum([2, 7, 11, 15], 9))  # Output: [0, 1]
