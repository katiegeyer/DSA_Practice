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
