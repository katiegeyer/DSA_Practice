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


def isValid(s):
    # Initialize a stack to keep track of opening brackets
    stack = []
    # Create a mapping from closing to opening brackets for easy lookup
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # Pop the top element if the stack is not empty
            # Otherwise, use a dummy value that won't match
            top_element = stack.pop() if stack else '#'

            # Check if the popped element is the mapping pair of the current character
            if mapping[char] != top_element:
                return False
        else:
            # If it's an opening bracket, push it onto the stack
            stack.append(char)

    # The stack should be empty at the end for a valid expression
    return not stack


# Example usage
print(isValid("()"))      # Output: True
print(isValid("()[]{}"))  # Output: True
print(isValid("(]"))      # Output: False
print(isValid("([)]"))    # Output: False
print(isValid("{[]}"))    # Output: True


def rotate(matrix):
    n = len(matrix)
    # Transpose the matrix
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Reverse each row
    for i in range(n):
        matrix[i].reverse()


# Example usage
matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rotate(matrix1)
print(matrix1)  # Output: [[7,4,1],[8,5,2],[9,6,3]]

matrix2 = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]
rotate(matrix2)
print(matrix2)  # Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]


def find_missing_number(arr):
    # Calculate the number of elements that should be in the array
    n = len(arr) + 1

    # Calculate the expected sum of numbers from 1 to n
    expected_sum = n * (arr[0] + arr[-1]) // 2

    # Calculate the actual sum of elements in the array
    actual_sum = sum(arr)

    # The difference is the missing number
    missing_number = expected_sum - actual_sum
    return missing_number


# Example usage
arr = [3, 7, 1, 2, 8, 4, 5]
arr.sort()  # Ensure the array is sorted for this method to work
print("The missing number is:", find_missing_number(arr))
