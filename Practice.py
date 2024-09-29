# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
from collections import Counter
from collections import OrderedDict
import pandas as pd
import schedule
import requests
import cv2
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np
import bisect
import collections
import heapq
from PIL import Image
import time
import random
import math


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # Definition for singly-linked list.


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        # Initialize current node to dummy head of the returning list.
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0

        # Loop through lists l1 and l2 until you reach both ends.
        while l1 is not None or l2 is not None:
            # At the start of each iteration, should add carry from last iteration
            sum = carry
            if l1 is not None:
                sum += l1.val
                l1 = l1.next
            if l2 is not None:
                sum += l2.val
                l2 = l2.next

            # Update carry for next calulation of sum
            carry = sum // 10

            # Create a new node with the digit value of (sum mod 10) and set it to the current node's next,
            # then advance current node to next.
            current.next = ListNode(sum % 10)
            current = current.next

        # Check if carry = 1, if so append a new node with digit 1 to the returning list.
        if carry > 0:
            current.next = ListNode(carry)

        # Return the dummy head's next node.
        return dummy_head.next


def is_palindrome(x):
    # Negative numbers are not palindromes
    if x < 0:
        return False

    # Copy the original number to a variable
    original_number = x
    reversed_number = 0

    # Reverse the number
    while x > 0:
        digit = x % 10  # Get the last digit
        reversed_number = (reversed_number * 10) + digit  # Append the digit
        x = x // 10  # Remove the last digit

    # Check if the original number and the reversed number are the same
    return original_number == reversed_number


# Test the function with the provided examples
test_cases = [121, -121, 10]
results = {x: is_palindrome(x) for x in test_cases}
results


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        char_map = {}
        left = 0
        max_length = 0

        for right in range(len(s)):
            if s[right] in char_map:
                # Move the left pointer to the right of the same character
                left = max(left, char_map[s[right]] + 1)
            # Update the last seen index of the character
            char_map[s[right]] = right
            # Update the max length
            max_length = max(max_length, right - left + 1)

        return max_length


# Example
solution = Solution()
example_string = "abcabcbb"
print(solution.lengthOfLongestSubstring(example_string))  # Expected output: 3


class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman_values = {'I': 1, 'V': 5, 'X': 10,
                        'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0

        for char in reversed(s):
            value = roman_values[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value

        return total


# Example
solution = Solution()
example_roman = "XXVII"
print(solution.romanToInt(example_roman))  # Expected output: 27


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""

        # Start with the first string as the initial prefix
        prefix = strs[0]

        # Compare the prefix with each string in the array
        for string in strs[1:]:
            # Reduce the prefix length until it matches the beginning of the string
            while string[:len(prefix)] != prefix:
                prefix = prefix[:-1]
                # If the prefix is empty, return an empty string
                if not prefix:
                    return ""

        return prefix


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        # Initialize a dummy node
        dummy = ListNode()
        current = dummy

        # Traverse through both lists
        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        # Append the remaining elements of list1 or list2
        if list1:
            current.next = list1
        elif list2:
            current.next = list2

        # Return the head of the merged list
        return dummy.next


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ""

        # Sort the input list of strings
        strs.sort()

        # Take the first and last strings (after sorting)
        first_str = strs[0]
        last_str = strs[-1]

        common_prefix = []
        for i in range(len(first_str)):
            # Compare characters at the same position in the first and last strings
            if first_str[i] == last_str[i]:
                common_prefix.append(first_str[i])
            else:
                break  # Stop at the first character mismatch

        return ''.join(common_prefix)


# Example usage:
solution = Solution()
strs1 = ["flower", "flow", "flight"]
print(solution.longestCommonPrefix(strs1))  # Output: "fl"

strs2 = ["dog", "racecar", "car"]
print(solution.longestCommonPrefix(strs2))


class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def generate(p, left, right, parens=[]):
            if left:
                generate(p + '(', left-1, right)
            if right > left:
                generate(p + ')', left, right-1)
            if not right:
                parens += p,
            return parens
        return generate('', n, n)


# Example usage:
solution = Solution()
n1 = 3
print(solution.generateParenthesis(n1))
# Output: ["((()))","(()())","(())()","()(())","()()()"]

n2 = 1
print(solution.generateParenthesis(n2))
# Output: ["()"]


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        char_map = {}
        left = 0
        right = 0
        ans = 0
        n = len(s)

        while right < n:
            if s[right] in char_map:
                left = max(left, char_map[s[right]] + 1)

            char_map[s[right]] = right
            ans = max(ans, right - left + 1)
            right += 1

        return ans


# Create a Solution object
solution = Solution()

# Ex 1
print(solution.lengthOfLongestSubstring("abcabcbb"))  # Output: 3

# Ex 2
print(solution.lengthOfLongestSubstring("bbbbb"))     # Output: 1

# Ex 3
print(solution.lengthOfLongestSubstring("pwwkew"))    # Output: 3


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        charIndexMap = {}
        start = 0
        maxLength = 0

        for end in range(len(s)):
            if s[end] in charIndexMap:
                # Move the start pointer. Avoid moving backward.
                start = max(start, charIndexMap[s[end]] + 1)
            # Update the last seen index of the character.
            charIndexMap[s[end]] = end
            # Calculate the length of the current window.
            maxLength = max(maxLength, end - start + 1)

        return maxLength


def intToRoman(num):
    """
    :type num: int
    :rtype: str
    """
    # Map of roman numerals
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syms = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]

    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


# Test the function with the given examples
example1 = intToRoman(3)  # Should return "III"
example2 = intToRoman(58)  # Should return "LVIII"
example3 = intToRoman(1994)  # Should return "MCMXCIV"

example1, example2, example3


class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        max_area = 0

        while left < right:
            # Calculate the area with the current pair of lines
            current_area = (right - left) * min(height[left], height[right])
            max_area = max(max_area, current_area)

            # Move the pointer pointing to the shorter line
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area


def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    nums.sort()
    res = []

    for i in range(len(nums) - 2):
        # Avoid duplicates for the first number
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                # Avoid duplicates for the second and third numbers
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1

    return res


# Test the function with the given examples
# Should return [[-1,-1,2],[-1,0,1]]
example1 = threeSum([-1, 0, 1, 2, -1, -4])
example2 = threeSum([0, 1, 1])  # Should return []
example3 = threeSum([0, 0, 0])  # Should return [[0,0,0]]

example1, example2, example3


# Given values
F = 80  # magnitude of the force in lb
alpha = math.radians(60)  # alpha angle in radians
beta = math.radians(45)  # beta angle in radians

# Calculating the cosine of gamma using the trigonometric identity
cos_gamma_squared = 1 - (math.cos(alpha)**2 + math.cos(beta)**2)
gamma = math.acos(math.sqrt(cos_gamma_squared))  # gamma angle in radians

# Now calculating the components of the force F
F_x = F * math.cos(alpha)
F_y = F * math.cos(beta)
F_z = F * math.cos(gamma)

# also converting gamma back to degrees for reference
F_x, F_y, F_z, math.degrees(gamma)

# Correcting the components of F1 using the 3-4-5 triangle ratios
# x component is negative since it's along the negative x-axis
F1_x = -F1 * (3/5)
F1_z = F1 * (4/5)   # z component is positive

# Summing up the components to get the resultant force components again
R_x = F1_x
R_y = F2_y  # Previously calculated
R_z = F1_z + F2_z + F3_z  # Summing up all the z components

# Magnitude of the resultant force
R_magnitude = np.sqrt(R_x**2 + R_y**2 + R_z**2)

# Direction angles of the resultant force (in radians)
alpha = np.arccos(R_x / R_magnitude) if R_magnitude != 0 else 0
beta = np.arccos(R_y / R_magnitude) if R_magnitude != 0 else 0
gamma = np.arccos(R_z / R_magnitude) if R_magnitude != 0 else 0

# Converting angles to degrees
alpha_degrees = np.degrees(alpha)
beta_degrees = np.degrees(beta)
gamma_degrees = np.degrees(gamma)

(R_x, R_y, R_z, R_magnitude), (alpha_degrees, beta_degrees, gamma_degrees)


def divide(dividend, divisor):
    # Constants for the 32-bit signed integer range
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31

    # Handle overflow cases
    if dividend == INT_MIN and divisor == -1:
        return INT_MAX
    if dividend == INT_MIN and divisor == 1:
        return INT_MIN

    # Determine the sign of the quotient
    negative = (dividend < 0) != (divisor < 0)

    # Work with positive numbers to avoid negative overflow issues
    dividend, divisor = abs(dividend), abs(divisor)

    # Perform the division using bit shifting
    quotient = 0
    the_sum = divisor
    multiple = 1
    while dividend >= divisor:
        if dividend >= the_sum:
            dividend -= the_sum
            quotient += multiple
            the_sum <<= 1
            multiple <<= 1
        else:
            the_sum >>= 1
            multiple >>= 1

    if negative:
        quotient = -quotient

    return min(max(INT_MIN, quotient), INT_MAX)


# Example 1
print(divide(10, 3))  # Output: 3

# Example 2
print(divide(7, -3))  # Output: -2


def merge_intervals(intervals):
    # Sort the intervals by their start times
    intervals.sort(key=lambda x: x[0])

    # Initialize an empty list to hold the merged intervals
    merged = []

    for interval in intervals:
        # If the list of merged intervals is empty or if the current interval
        # does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so we merge the current and previous intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


# Example usage
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge_intervals(intervals))


def generate_story():
    characters = ['a pirate', 'an astronaut',
                  'a wizard', 'a dragon', 'a robot']
    settings = ['in outer space', 'at the sea', 'in a magical kingdom',
                'in a futuristic city', 'in a haunted house']
    problems = ['lost a treasure', 'forgot the way back home',
                'cast a wrong spell', 'ran out of fuel', 'lost its memory']
    solutions = ['finds a map', 'meets a new friend who helps out', 'discovers a magical artifact',
                 'finds a renewable energy source', 'recovers its backup data']

    character = random.choice(characters)
    setting = random.choice(settings)
    problem = random.choice(problems)
    solution = random.choice(solutions)

    story = f"Once upon a time, {character} was {setting}. But then, they {problem}. Luckily, {character} {solution}."

    return story


# Generate and print the story
print(generate_story())


def rocket_ship():
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("💥💥💥")
    print("🚀🚀🚀🚀🚀")
    print("🌎✨🌟🌠")
    print("🌌🌌🌌🌌🌌")
    print("Blast Off!")


rocket_ship()


def magic_8_ball():
    responses = [
        "It is certain.",
        "It is decidedly so.",
        "Without a doubt.",
        "Yes – definitely.",
        "You may rely on it.",
        "As I see it, yes.",
        "Most likely.",
        "Outlook good.",
        "Yes.",
        "Signs point to yes.",
        "Reply hazy, try again.",
        "Ask again later.",
        "Better not tell you now.",
        "Cannot predict now.",
        "Concentrate and ask again.",
        "Don't count on it.",
        "My reply is no.",
        "My sources say no.",
        "Outlook not so good.",
        "Very doubtful."
    ]

    input("What is your question? ")
    print("Thinking...")
    print(random.choice(responses))


magic_8_ball()


def numUniqueEmails(emails):
    unique_emails = set()

    for email in emails:
        local, domain = email.split('@')
        # Ignore everything after the first '+'
        if '+' in local:
            local = local[:local.index('+')]
        # Remove all periods
        local = local.replace('.', '')
        normalized_email = local + '@' + domain
        unique_emails.add(normalized_email)

    return len(unique_emails)


# Example usage
emails = ["test.email+alex@leetcode.com",
          "test.e.mail+bob.cathy@leetcode.com", "testemail+david@lee.tcode.com"]
print(numUniqueEmails(emails))


def two_sum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    # Create a dictionary to hold value:index pairs.
    num_map = {}

    # Loop through the array of numbers.
    for i, num in enumerate(nums):
        # Calculate the complement of the current number.
        complement = target - num

        # Check if the complement is in our dictionary.
        if complement in num_map:
            # If it is, return the current index and the complement's index.
            return [num_map[complement], i]

        # Otherwise, add the current number's index to the dictionary.
        num_map[num] = i


# Example usage
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))


def reverse_integer(x):
    """
    :type x: int
    :rtype: int
    """
    # Handle negative numbers by storing the sign and working with the absolute value
    sign = -1 if x < 0 else 1
    x = abs(x)

    # Reverse the integer
    reversed_x = 0
    while x != 0:
        pop = x % 10
        x //= 10
        reversed_x = reversed_x * 10 + pop

    # Check for overflow and return 0 if it overflows
    if reversed_x > 2**31 - 1 or reversed_x < -2**31:
        return 0

    return sign * reversed_x


# Example usage
print(reverse_integer(123))  # Outputs: 321
print(reverse_integer(-123))  # Outputs: -321
print(reverse_integer(120))  # Outputs: 21


def two_sum(nums, target):
    """
    Find two numbers in nums that add up to target and return their indices.

    :param nums: List of integers.
    :param target: Target sum.
    :return: Indices of the two numbers such that they add up to target.
    """
    # Initialize a dictionary to store the value and its index
    prev_map = {}  # value -> index

    # Iterate over the list of numbers
    for i, n in enumerate(nums):
        # Calculate the difference needed to reach the target
        diff = target - n

        # Check if the difference is already in our map
        if diff in prev_map:
            # If found, return the current index and the index of the difference
            return [prev_map[diff], i]

        # If not found, add the current number and its index to the map
        prev_map[n] = i

    # If no solution is found, return an empty list (or throw an exception, depending on requirements)
    return []


# Example usage
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))


def rotate_and_sum_array(nums, k):
    """
    Rotate the array to the right by k steps and sum the original and rotated arrays.

    :param nums: List of integers (the original array).
    :param k: Number of steps to rotate the array by.
    :return: A new array containing the sums of the original and rotated arrays.
    """
    # Calculate the rotated array
    rotated_array = [0] * len(nums)  # Initialize with zeros
    for i, num in enumerate(nums):
        rotated_array[(i + k) % len(nums)] = num

    # Sum the original and rotated arrays
    sum_array = [nums[i] + rotated_array[i] for i in range(len(nums))]

    return sum_array


# Example usage
nums = [1, 2, 3, 4, 5]
k = 2
print(rotate_and_sum_array(nums, k))
# Expected output: [6, 8, 5, 7, 9]


def is_happy(n):
    """
    Determine if a number is a "happy number".

    A happy number is defined as a number which eventually reaches 1 when replaced by the sum of the square of each digit.

    :param n: Integer, the number to check.
    :return: Boolean, True if n is a happy number, and False otherwise.
    """
    def get_next(number):
        """
        Calculate the sum of the squares of the digits of the input number.

        :param number: Integer, the number to process.
        :return: Integer, the sum of the squares of the digits.
        """
        return sum(int(char) ** 2 for char in str(number))

    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1


# Example usage
# Expected: True, because 1^2 + 9^2 = 82, 8^2 + 2^2 = 68, 6^2 + 8^2 = 100, and 1^2 + 0^2 + 0^2 = 1.
print(is_happy(19))
# Expected: False, because 2 leads to a cycle that doesn't include 1.
print(is_happy(2))


def fizzBuzzFun(n):
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("🚀🌟")
        elif i % 3 == 0:
            result.append("🚀")
        elif i % 5 == 0:
            result.append("🌟")
        else:
            result.append(str(i))
    return result


# Example usage
n = 15
print(fizzBuzzFun(n))


def merge(intervals):
    if not intervals:
        return []

    # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        # Get the end time of the last interval in merged
        last_end = merged[-1][1]

        # If the current interval overlaps with the last interval in merged, update the end time of the last interval
        if current_start <= last_end:
            merged[-1][1] = max(last_end, current_end)
        else:
            # Otherwise, add the current interval to merged
            merged.append([current_start, current_end])

    return merged


def convert_image_to_ascii(image_path, output_width=100):
    """Convert an image to ASCII art."""
    # Define the ASCII characters in a scale of brightness
    ascii_chars = "@%#*+=-:. "

    # Load the image
    img = Image.open(image_path)

    # Convert image to grayscale
    img = img.convert("L")

    # Resize the image based on the output width, while maintaining the aspect ratio
    width, height = img.size
    aspect_ratio = height / width
    new_height = int(output_width * aspect_ratio)
    img = img.resize((output_width, new_height))

    # Convert each pixel to the corresponding ASCII character
    pixels = img.getdata()
    ascii_str = ""
    for pixel_value in pixels:
        # Map the pixel value to 0-10
        ascii_str += ascii_chars[pixel_value // 25]
    img_ascii = [ascii_str[index:index + output_width]
                 for index in range(0, len(ascii_str), output_width)]

    return "\n".join(img_ascii)


def main():
    # Path to the image file
    image_path = "path/to/your/image.jpg"

    # Convert the image to ASCII and print it
    ascii_art = convert_image_to_ascii(image_path, output_width=100)
    print(ascii_art)


if __name__ == "__main__":
    main()


def is_palindrome(s):
    cleaned = ''.join(filter(str.isalnum, s.casefold()))
    return cleaned == cleaned[::-1]


# Example usage
test_string = "A man, a plan, a canal: Panama"
print(f"'{test_string}' is a palindrome: {is_palindrome(test_string)}")


def reverse_string(s):
    return s[::-1]


# Example usage
original_string = "Hello, world!"
reversed_string = reverse_string(original_string)
print(f"Original: {original_string}")
print(f"Reversed: {reversed_string}")

for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)


def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_element = fib_sequence[i-1] + fib_sequence[i-2]
        fib_sequence.append(next_element)
    return fib_sequence[:n]


print(fibonacci(10))


def weighted_random_select(elements, weights):
    total_weight = sum(weights)
    cumulative_weights = [sum(weights[:i+1]) for i in range(len(weights))]
    r = random.uniform(0, total_weight)
    for i, total in enumerate(cumulative_weights):
        if r <= total:
            return elements[i]


# Example usage
elements = ['apple', 'banana', 'cherry']
weights = [10, 1, 1]
# 'apple' will be picked most often
print(weighted_random_select(elements, weights))


def find_peak_bitonic(arr):
    low, high = 0, len(arr) - 1
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < arr[mid + 1]:
            low = mid + 1
        else:
            high = mid
    return arr[low]


# Example usage
arr = [1, 3, 8, 12, 4, 2]
print(find_peak_bitonic(arr))  # Output: 12


class TinyURL:
    def __init__(self):
        self.url_to_code = {}
        self.code_to_url = {}
        self.base62_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def encode(self, longUrl):
        if longUrl in self.url_to_code:
            return "http://tinyurl.com/" + self.url_to_code[longUrl]
        else:
            code = ''.join(random.choices(self.base62_chars, k=6))
            if code not in self.code_to_url:
                self.url_to_code[longUrl] = code
                self.code_to_url[code] = longUrl
                return "http://tinyurl.com/" + code
            else:
                return self.encode(longUrl)  # retry with a new code

    def decode(self, shortUrl):
        code = shortUrl.split("/")[-1]
        if code in self.code_to_url:
            return self.code_to_url[code]
        else:
            return None


# Example usage
codec = TinyURL()
url = "https://www.example.com"
encoded_url = codec.encode(url)
decoded_url = codec.decode(encoded_url)
print(decoded_url)  # Output: https://www.example.com


class CircularQueue:
    def __init__(self, size):
        self.queue = [None] * size
        self.head = self.tail = -1
        self.size = size

    def enqueue(self, value):
        if (self.tail + 1) % self.size == self.head:
            print("Queue is full")
            return False
        if self.head == -1:
            self.head = 0
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True

    def dequeue(self):
        if self.head == -1:
            print("Queue is empty")
            return False
        result = self.queue[self.head]
        if self.head == self.tail:
            self.head = self.tail = -1
        else:
            self.head = (self.head + 1) % self.size
        return result


# Example usage
cq = CircularQueue(3)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
print(cq.dequeue())  # Output: 1
print(cq.dequeue())  # Output: 2


def square_root(n, precision=0.0001):
    low, high = 0, n
    while high - low > precision:
        mid = (low + high) /


def find_pairs_with_sum(nums, target):
    """Find all pairs in the list that sum up to the target."""
    seen = {}
    pairs = []
    for num in nums:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen[num] = True
    return pairs


# Example usage:
nums = [2, 7, 11, 15, -2]
target = 9
print(find_pairs_with_sum(nums, target))  # Output: [(2, 7), (-2, 11)]


def rotate_array(nums, k):
    """Rotate the array to the right by k steps."""
    k = k % len(nums)  # In case k is greater than the length of the list
    nums[:] = nums[-k:] + nums[:-k]
    return nums


# Example usage:
nums = [1, 2, 3, 4, 5]
k = 2
print(rotate_array(nums, k))  # Output: [4, 5, 1, 2, 3]


def spreadsheet_column_to_number(column_title):
    """Convert spreadsheet column title to a number."""
    result = 0
    for char in column_title:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result


# Example usage:
column_title = "AB"
print(spreadsheet_column_to_number(column_title))  # Output: 28


def first_non_repeated_char(s):
    """Find the first non-repeated character in string s."""
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    for char in s:
        if char_count[char] == 1:
            return char
    return None


# Example usage:
string = "interview"
print(first_non_repeated_char(string))  # Output: 'i'


def is_palindrome(s):
    """Check if a given string s is a palindrome."""
    return s == s[::-1]


# Example usage:
word = "racecar"
print(is_palindrome(word))  # Output: True


def solve_sudoku(board):
    def is_valid(num, row, col):
        for i in range(9):
            if board[i][col] == num or board[row][i] == num:
                return False
            if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
                return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(num, i, j):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True

    backtrack()
    return board


class TinyURL:
    def __init__(self):
        self.url_map = {}
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        self.key_length = 6
        self.base_url = "http://tinyurl.com/"

    def encode(self, longUrl):
        key = ''.join(random.choices(self.alphabet, k=self.key_length))
        while key in self.url_map:
            key = ''.join(random.choices(self.alphabet, k=self.key_length))
        self.url_map[key] = longUrl
        return self.base_url + key

    def decode(self, shortUrl):
        key = shortUrl.split('/')[-1]
        return self.url_map.get(key, None)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def all_possible_fbt(N):
    if N % 2 == 0:
        return []
    if N == 1:
        return [TreeNode(0)]
    result = []
    for i in range(1, N, 2):
        for left in all_possible_fbt(i):
            for right in all_possible_fbt(N - 1 - i):
                root = TreeNode(0)
                root.left = left
                root.right = right
                result.append(root)
    return result


def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def free_slots(working_hours, meetings, duration):
    bounds = (hours_to_minutes(
        working_hours[0]), hours_to_minutes(working_hours[1]))
    booked = merge_intervals(
        [[hours_to_minutes(start), hours_to_minutes(end)] for start, end in meetings])
    free = []
    start = bounds[0]
    for end in booked:
        if end[0] - start >= duration:
            free.append((start, end[0]))
        start = max(start, end[1])
    if bounds[1] - start >= duration:
        free.append((start, bounds[1]))
    return [(minutes_to_hours(s), minutes_to_hours(e)) for s, e in free]


def hours_to_minutes(time):
    h, m = map(int, time.split(':'))
    return h * 60 + m


def minutes_to_hours(minutes):
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def min_transactions(debts):
    balance = {}
    for debtor, creditor, amount in debts:
        balance[debtor] = balance.get(debtor, 0) - amount
        balance[creditor] = balance.get(creditor, 0) + amount

    balances = list(filter(lambda x: x != 0, balance.values()))

    def settle(balances):
        if not balances:
            return 0
        min_trans = float('inf')
        for i in range(1, len(balances)):
            if balances[0] * balances[i] < 0:
                balances[i] += balances[0]
                min_trans = min(min_trans, 1 + settle(balances[1:]))
                balances[i] -= balances[0]
        return min_trans

    return settle(balances)


class Codec:
    def encode(self, strs):
        return ''.join([str(len(s)) + '/' + s for s in strs])

    def decode(self, s):
        res = []
        i = 0
        while i < len(s):
            slash_index = s.find('/', i)
            size = int(s[i:slash_index])
            i = slash_index + size + 1
            res.append(s[slash_index + 1: i])
        return res


# Example usage:
codec = Codec()
encoded_str = codec.encode(["abc", "def", "ghi"])
print("Encoded String:", encoded_str)
decoded_strs = codec.decode(encoded_str)
print("Decoded Strings:", decoded_strs)


class TinyURL:
    def __init__(self):
        self.url_map = {}
        self.id = 0

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL."""
        shortUrl = "http://tinyurl.com/" + str(self.id)
        self.url_map[shortUrl] = longUrl
        self.id += 1
        return shortUrl

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL."""
        return self.url_map.get(shortUrl, None)


# Example usage:
codec = TinyURL()
url = "https://www.example.com"
encoded_url = codec.encode(url)
decoded_url = codec.decode(encoded_url)
print(f"Encoded: {encoded_url}, Decoded: {decoded_url}")


class SnapshotArray:
    def __init__(self, length):
        self.array = [[[-1, 0]] for _ in range(length)]
        self.snap_id = 0

    def set(self, index, val):
        if self.array[index][-1][0] == self.snap_id:
            self.array[index][-1][1] = val
        else:
            self.array[index].append([self.snap_id, val])

    def snap(self):
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index, snap_id):
        snapshots = self.array[index]
        # Binary search for the first element in snapshots with a snap_id <= given snap_id
        lo, hi = 0, len(snapshots) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if snapshots[mid][0] <= snap_id:
                lo = mid
            else:
                hi = mid - 1
        return snapshots[lo][1]


# Example usage:
snap_arr = SnapshotArray(3)
snap_arr.set(0, 5)
snap_id = snap_arr.snap()  # Take a snapshot, returns 0
print(snap_arr.get(0, snap_id))  # Output: 5


class RandomizedCollection:
    def __init__(self):
        self.vals, self.idx = [], {}

    def insert(self, val):
        self.vals.append(val)
        if val in self.idx:
            self.idx[val].add(len(self.vals) - 1)
        else:
            self.idx[val] = {len(self.vals) - 1}
        return len(self.idx[val]) == 1

    def remove(self, val):
        if val not in self.idx or not self.idx[val]:
            return False
        remove_idx, last_element = self.idx[val].pop(), self.vals[-1]
        self.vals[remove_idx] = last_element
        if self.idx[last_element]:
            self.idx[last_element].add(remove_idx)
            self.idx[last_element].discard(len(self.vals) - 1)
        self.vals.pop()
        return True

    def getRandom(self):
        return random.choice(self.vals)


# Example usage:
obj = RandomizedCollection()
print(obj.insert(1))  # True, inserted first 1
print(obj.insert(1))  # False, inserted second 1
print(obj.remove(1))  # True, removed one 1
print(obj.getRandom())  # Randomly get 1


class MedianFinder:
    def __init__(self):
        self.lo = []  # Max heap for the lower half
        self.hi = []  # Min heap for the upper half

    def addNum(self, num):
        # Push negated num to maintain max heap property
        heapq.heappush(self.lo, -num)
        # Balance heaps
        if self.lo and self.hi and (-self.lo[0] > self.hi[0]):
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (self.hi[0] - self.lo[0]) / 2.0


# Example usage:
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # Output: 1.5
mf.addNum(3)
print(mf.findMedian())  # Output: 2


class TimeMap:
    def __init__(self):
        self.store = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        A = self.store[key]
        i = bisect.bisect_right(A, (timestamp, chr(255)))
        return A[i-1][1] if i else ""


# Example usage:
tm = TimeMap()
tm.set("foo", "bar", 1)
print(tm.get("foo", 1))  # Output: "bar"
# Should return "bar" since timestamp 3 >= 1 and there are no later entries
print(tm.get("foo", 3))


def longest_palindrome(s):
    if not s or len(s) < 1:
        return ""
    start, end = 0, 0
    for i in range(len(s)):
        len1 = expand_around_center(s, i, i)
        len2 = expand_around_center(s, i, i + 1)
        length = max(len1, len2)
        if length > end - start:
            start = i - (length - 1) // 2
            end = i + length // 2
    return s[start:end + 1]


def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1


# Test
print(longest_palindrome("babad"))  # Output: "bab" or "aba"
print(longest_palindrome("cbbd"))  # Output: "bb"


def remove_duplicates(nums):
    if not nums:
        return 0
    j = 0
    for i in range(1, len(nums)):
        if nums[i] != nums[j]:
            j += 1
            nums[j] = nums[i]
    return j + 1


# Test
nums = [1, 1, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(nums)
print(length)  # Output: 5
print(nums[:length])  # Output: [1, 2, 3, 4, 5]


def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# Test
print(fibonacci(10))  # Output: 55


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# Test
print(is_prime(7))  # Output: True
print(is_prime(10))  # Output: False


def reverse_string(s):
    return s[::-1]


# Test
print(reverse_string("hello"))  # Output: "olleh"


def length_of_longest_substring(s):
    char_map = {}
    max_len = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_map and char_map[char] >= start:
            start = char_map[char] + 1
        char_map[char] = end
        max_len = max(max_len, end - start + 1)

    return max_len


# Example usage
print(length_of_longest_substring("abcabcbb"))  # Output: 3


def is_valid(s):
    stack = []
    paren_map = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in paren_map:
            top_element = stack.pop() if stack else '#'
            if paren_map[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack


# Example usage
print(is_valid("()[]{}"))  # Output: True
print(is_valid("(]"))  # Output: False


def merge(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged


# Example usage
# Output: [[1, 6], [8, 10], [15, 18]]
print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))


def lengthOfLongestSubstring(s: str) -> int:
    char_map = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        if s[right] in char_map:
            left = max(left, char_map[s[right]] + 1)
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length


# Example usage:
print(lengthOfLongestSubstring("abcabcbb"))  # Output: 3
print(lengthOfLongestSubstring("bbbbb"))  # Output: 1


def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    nums = sorted(nums1 + nums2)
    n = len(nums)
    if n % 2 == 1:
        return nums[n // 2]
    else:
        return (nums[n // 2 - 1] + nums[n // 2]) / 2


# Example usage:
print(findMedianSortedArrays([1, 3], [2]))  # Output: 2.0
print(findMedianSortedArrays([1, 2], [3, 4]))  # Output: 2.5


def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        max_area = max(max_area, min(height[left], height[right]) * width)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area


# Example usage:
print(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # Output: 49


def generateParenthesis(n: int) -> list[str]:
    def backtrack(S='', left=0, right=0):
        if len(S) == 2 * n:
            result.append(S)
            return
        if left < n:
            backtrack(S + '(', left + 1, right)
        if right < left:
            backtrack(S + ')', left, right + 1)

    result = []
    backtrack()
    return result


# Example usage:
print(generateParenthesis(3))
# Output: ["((()))","(()())","(())()","()(())","()()()"]


def merge(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    merged = []

    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


# Example usage:
print(merge([[1, 3], [2, 6], [8, 10], [15, 18]]))
# Output: [[1,6],[8,10],[15,18]]
print(merge([[1, 4], [4, 5]]))
# Output: [[1,5]]


class GeneticAlgorithmTSP:
    def __init__(self, cities, population_size, mutation_rate, generations):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            route = random.sample(self.cities, len(self.cities))
            population.append(route)
        return population

    def fitness(self, route):
        return sum(np.linalg.norm(np.array(route[i]) - np.array(route[i+1])) for i in range(len(route)-1))

    def selection(self):
        sorted_population = sorted(self.population, key=self.fitness)
        return sorted_population[:self.population_size//2]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1)-1)
        child = parent1[:crossover_point] + \
            [city for city in parent2 if city not in parent1[:crossover_point]]
        return child

    def mutate(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve(self):
        for _ in range(self.generations):
            new_population = []
            selected = self.selection()
            for i in range(0, len(selected), 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1 = self.mutate(self.crossover(parent1, parent2))
                child2 = self.mutate(self.crossover(parent2, parent1))
                new_population.extend([child1, child2])
            self.population = new_population

    def get_best_route(self):
        return min(self.population, key=self.fitness)


class MusicRecommender:
    def __init__(self, data):
        self.data = data
        self.user_song_matrix = self.create_user_song_matrix()

    def create_user_song_matrix(self):
        return self.data.pivot(index='user_id', columns='song_id', values='listen_count').fillna(0)

    def calculate_similarity(self):
        user_song_sparse = csr_matrix(self.user_song_matrix.values)
        return cosine_similarity(user_song_sparse)

    def recommend_songs(self, user_id, top_n=5):
        user_index = self.user_song_matrix.index.get_loc(user_id)
        similarity_matrix = self.calculate_similarity()
        user_similarity_scores = similarity_matrix[user_index]
        song_listens = self.user_song_matrix.values[user_index]
        scores = user_similarity_scores.dot(
            self.user_song_matrix.values) / np.array([np.abs(user_similarity_scores).sum()])
        song_recommendations = list(
            self.user_song_matrix.columns[np.argsort(scores)[::-1]])
        return [song for song in song_recommendations if song_listens[song] == 0][:top_n]


data = pd.read_csv('user_song_data.csv')
recommender = MusicRecommender(data)
print(recommender.recommend_songs(user_id=1))


class FaceFilter:
    def __init__(self, image_path, filter_type):
        self.image_path = image_path
        self.filter_type = filter_type
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def apply_filter(self, face):
        if self.filter_type == 'blur':
            return cv2.GaussianBlur(face, (99, 99), 30)
        elif self.filter_type == 'cartoon':
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
            color = cv2.bilateralFilter(face, 9, 300, 300)
            return cv2.bitwise_and(color, color, mask=edges)
        else:
            return face

    def process_image(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            filtered_face = self.apply_filter(face)
            image[y:y+h, x:x+w] = filtered_face
        cv2.imshow('Filtered Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


face_filter = FaceFilter('path/to/your/image.jpg', 'cartoon')
face_filter.process_image()


class StockMonitor:
    def __init__(self, stock_symbol, threshold):
        self.stock_symbol = stock_symbol
        self.threshold = threshold
        self.api_url = f'https://api.example.com/stocks/{self.stock_symbol}'

    def get_stock_price(self):
        response = requests.get(self.api_url)
        data = response.json()
        return data['price']

    def check_price(self):
        price = self.get_stock_price()
        print(f"Current price of {self.stock_symbol}: {price}")
        if price > self.threshold:
            print(
                f"Alert: {self.stock_symbol} price crossed the threshold of {self.threshold}")

    def start_monitoring(self, interval=1):
        schedule.every(interval).minutes.do(self.check_price)
        while True:
            schedule.run_pending()
            time.sleep(1)


monitor = StockMonitor('AAPL', 150)
monitor.start_monitoring()


class Room:
    def __init__(self, name, description, items=None):
        self.name = name
        self.description = description
        self.items = items if items else []
        self.connected_rooms = {}

    def connect_room(self, room, direction):
        self.connected_rooms[direction] = room

    def get_room_in_direction(self, direction):
        return self.connected_rooms.get(direction)


class AdventureGame:
    def __init__(self):
        self.rooms = self.create_rooms()
        self.current_room = self.rooms['entrance']

    def create_rooms(self):
        entrance = Room('Entrance', 'You are at the entrance of a dark cave.')
        hall = Room(
            'Hall', 'You are in a large hall with torches on the walls.')
        treasure_room = Room(
            'Treasure Room', 'You found the treasure room!', ['gold', 'jewels'])

        entrance.connect_room(hall, 'north')
        hall.connect_room(entrance, 'south')
        hall.connect_room(treasure_room, 'east')
        treasure_room.connect_room(hall, 'west')

        return {'entrance': entrance, 'hall': hall, 'treasure_room': treasure_room}

    def move(self, direction):
        next_room = self.current_room.get_room_in_direction(direction)
        if next_room:
            self.current_room = next_room
            print(f"You moved to the {self.current_room.name}.")
            print(self.current_room.description)
        else:
            print("You can't go that way.")

    def play(self):
        while True:
            command = input("Enter a command: ").strip().lower()
            if command in ['north', 'south', 'east', 'west']:
                self.move(command)
            elif command == 'quit':
                print("Thanks for playing!")
                break
            else:
                print("Unknown command.")


game = AdventureGame()
game.play()


def letterCombinations(digits):
    if not digits:
        return []

    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index, current):
        if index == len(digits):
            result.append(''.join(current))
            return
        letters = phone_map[digits[index]]
        for letter in letters:
            current.append(letter)
            backtrack(index + 1, current)
            current.pop()

    backtrack(0, [])
    return result


# Example usage:
print(letterCombinations("23"))
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]


def longestPalindrome(s):
    if len(s) <= 1:
        return s

    start, max_length = 0, 1

    def expandAroundCenter(left, right):
        nonlocal start, max_length
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_length:
                start = left
                max_length = right - left + 1
            left -= 1
            right += 1

    for i in range(len(s)):
        expandAroundCenter(i, i)  # Odd length
        expandAroundCenter(i, i + 1)  # Even length

    return s[start:start + max_length]


# Example usage:
print(longestPalindrome("babad"))  # Output: "bab" or "aba"
print(longestPalindrome("cbbd"))  # Output: "bb"


def groupAnagrams(strs):
    from collections import defaultdict
    anagrams = defaultdict(list)

    for s in strs:
        sorted_str = ''.join(sorted(s))
        anagrams[sorted_str].append(s)

    return list(anagrams.values())


# Example usage:
print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# Output: [["eat","tea","ate"],["tan","nat"],["bat"]]


def compute_lps_array(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text, pattern):
    lps = compute_lps_array(pattern)
    result = []
    i = 0  # index for text
    j = 0  # index for pattern

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            result.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return result


print(kmp_search("ababcabcabababd", "ababd"))  # Output: [10]


def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 'Q':
                return False
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        return True

    def solve(board, col):
        if col >= n:
            result.append([''.join(row) for row in board])
            return
        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 'Q'
                solve(board, col + 1)
                board[i][col] = '.'

    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    solve(board, 0)
    return result


print(solve_n_queens(4))
# Output:
# [
#   [".Q..", "...Q", "Q...", "..Q."],
#   ["..Q.", "Q...", "...Q", ".Q.."]
# ]


class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {v: [] for v in range(vertices)}

    def add_edge(self, v, w):
        self.adj_list[v].append(w)

    def is_cyclic_util(self, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.adj_list[v]:
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def is_cyclic(self):
        visited = [False] * self.vertices
        rec_stack = [False] * self.vertices

        for node in range(self.vertices):
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False


graph = Graph(4)
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 0)
graph.add_edge(2, 3)

print(graph.is_cyclic())  # Output: True


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


trie = Trie()
trie.insert("apple")
print(trie.search("apple"))   # Output: True
print(trie.search("app"))     # Output: False
print(trie.starts_with("app"))  # Output: True
trie.insert("app")
print(trie.search("app"))     # Output: True


def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


print(longest_common_subsequence("ABCBDAB", "BDCAB"))  # Output: 4


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def middle_node(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


# Example usage:
# Input: 1 -> 2 -> 3 -> 4 -> 5
# Output: 3 -> 4 -> 5
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# Output: ListNode { val: 3, next: ListNode { val: 4, next: [ListNode] } }
print(middle_node(head))


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(l1, l2):
    dummy = ListNode()
    current = dummy

    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 if l1 else l2

    return dummy.next


# Example usage:
# l1: 1 -> 2 -> 4
# l2: 1 -> 3 -> 4
l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(4)))
merged = merge_two_lists(l1, l2)
# Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4


def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)

    return not stack


# Example usage:
print(is_valid("()[]{}"))  # Output: True
print(is_valid("(]"))      # Output: False


def reverse_string(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


# Example usage:
s = ['h', 'e', 'l', 'l', 'o']
reverse_string(s)
print(s)  # Output: ['o', 'l', 'l', 'e', 'h']


def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []


# Example usage:
print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]


def exist(board: List[List[str]], word: str) -> bool:
    def dfs(board, word, i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], '/'
        res = dfs(board, word, i + 1, j, k + 1) or \
            dfs(board, word, i - 1, j, k + 1) or \
            dfs(board, word, i, j + 1, k + 1) or \
            dfs(board, word, i, j - 1, k + 1)
        board[i][j] = tmp
        return res

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(board, word, i, j, 0):
                return True
    return False


# Example usage:
board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
word = "ABCCED"
print(exist(board, word))  # Output: True


def trap(height: List[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    water_trapped = 0

    while left < right:
        if left_max < right_max:
            left += 1
            left_max = max(left_max, height[left])
            water_trapped += left_max - height[left]
        else:
            right -= 1
            right_max = max(right_max, height[right])
            water_trapped += right_max - height[right]

    return water_trapped


# Example usage:
print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # Output: 6


def find_min(nums: List[int]) -> int:
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid

    return nums[left]


# Example usage:
print(find_min([3, 4, 5, 1, 2]))  # Output: 1


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


# Example usage:
lru_cache = LRUCache(2)
lru_cache.put(1, 1)
lru_cache.put(2, 2)
print(lru_cache.get(1))  # Output: 1
lru_cache.put(3, 3)
print(lru_cache.get(2))  # Output: -1 (as 2 was evicted)


def permute(s):
    def backtrack(path, options):
        if not options:
            res.append(path)
            return
        for i in range(len(options)):
            backtrack(path + options[i], options[:i] + options[i+1:])

    res = []
    backtrack("", s)
    return res


# Example usage:
s = "abc"
print(permute(s))  # Output: ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']


def num_islands(grid):
    if not grid:
        return 0

    def dfs(grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(grid, i + 1, j)
        dfs(grid, i - 1, j)
        dfs(grid, i, j + 1)
        dfs(grid, i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                dfs(grid, i, j)

    return count


# Example usage:
grid = [
    ["1", "1", "1", "1", "0"],
    ["1", "1", "0", "1", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "0", "0", "0"]
]
print(num_islands(grid))  # Output: 1


def max_ones_after_flip(arr):
    total_ones = sum(arr)
    max_diff = 0
    current_diff = 0

    for i in range(len(arr)):
        value = 1 if arr[i] == 0 else -1
        current_diff = max(value, current_diff + value)
        max_diff = max(max_diff, current_diff)

    return total_ones + max_diff


# Example usage:
arr = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
print(max_ones_after_flip(arr))  # Output: 8


def min_window(s: str, t: str) -> str:
    if not s or not t:
        return ""

    dict_t = Counter(t)
    required = len(dict_t)

    l, r = 0, 0
    formed = 0
    window_counts = {}

    ans = float("inf"), None, None

    while r < len(s):
        char = s[r]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while l <= r and formed == required:
            char = s[l]

            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            l += 1

        r += 1

    return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]


# Example usage:
s = "ADOBECODEBANC"
t = "ABC"
print(min_window(s, t))  # Output: "BANC"


def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


def is_palindrome(x):
    s = str(x)
    return s == s[::-1]


def length_of_longest_substring(s):
    char_map = {}
    left = max_length = 0
    for right, char in enumerate(s):
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        max_length = max(max_length, right - left + 1)
    return max_length


def is_valid(s):
    stack = []
    parens = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in parens:
            top_element = stack.pop() if stack else '#'
            if top_element != parens[char]:
                return False
        else:
            stack.append(char)
    return not stack


def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def max_sub_array(nums):
    current_sum = max_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


def climb_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n+1):
        a, b = b, a + b
    return b


def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def mask_data(data):
    return '*' * (len(data) - 4) + data[-4:]


print(mask_data("1234567812345678"))  # Output: ************5678


def check_winner(board):
    # Check rows, columns, and diagonals
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != ' ':
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    return None


board = [
    ['X', 'O', 'X'],
    [' ', 'X', 'O'],
    ['O', 'X', ' ']
]
print(check_winner(board))  # Output: X


def has_unique_characters(s):
    return len(set(s)) == len(s)


print(has_unique_characters("abcdef"))
print(has_unique_characters("hello"))


def reverse_words(sentence):
    return ' '.join(word[::-1] for word in sentence.split())


print(reverse_words("Hello World"))
