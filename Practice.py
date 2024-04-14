# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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
