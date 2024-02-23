# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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

import random

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
