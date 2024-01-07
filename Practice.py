# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
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

