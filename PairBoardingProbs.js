function removeDuplicates(arr) {
    return [...new Set(arr)];
}

// Example usage:
const numbers = [1, 2, 3, 2, 4, 3, 5];
const uniqueNumbers = removeDuplicates(numbers);
console.log(uniqueNumbers); // Output: [1, 2, 3, 4, 5]

function removeDuplicates(arr) {
    return arr.filter((item, index) => arr.indexOf(item) === index);
}

// Example usage:
const numbers = [1, 2, 3, 2, 4, 3, 5];
const uniqueNumbers = removeDuplicates(numbers);
console.log(uniqueNumbers); // Output: [1, 2, 3, 4, 5]


//commented tech interview pairboarding, listnode helper and reverse list

function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val)
    this.next = (next === undefined ? null : next)
}

function reverseList(head) {
    let prev = null;
    let current = head;
    let next = null;

    while (current != null) {
        // Store the next node
        next = current.next;
        // Reverse the current node's pointer
        current.next = prev;
        // Move pointers one position ahead
        prev = current;
        current = next;
    }
    head = prev;

    return head;
}

//Valid Paranthesis tech interview question pairboarding, commented
//instructions:
// Given a string s containing just the characters '(', ')', '{', '}', '[', and ']', determine if the input string is valid.

// An input string is valid if:

// Open brackets must be closed by the same type of brackets.
// Open brackets must be closed in the correct order.
// Example 1:
// Input: s = "()"
// Output: true
// Example 2:
// Input: s = "()[]{}"
// Output: true
// Example 3:
// Input: s = "(]"
// Output: false
// Example 4:
// Input: s = "([)]"
// Output: false
// Example 5:
// Input: s = "{[]}"
// Output: true
// Constraints:
// 1 <= s.length <= 10^4
// s consists of parentheses only '()[]{}'.

function isValid(s) {
    // Initialize a stack to keep track of opening brackets
    const stack = [];
    // Create a mapping for the parentheses
    const mapping = {
        ')': '(',
        '}': '{',
        ']': '['
    };

    for (let char of s) {
        if (mapping[char]) {
            // Pop the top element if the stack is not empty
            // Otherwise assign a dummy value that won't match
            const topElement = stack.length === 0 ? '#' : stack.pop();
            // Check if the popped element is the mapping pair of the current character
            if (topElement !== mapping[char]) {
                return false;
            }
        } else {
            // If it's an opening bracket, push it onto the stack
            stack.push(char);
        }
    }

    // The stack should be empty at the end for a valid expression
    return stack.length === 0;
}

function findMissingNumber(arr) {
    // Ensure the array is sorted to correctly calculate the expected sum
    arr.sort((a, b) => a - b);

    // Calculate the number of elements that should be in the array
    const n = arr.length + 1;

    // Calculate the expected sum of numbers from 1 to n
    const expectedSum = n * (arr[0] + arr[arr.length - 1]) / 2;

    // Calculate the actual sum of elements in the array
    const actualSum = arr.reduce((sum, current) => sum + current, 0);

    // The difference is the missing number
    return expectedSum - actualSum;
}

// Example usage
const arr = [3, 7, 1, 2, 8, 4, 5];
console.log("The missing number is:", findMissingNumber(arr));
