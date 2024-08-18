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

// Example
const arr = [3, 7, 1, 2, 8, 4, 5];
console.log("The missing number is:", findMissingNumber(arr));

//Depth of binary tree with Tyler

class TreeNode {
    constructor(val = 0, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function maxDepth(root) {
    if (!root) return 0;
    const leftDepth = maxDepth(root.left);
    const rightDepth = maxDepth(root.right);
    return Math.max(leftDepth, rightDepth) + 1;
}

// Example usage requires building a binary tree

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key) {
        if (!this.cache.has(key)) return -1;
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    put(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size === this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }
}

function deepClone(obj, hash = new WeakMap()) {
    if (Object(obj) !== obj) return obj;
    if (hash.has(obj)) return hash.get(obj);

    const result = Array.isArray(obj) ? [] : obj.constructor ? new obj.constructor() : Object.create(null);
    hash.set(obj, result);

    if (obj instanceof Map) {
        obj.forEach((value, key) => result.set(key, deepClone(value, hash)));
    }

    return Object.assign(result, ...Object.keys(obj).map(
        key => ({ [key]: deepClone(obj[key], hash) })
    ));
}

function throttle(func, limit) {
    let inThrottle;
    return function (...args) {
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

class EventEmitter {
    constructor() {
        this.events = {};
    }

    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
    }

    emit(event, ...args) {
        if (this.events[event]) {
            this.events[event].forEach(listener => listener.apply(this, args));
        }
    }

    off(event, listener) {
        if (!this.events[event]) return;

        const index = this.events[event].indexOf(listener);
        if (index > -1) {
            this.events[event].splice(index, 1);
        }
    }
}

function isPrime(num) {
    if (num < 2) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
        if (num % i === 0) return false;
    }
    return true;
}

function primeChecker() {
    const cache = {};

    return function(num) {
        if (cache[num] !== undefined) {
            return cache[num];
        }

        const result = isPrime(num);
        cache[num] = result;
        return result;
    };
}

// Example usage
const checkPrime = primeChecker();
console.log(checkPrime(11)); // Output: true
console.log(checkPrime(11)); // Output: true (faster due to memoization)

function permute(str) {
    if (str.length <= 1) {
        return [str];
    }

    const permutations = [];
    for (let i = 0; i < str.length; i++) {
        const char = str[i];
        const remainingStr = str.slice(0, i) + str.slice(i + 1);
        for (let subPermutation of permute(remainingStr)) {
            permutations.push(char + subPermutation);
        }
    }

    return permutations;
}

// Example usage
console.log(permute("abc")); // Output: ["abc", "acb", "bac", "bca", "cab", "cba"]

function debounce(func, delay) {
    let timeoutId;

    return function(...args) {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
}

// Example usage
const log = () => console.log('Debounced function called');
const debouncedLog = debounce(log, 1000);
debouncedLog();
debouncedLog();
debouncedLog(); // "Debounced function called" will only be logged once after 1 second
