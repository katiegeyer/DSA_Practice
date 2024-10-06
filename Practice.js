function twoSum(nums, target) {
    let indices = {};

    for (let i = 0; i < nums.length; i++) {
        let complement = target - nums[i];

        if (indices[complement] !== undefined) {
            return [indices[complement], i];
        }

        indices[nums[i]] = i;
    }

    return []; // Return an empty array if no pair is found
}

/**
 * @param {number} n
 * @return {string[]}
 */
var generateParenthesis = function (n) {
    const result = [];

    function backtrack(s = '', open = 0, close = 0) {
        if (s.length === 2 * n) {
            result.push(s);
            return;
        }

        if (open < n) {
            backtrack(s + '(', open + 1, close);
        }

        if (close < open) {
            backtrack(s + ')', open, close + 1);
        }
    }

    backtrack();
    return result;
};

var divide = function (dividend, divisor) {
    // Constants for the 32-bit signed integer range
    const INT_MAX = Math.pow(2, 31) - 1;
    const INT_MIN = Math.pow(-2, 31);

    // Handle overflow cases
    if (dividend === INT_MIN && divisor === -1) {
        return INT_MAX;
    }

    // Determine the sign of the quotient
    let negative = (dividend < 0) !== (divisor < 0);

    // Work with positive numbers to avoid negative overflow issues
    dividend = Math.abs(dividend);
    divisor = Math.abs(divisor);

    // Perform the division using subtraction
    let quotient = 0;
    while (dividend >= divisor) {
        let tempDivisor = divisor;
        let multiple = 1;
        while ((tempDivisor << 1) <= dividend) {
            tempDivisor <<= 1;
            multiple <<= 1;
        }
        dividend -= tempDivisor;
        quotient += multiple;
    }

    // Apply the sign to the quotient
    if (negative) {
        quotient = -quotient;
    }

    // Ensure the quotient is within the 32-bit signed integer range
    return Math.max(INT_MIN, Math.min(INT_MAX, quotient));
};

// Example 1
console.log(divide(10, 3));  // Output: 3

// Example 2
console.log(divide(7, -3));  // Output: -2

document.addEventListener('DOMContentLoaded', function () {
    // This function runs when the DOM is fully loaded.

    // Get references to the DOM elements.
    const textElement = document.getElementById('textElement');
    const changeTextButton = document.getElementById('changeTextButton');

    // Add an event listener to the button for the 'click' event.
    changeTextButton.addEventListener('click', function () {
        // Change the text and color of the paragraph element.
        textElement.textContent = 'The text has been changed!';
        textElement.style.color = 'blue';
    });
});

function twoSum(nums, target) {
    // Initialize a map to store the value and its index
    const numMap = new Map();

    // Iterate over the array
    for (let i = 0; i < nums.length; i++) {
        // Calculate the complement by subtracting the current value from the target
        const complement = target - nums[i];

        // Check if the complement exists in our map
        if (numMap.has(complement)) {
            // If found, return an array containing the index of the complement and the current index
            return [numMap.get(complement), i];
        }

        // Store the current value and its index in the map
        numMap.set(nums[i], i);
    }

    // If no solution is found, throw an error or return null/undefined
    return null;
}

// Example usage
const nums = [2, 7, 11, 15];
const target = 9;
console.log(twoSum(nums, target));

function twoSum(nums, target) {
    // Create a map to store numbers and their indices
    const map = new Map();

    // Iterate through the array of numbers
    for (let i = 0; i < nums.length; i++) {
        // Calculate the complement of the current number
        const complement = target - nums[i];

        // Check if the complement exists in the map
        if (map.has(complement)) {
            // If it does, return the current index and the index of the complement
            return [map.get(complement), i];
        }

        // If not, add the current number along with its index to the map
        map.set(nums[i], i);
    }

    // If no two numbers sum up to the target, return an empty array or a message
    return [];
}

// Example usage:
const nums = [2, 7, 11, 15];
const target = 9;
console.log(twoSum(nums, target)); // Output: [0, 1]

function rotateAndSumArray(nums, k) {
    // The array that will hold the rotated version of nums
    let rotatedArray = new Array(nums.length);
    // The final array that will contain the sum of the original and rotated arrays
    let sumArray = [];

    // First, rotate the array by k steps
    for (let i = 0; i < nums.length; i++) {
        // Calculate the new position for each element after rotation
        let newPosition = (i + k) % nums.length;
        // Place each element in its new position in the rotatedArray
        rotatedArray[newPosition] = nums[i];
    }

    // Next, sum the original array and the rotated array
    for (let i = 0; i < nums.length; i++) {
        sumArray.push(nums[i] + rotatedArray[i]);
    }

    return sumArray;
}

// Example usage
const nums = [1, 2, 3, 4, 5];
const k = 2;
console.log(rotateAndSumArray(nums, k));
// Expected output for the example: [6, 8, 5, 7, 9]

function rotate(matrix) {
    const n = matrix.length;

    // Transpose the matrix
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            // Swap elements matrix[i][j] and matrix[j][i]
            [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
        }
    }

    // Reverse each row
    for (let i = 0; i < n; i++) {
        matrix[i].reverse();
    }
}

// Example usage
const matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
rotate(matrix1);
console.log(matrix1); // Output: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

const matrix2 = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]];
rotate(matrix2);
console.log(matrix2); // Output: [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]
function isPalindrome(s) {
    let cleaned = s.toLowerCase().replace(/[\W_]/g, ''); // \W matches any non-word character, _ is underscore
    return cleaned === cleaned.split('').reverse().join('');
}

// Example usage
const testString = "A man, a plan, a canal: Panama";
console.log(`'${testString}' is a palindrome: ${isPalindrome(testString)}`);

function reverseString(s) {
    return s.split('').reverse().join('');
}

// Example usage
const originalString = "Hello, world!";
const reversedString = reverseString(originalString);
console.log(`Original: ${originalString}`);
console.log(`Reversed: ${reversedString}`);

function isArray(value) {
    return Array.isArray(value);
}

console.log(isArray([1, 2, 3])); // true
console.log(isArray({ foo: 123 })); // false

function makeCounter() {
    let count = 0;
    return function () {
        return count++;
    };
}

let counter = makeCounter();
console.log(counter()); // 0
console.log(counter()); // 1

let a;
console.log(a); // undefined

let b = null;
console.log(b); // null

let obj = { a: 1, b: { c: 2 } };
let clone = JSON.parse(JSON.stringify(obj));

clone.b.c = 20;
console.log(obj.b.c); // 2

function generateSubsets(set) {
    const subsets = [];
    const totalSubsets = Math.pow(2, set.length);
    for (let i = 0; i < totalSubsets; i++) {
        const subset = [];
        for (let j = 0; j < set.length; j++) {
            if (i & (1 << j)) {
                subset.push(set[j]);
            }
        }
        subsets.push(subset);
    }
    return subsets;
}

// Example usage:
const mySet = [1, 2, 3];
console.log(generateSubsets(mySet));

function solveSudoku(board) {
    function isValid(num, row, col) {
        for (let i = 0; i < 9; i++) {
            const boxRow = 3 * Math.floor(row / 3) + Math.floor(i / 3);
            const boxCol = 3 * Math.floor(col / 3) + i % 3;
            if (board[i][col] === num || board[row][i] === num || board[boxRow][boxCol] === num) {
                return false;
            }
        }
        return true;
    }

    function backtrack() {
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                if (board[row][col] === '.') {
                    for (let num = 1; num <= 9; num++) {
                        if (isValid(String(num), row, col)) {
                            board[row][col] = String(num);
                            if (backtrack()) return true;
                            board[row][col] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    backtrack();
    return board;
}

function simpleGrep(pattern, lines) {
    return lines.filter(line => line.includes(pattern));
}

// Example usage:
const lines = ["hello world", "hello there", "hello", "world"];
console.log(simpleGrep("hello", lines));  // Output: ["hello world", "hello there", "hello"]

function flattenArray(arr) {
    const flat = [];
    arr.forEach(item => {
        if (Array.isArray(item)) {
            flat.push(...flattenArray(item));
        } else {
            flat.push(item);
        }
    });
    return flat;
}

// Example usage:
const nestedArray = [1, [2, [3, [4]], 5]];
console.log(flattenArray(nestedArray));  // Output: [1, 2, 3, 4, 5]

function debounce(func, wait) {
    let timeout;
    return function () {
        const context = this, args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// Example usage:
window.addEventListener('resize', debounce(() => {
    console.log('Resize event triggered!');
}, 200));

function areAnagrams(str1, str2) {
    const normalize = str => str.toLowerCase().replace(/[\W_]+/g, '').split('').sort().join('');
    return normalize(str1) === normalize(str2);
}

// Example usage:
console.log(areAnagrams("listen", "silent"));  // Output: true

function generateSubsets(set) {
    const subsets = [];
    const totalSubsets = Math.pow(2, set.length);
    for (let i = 0; i < totalSubsets; i++) {
        const subset = [];
        for (let j = 0; j < set.length; j++) {
            if (i & (1 << j)) {
                subset.push(set[j]);
            }
        }
        subsets.push(subset);
    }
    return subsets;
}

// Example usage:
const mySet = [1, 2, 3];
console.log(generateSubsets(mySet));

function findPeakElement(nums) {
    let left = 0;
    let right = nums.length - 1;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] > nums[mid + 1]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}

// Example usage:
const nums = [1, 2, 1, 3, 5, 6, 4];
console.log("Peak Element Index:", findPeakElement(nums));

function maxProductSubarray(nums) {
    let maxProduct = nums[0];
    let minProduct = nums[0];
    let result = nums[0];

    for (let i = 1; i < nums.length; i++) {
        const temp = maxProduct;
        maxProduct = Math.max(nums[i], maxProduct * nums[i], minProduct * nums[i]);
        minProduct = Math.min(nums[i], temp * nums[i], minProduct * nums[i]);
        result = Math.max(result, maxProduct);
    }

    return result;
}

// Example usage:
const nums = [2, 3, -2, 4];
console.log("Maximum Product Subarray:", maxProductSubarray(nums));

function rotateWord(word, k) {
    return word.slice(-k) + word.slice(0, -k);
}

function rotateWords(words, k) {
    return words.map(word => rotateWord(word, k));
}

// Example usage:
const words = ["hello", "world", "python"];
const rotatedWords = rotateWords(words, 2);
console.log("Rotated Words:", rotatedWords);

class Codec {
    encode(strs) {
        return strs.map(str => str.length + '/' + str).join('');
    }

    decode(s) {
        const res = [];
        let i = 0;
        while (i < s.length) {
            const slashIndex = s.indexOf('/', i);
            const size = parseInt(s.substring(i, slashIndex));
            i = slashIndex + size + 1;
            res.push(s.substring(slashIndex + 1, i));
        }
        return res;
    }
}

// Example usage:
const codec = new Codec();
const encodedStr = codec.encode(["abc", "def", "ghi"]);
console.log("Encoded String:", encodedStr);
const decodedStrs = codec.decode(encodedStr);
console.log("Decoded Strings:", decodedStrs);

function uniquePaths(grid) {
    const rows = grid.length;
    const cols = grid[0].length;
    const dp = Array.from(Array(rows), () => Array(cols).fill(0));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (grid[i][j] === 1) {
                dp[i][j] = 0;
            } else if (i === 0 && j === 0) {
                dp[i][j] = 1;
            } else if (i === 0) {
                dp[i][j] = dp[i][j - 1];
            } else if (j === 0) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }
    return dp[rows - 1][cols - 1];
}

// Example usage:
const grid = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
];
console.log("Unique Paths:", uniquePaths(grid));

function flip(stack, k) {
    return stack.slice(0, k).reverse().concat(stack.slice(k));
}

function pancakeSort(stack) {
    const n = stack.length;
    for (let size = n; size > 0; size--) {
        const maxIndex = stack.indexOf(size);
        if (maxIndex !== size - 1) {
            stack = flip(stack, maxIndex + 1);
            stack = flip(stack, size);
        }
    }
    return stack;
}

// Example usage:
const stack = [3, 1, 4, 2, 5];
const sortedStack = pancakeSort(stack);
console.log("Sorted Stack:", sortedStack);

const find_peak_element(nums):
left, right = 0, len(nums) - 1

while left < right:
    mid = (left + right) // 2
if nums[mid] > nums[mid + 1]:
    right = mid
else:
left = mid + 1

return left

//Example usage:
nums = [2, 3, -2, 4]
print("Maximum Product Subarray:", max_product_subarray(nums))

const { MinPriorityQueue, MaxPriorityQueue } = require('@datastructures-js/priority-queue');

class MedianFinder {
    constructor() {
        this.lo = new MaxPriorityQueue();  // Max heap
        this.hi = new MinPriorityQueue();  // Min heap
    }

    addNum(num) {
        this.lo.enqueue(num);
        if (this.lo.size() > 0 && this.hi.size() > 0 && this.lo.front().element > this.hi.front().element) {
            this.hi.enqueue(this.lo.dequeue().element);
        }
        if (this.lo.size() > this.hi.size() + 1) {
            this.hi.enqueue(this.lo.dequeue().element);
        }
        if (this.hi.size() > this.lo.size()) {
            this.lo.enqueue(this.hi.dequeue().element);
        }
    }

    findMedian() {
        if (this.lo.size() > this.hi.size()) {
            return this.lo.front().element;
        }
        return (this.hi.front().element + this.lo.front().element) / 2.0;
    }
}

// Example usage:
const mf = new MedianFinder();
mf.addNum(1);
mf.addNum

class RandomizedCollection {
    constructor() {
        this.vals = [];
        this.idx = new Map();
    }

    insert(val) {
        this.vals.push(val);
        const set = this.idx.get(val) || new Set();
        set.add(this.vals.length - 1);
        this.idx.set(val, set);
        return set.size === 1;
    }

    remove(val) {
        if (!this.idx.has(val) || this.idx.get(val).size === 0) {
            return false;
        }
        let removeIdx = Array.from(this.idx.get(val))[0];
        let lastElement = this.vals[this.vals.length - 1];
        this.vals[removeIdx] = lastElement;
        this.idx.get(val).delete(removeIdx);
        this.idx.get(lastElement).add(removeIdx);
        this.idx.get(lastElement).delete(this.vals.length - 1);
        this.vals.pop();
        return true;
    }

    getRandom() {
        return this.vals[Math.floor(Math.random() * this.vals.length)];
    }
}

// Example usage:
const obj = new RandomizedCollection();
console.log(obj.insert(1));  // True
console.log(obj.insert(1));  // False
console.log(obj.remove(1));  // True
console.log(obj.getRandom());  // Randomly get 1

class SnapshotArray {
    constructor(length) {
        this.array = Array.from({ length }, () => [[-1, 0]]);
        this.snapId = 0;
    }

    set(index, val) {
        const arr = this.array[index];
        if (arr[arr.length - 1][0] === this.snapId) {
            arr[arr.length - 1][1] = val;
        } else {
            arr.push([this.snapId, val]);
        }
    }

    snap() {
        return this.snapId++;
    }

    get(index, snapId) {
        const snapshots = this.array[index];
        let lo = 0, hi = snapshots.length - 1;
        while (lo < hi) {
            const mid = Math.floor((lo + hi + 1) / 2);
            if (snapshots[mid][0] <= snapId) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return snapshots[lo][1];
    }
}

// Example usage:
const snapArray = new SnapshotArray(3);
snapArray.set(0, 5);
const snapId = snapArray.snap();
console.log(snapArray.get(0, snapId));  // Output: 5

function removeDuplicates(nums) {
    if (nums.length === 0) return 0;
    let j = 0;
    for (let i = 1; i < nums.length; i++) {
        if (nums[i] !== nums[j]) {
            j++;
            nums[j] = nums[i];
        }
    }
    return j + 1;
}

// Test
const nums = [1, 1, 2, 2, 3, 4, 4, 5];
const length = removeDuplicates(nums);
console.log(length); // Output: 5
console.log(nums.slice(0, length)); // Output: [1, 2, 3, 4, 5]


class Node {
    constructor(value) {
        this.value = value;
        this.next = null;
    }
}

class LinkedList {
    constructor() {
        this.head = null;
        this.length = 0;
    }

    append(value) {
        const newNode = new Node(value);
        if (!this.head) {
            this.head = newNode;
        } else {
            let current = this.head;
            while (current.next) {
                current = current.next;
            }
            current.next = newNode;
        }
        this.length++;
    }

    prepend(value) {
        const newNode = new Node(value);
        newNode.next = this.head;
        this.head = newNode;
        this.length++;
    }

    delete(value) {
        if (!this.head) return;

        if (this.head.value === value) {
            this.head = this.head.next;
            this.length--;
            return;
        }

        let current = this.head;
        while (current.next && current.next.value !== value) {
            current = current.next;
        }

        if (current.next) {
            current.next = current.next.next;
            this.length--;
        }
    }

    find(value) {
        let current = this.head;
        while (current) {
            if (current.value === value) return current;
            current = current.next;
        }
        return null;
    }

    print() {
        let current = this.head;
        while (current) {
            console.log(current.value);
            current = current.next;
        }
    }
}

// Example usage:
const list = new LinkedList();
list.append(1);
list.append(2);
list.prepend(0);
list.print(); // Output: 0 1 2
list.delete(1);
list.print(); // Output: 0 2
console.log(list.find(2)); // Output: Node { value: 2, next: null }

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

    off(event, listenerToRemove) {
        if (!this.events[event]) return;
        this.events[event] = this.events[event].filter(listener => listener !== listenerToRemove);
    }

    emit(event, ...args) {
        if (!this.events[event]) return;
        this.events[event].forEach(listener => listener(...args));
    }
}

// Example usage:
const emitter = new EventEmitter();

function onFoo(data) {
    console.log('foo event:', data);
}

emitter.on('foo', onFoo);
emitter.emit('foo', { some: 'data' }); // Output: "foo event: { some: 'data' }"
emitter.off('foo', onFoo);
emitter.emit('foo', { some: 'data' }); // No output

class SimplePromise {
    constructor(executor) {
        this.state = 'pending';
        this.value = undefined;
        this.reason = undefined;
        this.onFulfilledCallbacks = [];
        this.onRejectedCallbacks = [];

        const resolve = value => {
            if (this.state === 'pending') {
                this.state = 'fulfilled';
                this.value = value;
                this.onFulfilledCallbacks.forEach(callback => callback(this.value));
            }
        };

        const reject = reason => {
            if (this.state === 'pending') {
                this.state = 'rejected';
                this.reason = reason;
                this.onRejectedCallbacks.forEach(callback => callback(this.reason));
            }
        };

        try {
            executor(resolve, reject);
        } catch (err) {
            reject(err);
        }
    }

    then(onFulfilled, onRejected) {
        return new SimplePromise((resolve, reject) => {
            if (this.state === 'fulfilled') {
                try {
                    const result = onFulfilled(this.value);
                    resolve(result);
                } catch (err) {
                    reject(err);
                }
            }

            if (this.state === 'rejected') {
                try {
                    const result = onRejected(this.reason);
                    resolve(result);
                } catch (err) {
                    reject(err);
                }
            }

            if (this.state === 'pending') {
                this.onFulfilledCallbacks.push(value => {
                    try {
                        const result = onFulfilled(value);
                        resolve(result);
                    } catch (err) {
                        reject(err);
                    }
                });

                this.onRejectedCallbacks.push(reason => {
                    try {
                        const result = onRejected(reason);
                        resolve(result);
                    } catch (err) {
                        reject(err);
                    }
                });
            }
        });
    }
}

// Example usage:
const promise = new SimplePromise((resolve, reject) => {
    setTimeout(() => resolve('Success!'), 1000);
});

promise.then(result => console.log(result)); // Output: "Success!"

function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        const arrCopy = [];
        obj.forEach((item, index) => {
            arrCopy[index] = deepClone(item);
        });
        return arrCopy;
    }

    const objCopy = {};
    Object.keys(obj).forEach(key => {
        objCopy[key] = deepClone(obj[key]);
    });
    return objCopy;
}

// Example usage:
const original = { a: 1, b: { c: 2 } };
const copied = deepClone(original);
console.log(copied); // Output: { a: 1, b: { c: 2 } }
console.log(copied.b === original.b); // Output: false

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// Example usage:
const debouncedFunction = debounce(() => console.log("Executed!"), 2000);
debouncedFunction();
debouncedFunction();
debouncedFunction(); // Only this call will execute the function after 2 seconds.

function fizzBuzz() {
    for (let i = 1; i <= 100; i++) {
        if (i % 3 === 0 && i % 5 === 0) {
            console.log("FizzBuzz");
        } else if (i % 3 === 0) {
            console.log("Fizz");
        } else if (i % 5 === 0) {
            console.log("Buzz");
        } else {
            console.log(i);
        }
    }
}

// Example usage:
fizzBuzz();

function findLargestNumber(arr) {
    return Math.max(...arr);
}

// Example usage:
console.log(findLargestNumber([1, 2, 3, 4, 5])); // Output: 5

function isPalindrome(str) {
    const cleanedStr = str.replace(/[^A-Za-z0-9]/g, '').toLowerCase();
    const reversedStr = cleanedStr.split('').reverse().join('');
    return cleanedStr === reversedStr;
}

// Example usage:
console.log(isPalindrome("A man, a plan, a canal, Panama")); // Output: true

function customPromiseAll(promises) {
    return new Promise((resolve, reject) => {
        let results = [];
        let completedPromises = 0;

        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then(result => {
                    results[index] = result;
                    completedPromises += 1;
                    if (completedPromises === promises.length) {
                        resolve(results);
                    }
                })
                .catch(error => reject(error));
        });
    });
}

function flattenArray(arr) {
    return arr.reduce((flat, toFlatten) => {
        return flat.concat(Array.isArray(toFlatten) ? flattenArray(toFlatten) : toFlatten);
    }, []);
}

function debounce(func, wait) {
    let timeout;
    return function (...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }

    return -1;
}

function twoSum(nums, target) {
    const map = new Map();
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (map.has(complement)) {
            return [map.get(complement), i];
        }
        map.set(nums[i], i);
    }
    return [];
}

// Example usage
console.log(twoSum([2, 7, 11, 15], 9)); // Output: [0, 1]

function lengthOfLongestSubstring(s) {
    let maxLen = 0;
    let start = 0;
    const map = new Map();

    for (let end = 0; end < s.length; end++) {
        if (map.has(s[end])) {
            start = Math.max(map.get(s[end]) + 1, start);
        }
        map.set(s[end], end);
        maxLen = Math.max(maxLen, end - start + 1);
    }

    return maxLen;
}

// Example usage
console.log(lengthOfLongestSubstring("abcabcbb")); // Output: 3

function isValid(s) {
    const stack = [];
    const map = {
        ')': '(',
        '}': '{',
        ']': '['
    };

    for (let i = 0; i < s.length; i++) {
        if (s[i] === '(' || s[i] === '{' || s[i] === '[') {
            stack.push(s[i]);
        } else {
            if (stack.pop() !== map[s[i]]) {
                return false;
            }
        }
    }

    return stack.length === 0;
}

// Example usage
console.log(isValid("()[]{}")); // Output: true
console.log(isValid("(]")); // Output: false

function merge(intervals) {
    if (intervals.length === 0) return [];

    intervals.sort((a, b) => a[0] - b[0]);
    const result = [intervals[0]];

    for (let i = 1; i < intervals.length; i++) {
        const current = intervals[i];
        const last = result[result.length - 1];

        if (current[0] <= last[1]) {
            last[1] = Math.max(last[1], current[1]);
        } else {
            result.push(current);
        }
    }

    return result;
}

// Example usage
console.log(merge([[1, 3], [2, 6], [8, 10], [15, 18]])); // Output: [[1, 6], [8, 10], [15, 18]]

class MedianFinder {
    constructor() {
        this.low = new MaxHeap();
        this.high = new MinHeap();
    }

    addNum(num) {
        if (this.low.size() === 0 || num < this.low.peek()) {
            this.low.insert(num);
        } else {
            this.high.insert(num);
        }

        if (this.low.size() > this.high.size() + 1) {
            this.high.insert(this.low.extractMax());
        } else if (this.high.size() > this.low.size()) {
            this.low.insert(this.high.extractMin());
        }
    }

    findMedian() {
        if (this.low.size() > this.high.size()) {
            return this.low.peek();
        } else {
            return (this.low.peek() + this.high.peek()) / 2;
        }
    }
}

class MaxHeap {
    constructor() {
        this.heap = [];
    }

    size() {
        return this.heap.length;
    }

    peek() {
        return this.heap[0];
    }

    insert(val) {
        this.heap.push(val);
        this._heapifyUp();
    }

    extractMax() {
        const max = this.heap[0];
        const end = this.heap.pop();
        if (this.size() > 0) {
            this.heap[0] = end;
            this._heapifyDown();
        }
        return max;
    }

    _heapifyUp() {
        let idx = this.size() - 1;
        const element = this.heap[idx];

        while (idx > 0) {
            const parentIdx = Math.floor((idx - 1) / 2);
            const parent = this.heap[parentIdx];

            if (element <= parent) break;

            this.heap[idx] = parent;
            this.heap[parentIdx] = element;
            idx = parentIdx;
        }
    }

    _heapifyDown() {
        let idx = 0;
        const length = this.size();
        const element = this.heap[idx];

        while (true) {
            const leftChildIdx = 2 * idx + 1;
            const rightChildIdx = 2 * idx + 2;
            let leftChild, rightChild;
            let swap = null;

            if (leftChildIdx < length) {
                leftChild = this.heap[leftChildIdx];
                if (leftChild > element) {
                    swap = leftChildIdx;
                }
            }

            if (rightChildIdx < length) {
                rightChild = this.heap[rightChildIdx];
                if ((swap === null && rightChild > element) || (swap !== null && rightChild > leftChild)) {
                    swap = rightChildIdx;
                }
            }

            if (swap === null) break;

            this.heap[idx] = this.heap[swap];
            this.heap[swap] = element;
            idx = swap;
        }
    }
}

class MinHeap {
    constructor() {
        this.heap = [];
    }

    size() {
        return this.heap.length;
    }

    peek() {
        return this.heap[0];
    }

    insert(val) {
        this.heap.push(val);
        this._heapifyUp();
    }

    extractMin() {
        const min = this.heap[0];
        const end = this.heap.pop();
        if (this.size() > 0) {
            this.heap[0] = end;
            this._heapifyDown();
        }
        return min;
    }

    _heapifyUp() {
        let idx = this.size() - 1;
        const element = this.heap[idx];

        while (idx > 0) {
            const parentIdx = Math.floor((idx - 1) / 2);
            const parent = this.heap[parentIdx];

            if (element >= parent) break;

            this.heap[idx] = parent;
            this.heap[parentIdx] = element;
            idx = parentIdx;
        }
    }

    _heapifyDown() {
        let idx = 0;
        const length = this.size();
        const element = this.heap[idx];

        while (true) {
            const leftChildIdx = 2 * idx + 1;
            const rightChildIdx = 2 * idx + 2;
            let leftChild, rightChild;
            let swap = null;

            if (leftChildIdx < length) {
                leftChild = this.heap[leftChildIdx];
                if (leftChild < element) {
                    swap = leftChildIdx;
                }
            }

            if (rightChildIdx < length) {
                rightChild = this.heap[rightChildIdx];
                if ((swap === null && rightChild < element) || (swap !== null && rightChild < leftChild)) {
                    swap = rightChildIdx;
                }
            }

            if (swap === null) break;

            this.heap[idx] = this.heap[swap];
            this.heap[swap] = element;
            idx = swap;
        }
    }
}

// Example usage
const mf = new MedianFinder();
mf.addNum(1);
mf.addNum(2);
console.log(mf.findMedian()); // Output: 1.5
mf.addNum(3);
console.log(mf.findMedian()); // Output: 2

function isValid(s) {
    const stack = [];
    const map = {
        '(': ')',
        '{': '}',
        '[': ']'
    };

    for (let i = 0; i < s.length; i++) {
        if (map[s[i]]) {
            stack.push(map[s[i]]);
        } else if (stack.length > 0 && stack[stack.length - 1] === s[i]) {
            stack.pop();
        } else {
            return false;
        }
    }

    return stack.length === 0;
}

// Example usage:
console.log(isValid("()")); // Output: true
console.log(isValid("()[]{}")); // Output: true
console.log(isValid("(]")); // Output: false

function ListNode(val, next = null) {
    this.val = val;
    this.next = next;
}

function mergeTwoLists(l1, l2) {
    const dummy = new ListNode(-1);
    let current = dummy;

    while (l1 !== null && l2 !== null) {
        if (l1.val <= l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }

    current.next = l1 === null ? l2 : l1;
    return dummy.next;
}

// Example usage:
const list1 = new ListNode(1, new ListNode(2, new ListNode(4)));
const list2 = new ListNode(1, new ListNode(3, new ListNode(4)));
console.log(mergeTwoLists(list1, list2)); // Output: [1, 1, 2, 3, 4, 4]

function lengthOfLongestSubstring(s) {
    const charMap = new Map();
    let left = 0;
    let maxLength = 0;

    for (let right = 0; right < s.length; right++) {
        if (charMap.has(s[right])) {
            left = Math.max(left, charMap.get(s[right]) + 1);
        }
        charMap.set(s[right], right);
        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
}

// Example usage:
console.log(lengthOfLongestSubstring("abcabcbb")); // Output: 3
console.log(lengthOfLongestSubstring("bbbbb")); // Output: 1

class GeneticAlgorithmTSP {
    constructor(cities, populationSize, mutationRate, generations) {
        this.cities = cities;
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.generations = generations;
        this.population = this.initializePopulation();
    }

    initializePopulation() {
        let population = [];
        for (let i = 0; i < this.populationSize; i++) {
            let route = [...this.cities].sort(() => Math.random() - 0.5);
            population.push(route);
        }
        return population;
    }

    fitness(route) {
        return route.reduce((acc, city, i) => {
            if (i === route.length - 1) return acc;
            return acc + Math.hypot(city[0] - route[i + 1][0], city[1] - route[i + 1][1]);
        }, 0);
    }

    selection() {
        return this.population.sort((a, b) => this.fitness(a) - this.fitness(b)).slice(0, this.populationSize / 2);
    }

    crossover(parent1, parent2) {
        let crossoverPoint = Math.floor(Math.random() * parent1.length);
        let child = parent1.slice(0, crossoverPoint).concat(parent2.filter(city => !parent1.slice(0, crossoverPoint).includes(city)));
        return child;
    }

    mutate(route) {
        if (Math.random() < this.mutationRate) {
            let [i, j] = [Math.floor(Math.random() * route.length), Math.floor(Math.random() * route.length)];
            [route[i], route[j]] = [route[j], route[i]];
        }
        return route;
    }

    evolve() {
        for (let generation = 0; generation < this.generations; generation++) {
            let newPopulation = [];
            let selected = this.selection();
            for (let i = 0; i < selected.length; i += 2) {
                let parent1 = selected[i];
                let parent2 = selected[i + 1];
                newPopulation.push(this.mutate(this.crossover(parent1, parent2)));
                newPopulation.push(this.mutate(this.crossover(parent2, parent1)));
            }
            this.population = newPopulation;
        }
    }

    getBestRoute() {
        return this.population.reduce((best, route) => this.fitness(route) < this.fitness(best) ? route : best);
    }
}

function longestSubstringWithoutRepeatingCharacters(s) {
    let left = 0;
    let right = 0;
    let maxLength = 0;
    let charSet = new Set();

    while (right < s.length) {
        if (!charSet.has(s[right])) {
            charSet.add(s[right]);
            right++;
            maxLength = Math.max(maxLength, right - left);
        } else {
            charSet.delete(s[left]);
            left++;
        }
    }

    return maxLength;
}

// Test cases
console.log(longestSubstringWithoutRepeatingCharacters("abcabcbb")); // Output: 3
console.log(longestSubstringWithoutRepeatingCharacters("bbbbb"));    // Output: 1
console.log(longestSubstringWithoutRepeatingCharacters("pwwkew"));   // Output: 3
console.log(longestSubstringWithoutRepeatingCharacters(""));         // Output: 0
console.log(longestSubstringWithoutRepeatingCharacters("dvdf"));     // Output: 3

function firstMissingPositive(nums) {
    let n = nums.length;

    for (let i = 0; i < n; i++) {
        while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] !== nums[i]) {
            [nums[nums[i] - 1], nums[i]] = [nums[i], nums[nums[i] - 1]];
        }
    }

    for (let i = 0; i < n; i++) {
        if (nums[i] !== i + 1) {
            return i + 1;
        }
    }

    return n + 1;
}

// Test cases
console.log(firstMissingPositive([1, 2, 0]));    // Output: 3
console.log(firstMissingPositive([3, 4, -1, 1])); // Output: 2
console.log(firstMissingPositive([7, 8, 9, 11, 12])); // Output: 1

function permute(nums) {
    let results = [];

    function backtrack(start) {
        if (start === nums.length) {
            results.push([...nums]);
            return;
        }

        for (let i = start; i < nums.length; i++) {
            [nums[start], nums[i]] = [nums[i], nums[start]];
            backtrack(start + 1);
            [nums[start], nums[i]] = [nums[i], nums[start]];
        }
    }

    backtrack(0);
    return results;
}

// Test case
console.log(permute([1, 2, 3]));
// Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,2,1],[3,1,2]]

function findMedianSortedArrays(nums1, nums2) {
    if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);

    let x = nums1.length;
    let y = nums2.length;

    let low = 0, high = x;

    while (low <= high) {
        let partitionX = (low + high) >> 1;
        let partitionY = ((x + y + 1) >> 1) - partitionX;

        let maxX = (partitionX === 0) ? -Infinity : nums1[partitionX - 1];
        let maxY = (partitionY === 0) ? -Infinity : nums2[partitionY - 1];

        let minX = (partitionX === x) ? Infinity : nums1[partitionX];
        let minY = (partitionY === y) ? Infinity : nums2[partitionY];

        if (maxX <= minY && maxY <= minX) {
            if ((x + y) % 2 === 0) {
                return (Math.max(maxX, maxY) + Math.min(minX, minY)) / 2;
            } else {
                return Math.max(maxX, maxY);
            }
        } else if (maxX > minY) {
            high = partitionX - 1;
        } else {
            low = partitionX + 1;
        }
    }

    throw new Error("Input arrays are not sorted");
}

// Test cases
console.log(findMedianSortedArrays([1, 3], [2])); // Output: 2
console.log(findMedianSortedArrays([1, 2], [3, 4])); // Output: 2.5

class ListNode {
    constructor(key, value) {
        this.key = key;
        this.value = value;
        this.prev = null;
        this.next = null;
    }
}

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.map = new Map();
        this.head = new ListNode();
        this.tail = new ListNode();
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    get(key) {
        if (!this.map.has(key)) return -1;

        let node = this.map.get(key);
        this._remove(node);
        this._add(node);

        return node.value;
    }

    put(key, value) {
        if (this.map.has(key)) {
            this._remove(this.map.get(key));
        }

        let node = new ListNode(key, value);
        this._add(node);
        this.map.set(key, node);

        if (this.map.size > this.capacity) {
            let lru = this.tail.prev;
            this._remove(lru);
            this.map.delete(lru.key);
        }
    }

    _remove(node) {
        let prev = node.prev;
        let next = node.next;
        prev.next = next;
        next.prev = prev;
    }

    _add(node) {
        let next = this.head.next;
        this.head.next = node;
        node.prev = this.head;
        node.next = next;
        next.prev = node;
    }
}

// Test case
let lruCache = new LRUCache(2);
lruCache.put(1, 1);
lruCache.put(2, 2);
console.log(lruCache.get(1)); // Output: 1
lruCache.put(3, 3); // LRU key was 2, evicts key 2
console.log(lruCache.get(2)); // Output: -1
lruCache.put(4, 4); // LRU key was 1, evicts key 1
console.log(lruCache.get(1)); // Output: -1
console.log(lruCache.get(3)); // Output: 3
console.log(lruCache.get(4)); // Output: 4

function exist(board, word) {
    const rows = board.length;
    const cols = board[0].length;

    function dfs(r, c, i) {
        if (i === word.length) return true;
        if (r < 0 || c < 0 || r >= rows || c >= cols || board[r][c] !== word[i]) return false;

        const temp = board[r][c];
        board[r][c] = '#';
        const found = dfs(r + 1, c, i + 1) || dfs(r - 1, c, i + 1) || dfs(r, c + 1, i + 1) || dfs(r, c - 1, i + 1);
        board[r][c] = temp;
        return found;
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (board[r][c] === word[0] && dfs(r, c, 0)) {
                return true;
            }
        }
    }

    return false;
}

// Example usage:
const board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
];
console.log(exist(board, "ABCCED")); // Output: true
console.log(exist(board, "SEE")); // Output: true
console.log(exist(board, "ABCB")); // Output: false

function letterCombinations(digits) {
    if (!digits) return [];

    const map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    };

    const result = [];

    function backtrack(index, current) {
        if (index === digits.length) {
            result.push(current.join(''));
            return;
        }

        const letters = map[digits[index]];
        for (let letter of letters) {
            current.push(letter);
            backtrack(index + 1, current);
            current.pop();
        }
    }

    backtrack(0, []);
    return result;
}

// Example usage:
console.log(letterCombinations("23"));
// Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

function longestPalindrome(s) {
    if (s.length <= 1) return s;

    let start = 0, maxLength = 1;

    function expandAroundCenter(left, right) {
        while (left >= 0 && right < s.length && s[left] === s[right]) {
            if (right - left + 1 > maxLength) {
                start = left;
                maxLength = right - left + 1;
            }
            left--;
            right++;
        }
    }

    for (let i = 0; i < s.length; i++) {
        expandAroundCenter(i, i); // Odd length
        expandAroundCenter(i, i + 1); // Even length
    }

    return s.substring(start, start + maxLength);
}

// Example usage:
console.log(longestPalindrome("babad")); // Output: "bab" or "aba"
console.log(longestPalindrome("cbbd")); // Output: "bb"

function groupAnagrams(strs) {
    const map = new Map();

    for (let str of strs) {
        const sorted = str.split('').sort().join('');
        if (!map.has(sorted)) {
            map.set(sorted, []);
        }
        map.get(sorted).push(str);
    }

    return Array.from(map.values());
}

// Example usage:
console.log(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]));
// Output: [["eat","tea","ate"],["tan","nat"],["bat"]]

function rotate(matrix) {
    const n = matrix.length;

    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            [matrix[i][j], matrix[j][i]] = [matrix[j][i], matrix[i][j]];
        }
    }

    for (let i = 0; i < n; i++) {
        matrix[i].reverse();
    }
}

// Example usage:
const matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];
rotate(matrix);
console.log(matrix);
// Output: [[7,4,1],[8,5,2],[9,6,3]]

function computeLPSArray(pattern) {
    const lps = Array(pattern.length).fill(0);
    let length = 0;
    let i = 1;

    while (i < pattern.length) {
        if (pattern[i] === pattern[length]) {
            length++;
            lps[i] = length;
            i++;
        } else {
            if (length !== 0) {
                length = lps[length - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

function KMPSearch(text, pattern) {
    const lps = computeLPSArray(pattern);
    const result = [];
    let i = 0; // index for text
    let j = 0; // index for pattern

    while (i < text.length) {
        if (pattern[j] === text[i]) {
            i++;
            j++;
        }

        if (j === pattern.length) {
            result.push(i - j);
            j = lps[j - 1];
        } else if (i < text.length && pattern[j] !== text[i]) {
            if (j !== 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    return result;
}

console.log(KMPSearch("ababcabcabababd", "ababd")); // Output: [10]

function longestCommonSubsequence(str1, str2) {
    const m = str1.length;
    const n = str2.length;
    const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (str1[i - 1] === str2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}

console.log(longestCommonSubsequence("ABCBDAB", "BDCAB")); // Output: 4

class TrieNode {
    constructor() {
        this.children = {};
        this.isEndOfWord = false;
    }
}

class Trie {
    constructor() {
        this.root = new TrieNode();
    }

    insert(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                node.children[char] = new TrieNode();
            }
            node = node.children[char];
        }
        node.isEndOfWord = true;
    }

    search(word) {
        let node = this.root;
        for (const char of word) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return node.isEndOfWord;
    }

    startsWith(prefix) {
        let node = this.root;
        for (const char of prefix) {
            if (!node.children[char]) {
                return false;
            }
            node = node.children[char];
        }
        return true;
    }
}

const trie = new Trie();
trie.insert("apple");
console.log(trie.search("apple"));   // Output: true
console.log(trie.search("app"));     // Output: false
console.log(trie.startsWith("app")); // Output: true
trie.insert("app");
console.log(trie.search("app"));     // Output: true

function middleNode(head) {
    let slow = head;
    let fast = head;

    while (fast !== null && fast.next !== null) {
        slow = slow.next;
        fast = fast.next.next;
    }

    return slow;
}

// Example usage:
// Input: 1 -> 2 -> 3 -> 4 -> 5
// Output: 3 -> 4 -> 5
let head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
console.log(middleNode(head)); // Output: ListNode { val: 3, next: ListNode { val: 4, next: [ListNode] } }

function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function mergeTwoLists(l1, l2) {
    let dummy = new ListNode();
    let current = dummy;

    while (l1 !== null && l2 !== null) {
        if (l1.val < l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }

    if (l1 !== null) {
        current.next = l1;
    } else {
        current.next = l2;
    }

    return dummy.next;
}

// Example usage:
// l1: 1 -> 2 -> 4
// l2: 1 -> 3 -> 4
let l1 = new ListNode(1, new ListNode(2, new ListNode(4)));
let l2 = new ListNode(1, new ListNode(3, new ListNode(4)));
let merged = mergeTwoLists(l1, l2);
// Output: 1 -> 1 -> 2 -> 3 -> 4 -> 4

function isValid(s) {
    const stack = [];
    const map = {
        '(': ')',
        '{': '}',
        '[': ']'
    };

    for (let char of s) {
        if (map[char]) {
            stack.push(map[char]);
        } else if (stack.length > 0 && stack[stack.length - 1] === char) {
            stack.pop();
        } else {
            return false;
        }
    }

    return stack.length === 0;
}

// Example usage:
console.log(isValid("()[]{}")); // Output: true
console.log(isValid("(]"));     // Output: false

function longestPalindromicSubstring(s) {
    if (!s) return "";

    let start = 0, end = 0;

    for (let i = 0; i < s.length; i++) {
        let len1 = expandAroundCenter(s, i, i);
        let len2 = expandAroundCenter(s, i, i + 1);
        let maxLen = Math.max(len1, len2);
        if (maxLen > end - start) {
            start = i - Math.floor((maxLen - 1) / 2);
            end = i + Math.floor(maxLen / 2);
        }
    }

    return s.substring(start, end + 1);
}

function expandAroundCenter(s, left, right) {
    while (left >= 0 && right < s.length && s[left] === s[right]) {
        left--;
        right++;
    }
    return right - left - 1;
}

// Example usage:
console.log(longestPalindromicSubstring("babad")); // Output: "bab" or "aba"

function mergeIntervals(intervals) {
    if (!intervals.length) return [];

    intervals.sort((a, b) => a[0] - b[0]);
    let merged = [intervals[0]];

    for (let i = 1; i < intervals.length; i++) {
        let last = merged[merged.length - 1];
        let current = intervals[i];
        if (current[0] <= last[1]) {
            last[1] = Math.max(last[1], current[1]);
        } else {
            merged.push(current);
        }
    }

    return merged;
}

// Example usage:
console.log(mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]])); // Output: [[1, 6], [8, 10], [15, 18]]


function findMin(nums) {
    let left = 0, right = nums.length - 1;

    while (left < right) {
        let mid = Math.floor((left + right) / 2);
        if (nums[mid] > nums[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return nums[left];
}

// Example usage:
console.log(findMin([3, 4, 5, 1, 2])); // Output: 1

function exist(board, word) {
    function dfs(board, word, i, j, k) {
        if (k === word.length) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] !== word[k]) return false;

        let tmp = board[i][j];
        board[i][j] = '/';
        let res = dfs(board, word, i + 1, j, k + 1) ||
            dfs(board, word, i - 1, j, k + 1) ||
            dfs(board, word, i, j + 1, k + 1) ||
            dfs(board, word, i, j - 1, k + 1);
        board[i][j] = tmp;
        return res;
    }

    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[0].length; j++) {
            if (dfs(board, word, i, j, 0)) return true;
        }
    }
    return false;
}

// Example usage:
let board = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
];
let word = "ABCCED";
console.log(exist(board, word)); // Output: true

function trap(height) {
    if (!height.length) return 0;

    let left = 0, right = height.length - 1;
    let leftMax = height[left], rightMax = height[right];
    let waterTrapped = 0;

    while (left < right) {
        if (leftMax < rightMax) {
            left++;
            leftMax = Math.max(leftMax, height[left]);
            waterTrapped += leftMax - height[left];
        } else {
            right--;
            rightMax = Math.max(rightMax, height[right]);
            waterTrapped += rightMax - height[right];
        }
    }

    return waterTrapped;
}

// Example usage:
console.log(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])); // Output: 6

function minWindow(s, t) {
    if (!s || !t) return "";

    const dictT = {};
    for (let char of t) {
        dictT[char] = (dictT[char] || 0) + 1;
    }
    let required = Object.keys(dictT).length;

    let l = 0, r = 0;
    let formed = 0;
    const windowCounts = {};
    let ans = [-1, 0, 0];

    while (r < s.length) {
        let char = s[r];
        windowCounts[char] = (windowCounts[char] || 0) + 1;

        if (char in dictT && windowCounts[char] === dictT[char]) {
            formed++;
        }

        while (l <= r && formed === required) {
            char = s[l];

            if (ans[0] === -1 || r - l + 1 < ans[0]) {
                ans = [r - l + 1, l, r];
            }

            windowCounts[char]--;
            if (char in dictT && windowCounts[char] < dictT[char]) {
                formed--;
            }

            l++;
        }

        r++;
    }

    return ans[0] === -1 ? "" : s.slice(ans[1], ans[2] + 1);
}

// Example usage:
const s = "ADOBECODEBANC";
const t = "ABC";
console.log(minWindow(s, t));  // Output: "BANC"

function findMissingNumber(arr) {
    const n = arr.length;
    const total = (n * (n + 1)) / 2;
    const sum = arr.reduce((acc, num) => acc + num, 0);
    return total - sum;
}

console.log(findMissingNumber([0, 1, 3, 4, 5])); // Output: 2

function flattenArray(arr) {
    return arr.reduce((flat, toFlatten) => flat.concat(Array.isArray(toFlatten) ? flattenArray(toFlatten) : toFlatten), []);
}

console.log(flattenArray([1, [2, [3, [4, 5]]], 6])); // Output: [1, 2, 3, 4, 5, 6]

function fizzBuzz() {
    for (let i = 1; i <= 100; i++) {
        if (i % 3 === 0 && i % 5 === 0) console.log("FizzBuzz");
        else if (i % 3 === 0) console.log("Fizz");
        else if (i % 5 === 0) console.log("Buzz");
        else console.log(i);
    }
}

fizzBuzz();

function removeDuplicates(arr) {
    return [...new Set(arr)];
}

console.log(removeDuplicates([1, 2, 2, 3, 4, 4, 5])); // Output: [1, 2, 3, 4, 5]

function capitalizeWords(str) {
    return str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

console.log(capitalizeWords("hello world")); // Output: "Hello World"

function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }

    return -1;
}

console.log(binarySearch([1, 3, 5, 7, 9], 5)); // Output: 2

function findLargest(arr) {
    return Math.max(...arr);
}

console.log(findLargest([1, 5, 3, 9, 2])); // Output: 9

function isPalindrome(str) {
    const reversed = str.split('').reverse().join('');
    return str === reversed;
}

console.log(isPalindrome("racecar")); // Output: true

function factorial(n) {
    if (n === 0 || n === 1) return 1;
    return n * factorial(n - 1);
}

console.log(factorial(5)); // Output: 120

function reverseString(str) {
    return str.split('').reverse().join('');
}

console.log(reverseString("hello")); // Output: "olleh"

function isPalindrome(str) {
    const sanitizedStr = str.toLowerCase().replace(/[^a-z0-9]/g, '');
    return sanitizedStr === sanitizedStr.split('').reverse().join('');
}

// Example
console.log(isPalindrome("A man, a plan, a canal: Panama")); // true

function fibonacci(n) {
    const fib = [0, 1];
    for (let i = 2; i <= n; i++) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib.slice(1); // Exclude the initial zero for a cleaner output
}

// Example
console.log(fibonacci(10)); // [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

function areAnagrams(str1, str2) {
    const normalize = str => str.toLowerCase().replace(/[^a-z0-9]/g, '').split('').sort().join('');
    return normalize(str1) === normalize(str2);
}

// Example
console.log(areAnagrams('listen', 'silent')); // true

function flattenArray(arr) {
    return arr.reduce((flat, toFlatten) => flat.concat(Array.isArray(toFlatten) ? flattenArray(toFlatten) : toFlatten), []);
}

// Example
console.log(flattenArray([1, [2, [3, 4], 5], [6, 7]])); // [1, 2, 3, 4, 5, 6, 7]

function longestWord(sentence) {
    const words = sentence.split(' ');
    return words.reduce((longest, currentWord) => currentWord.length > longest.length ? currentWord : longest, '');
}

// Example
console.log(longestWord("The quick brown fox jumps over the lazy dog")); // "jumps"

function removeDuplicates(arr) {
    return [...new Set(arr)];
}

// Example
console.log(removeDuplicates([1, 2, 2, 3, 4, 4, 5])); // [1, 2, 3, 4, 5]

function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1; // target not found
}

// Example
console.log(binarySearch([1, 2, 3, 4, 5, 6, 7], 5)); // 4

function hasUniqueChars(str) {
    const charSet = new Set();
    for (let char of str) {
        if (charSet.has(char)) return false;
        charSet.add(char);
    }
    return true;
}

// Example
console.log(hasUniqueChars("abcdef")); // true
console.log(hasUniqueChars("aabbcc")); // false

function rotateArray(arr, k) {
    k = k % arr.length;
    return arr.slice(-k).concat(arr.slice(0, -k));
}

// Example
console.log(rotateArray([1, 2, 3, 4, 5], 2)); // [4, 5, 1, 2, 3]

function numIslands(grid) {
    if (!grid.length) return 0;

    let count = 0;

    function dfs(i, j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] === '0') return;
        grid[i][j] = '0'; // mark as visited
        dfs(i + 1, j);
        dfs(i - 1, j);
        dfs(i, j + 1);
        dfs(i, j - 1);
    }

    for (let i = 0; i < grid.length; i++) {
        for (let j = 0; j < grid[0].length; j++) {
            if (grid[i][j] === '1') {
                count++;
                dfs(i, j);
            }
        }
    }

    return count;
}

const grid = [
    ['1', '1', '0', '0', '0'],
    ['1', '1', '0', '0', '0'],
    ['0', '0', '1', '0', '0'],
    ['0', '0', '0', '1', '1']
];

console.log(numIslands(grid)); // Output: 3

function fizzBuzz(n) {
    return Array.from({ length: n }, (_, i) =>
        (i % 3 ? "" : "Fizz") + (i % 5 ? "" : "Buzz") || i + 1
    );
}

console.log(fizzBuzz(15)); // Output: [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]

function findDuplicates(nums) {
    const seen = new Set();
    const duplicates = new Set();

    for (let num of nums) {
        if (seen.has(num)) {
            duplicates.add(num);
        } else {
            seen.add(num);
        }
    }

    return [...duplicates];
}

console.log(findDuplicates([1, 2, 3, 4, 3, 2, 1])); // Output: [3, 2, 1]

class LRUCache {
    constructor(limit) {
        this.cache = new Map();
        this.limit = limit;
    }

    get(key) {
        if (!this.cache.has(key)) return -1;
        const value = this.cache.get(key);
        this.cache.delete(key);
        this.cache.set(key, value);
        return value;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size === this.limit) {
            this.cache.delete(this.cache.keys().next().value);
        }
        this.cache.set(key, value);
    }
}

const lru = new LRUCache(2);
lru.set(1, 1);
lru.set(2, 2);
console.log(lru.get(1)); // Output: 1
lru.set(3, 3);
console.log(lru.get(2)); // Output: -1

function flatten(arr) {
    let flatArray = [];

    arr.forEach(item => {
        if (Array.isArray(item)) {
            flatArray = flatArray.concat(flatten(item));
        } else {
            flatArray.push(item);
        }
    });

    return flatArray;
}

console.log(flatten([1, [2, [3, [4]], 5]])); // Output: [1, 2, 3, 4, 5]

function firstNonRepeatingChar(str) {
    const count = {};

    for (let char of str) count[char] = (count[char] || 0) + 1;

    for (let char of str) {
        if (count[char] === 1) return char;
    }

    return null;
}

console.log(firstNonRepeatingChar("swiss")); // Output: "w"

function permute(str) {
    if (str.length <= 1) return [str];
    let permutations = [];
    for (let i = 0; i < str.length; i++) {
        let char = str[i];
        let remaining = str.slice(0, i) + str.slice(i + 1);
        for (let perm of permute(remaining)) {
            permutations.push(char + perm);
        }
    }
    return permutations;
}

console.log(permute("abc")); // Output: ["abc", "acb", "bac", "bca", "cab", "cba"]

function areAnagrams(str1, str2) {
    if (str1.length !== str2.length) return false;

    const count = {};

    for (let char of str1) count[char] = (count[char] || 0) + 1;
    for (let char of str2) {
        if (!count[char]) return false;
        count[char]--;
    }

    return true;
}

console.log(areAnagrams("listen", "silent")); // Output: true

function maxSubArray(nums) {
    let maxSoFar = nums[0];
    let maxEndingHere = nums[0];
    for (let i = 1; i < nums.length; i++) {
        maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    return maxSoFar;
}

console.log(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4])); // Output: 6

function reverseString(str) {
    let reversed = '';
    for (let char of str) {
        reversed = char + reversed;
    }
    return reversed;
}

console.log(reverseString("hello")); // Output: "olleh"

function reverseString(str) {
    return str.split('').reverse().join('');
}

// Example
console.log(reverseString('hello')); // Output: 'olleh'

function findMax(arr) {
    return Math.max(...arr);
}

// Example
console.log(findMax([1, 5, 3, 9, 2])); // Output: 9

function isPalindrome(str) {
    const cleanedStr = str.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    return cleanedStr === cleanedStr.split('').reverse().join('');
}

// Example
console.log(isPalindrome('A man, a plan, a canal, Panama')); // Output: true

function factorial(num) {
    if (num <= 1) return 1;
    return num * factorial(num - 1);
}

// Example
console.log(factorial(5)); // Output: 120

function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Example
console.log(fibonacci(6)); // Output: 8

function removeDuplicates(arr) {
    return [...new Set(arr)];
}

// Example
console.log(removeDuplicates([1, 2, 3, 3, 4, 5, 5])); // Output: [1, 2, 3, 4, 5]

function secondLargest(arr) {
    let max = -Infinity, secondMax = -Infinity;
    for (let num of arr) {
        if (num > max) {
            secondMax = max;
            max = num;
        } else if (num > secondMax && num !== max) {
            secondMax = num;
        }
    }
    return secondMax;
}

// Example
console.log(secondLargest([10, 5, 8, 12, 7])); // Output: 10

function removeDuplicates(arr) {
    return [...new Set(arr)];
}

// Example
console.log(removeDuplicates([1, 2, 3, 3, 4, 5, 5])); // Output: [1, 2, 3, 4, 5]
