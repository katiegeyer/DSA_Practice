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
