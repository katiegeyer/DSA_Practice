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
var generateParenthesis = function(n) {
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

var divide = function(dividend, divisor) {
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

document.addEventListener('DOMContentLoaded', function() {
    // This function runs when the DOM is fully loaded.

    // Get references to the DOM elements.
    const textElement = document.getElementById('textElement');
    const changeTextButton = document.getElementById('changeTextButton');

    // Add an event listener to the button for the 'click' event.
    changeTextButton.addEventListener('click', function() {
        // Change the text and color of the paragraph element.
        textElement.textContent = 'The text has been changed!';
        textElement.style.color = 'blue';
    });
});
