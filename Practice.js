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
