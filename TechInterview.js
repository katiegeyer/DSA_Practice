function mergeIntervals(intervals) {
    intervals.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (let interval of intervals) {
        if (!merged.length || merged[merged.length - 1][1] < interval[0]) {
            merged.push(interval);
        } else {
            merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], interval[1]);
        }
    }
    return merged;
}

// Example usage
console.log(mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]]));  // Output: [[1,6],[8,10],[15,18]]

class TreeNode {
    constructor(val = 0, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function isValidBST(root, low = -Infinity, high = Infinity) {
    if (!root) return true;
    if (root.val <= low || root.val >= high) return false;
    return isValidBST(root.left, low, root.val) && isValidBST(root.right, root.val, high);
}

// Example requires building a binary tree
