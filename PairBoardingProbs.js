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
