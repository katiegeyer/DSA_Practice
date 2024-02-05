//make a git commit mock
// Mock repository: Array of commit objects
let repository = [];

// Function to create a new commit
function commit(message) {
    const commit = {
        id: generateCommitId(),
        timestamp: new Date().toISOString(),
        message: message,
    };
    repository.push(commit);
    console.log(`Commit successful: ${commit.id} - ${commit.message}`);
}

// Helper function to generate a mock commit ID (simple UUID-like string for demo)
function generateCommitId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Example usage
commit("Initial commit");
commit("Add README file");
commit("Fix bug in the login feature");

// Function to log all commits in the repository
function logCommits() {
    console.log("Repository Commit Log:");
    repository.forEach(commit => {
        console.log(`${commit.id} | ${commit.timestamp} | ${commit.message}`);
    });
}

// Call logCommits to see all the commits made
logCommits();

