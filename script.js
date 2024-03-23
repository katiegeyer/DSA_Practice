
document.getElementById('askButton').addEventListener('click', function () {
    var answers = [
        "It is certain.",
        "It is decidedly so.",
        "Without a doubt.",
        "Yes - definitely.",
        "You may rely on it.",
        "As I see it, yes.",
        "Most likely.",
        "Outlook good.",
        "Yes.",
        "Signs point to yes.",
        "Reply hazy, try again.",
        "Ask again later.",
        "Better not tell you now.",
        "Cannot predict now.",
        "Concentrate and ask again.",
        "Don't count on it.",
        "My reply is no.",
        "My sources say no.",
        "Outlook not so good.",
        "Very doubtful."
    ];

    var question = document.getElementById('question').value.trim();
    if (question === "") {
        alert("Please ask a question.");
        return;
    }

    if (!question.endsWith('?')) {
        alert("Questions usually end with a '?'. Please try again.");
        return;
    }

    var answer = answers[Math.floor(Math.random() * answers.length)];
    document.getElementById('answer').innerText = answer;
});
