const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const basket = {
    x: canvas.width / 2 - 30,
    y: canvas.height - 30,
    width: 60,
    height: 20,
    dx: 7
};

let items = [];
let score = 0;
let missed = 0;
let gameInterval;
let itemInterval;

function drawBasket() {
    ctx.fillStyle = '#008080';
    ctx.fillRect(basket.x, basket.y, basket.width, basket.height);
}

function drawItems() {
    ctx.fillStyle = '#ff6347';
    items.forEach(item => {
        ctx.beginPath();
        ctx.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.closePath();
    });
}

function drawScore() {
    ctx.fillStyle = '#000';
    ctx.font = '20px Arial';
    ctx.fillText('Score: ' + score, 10, 20);
    ctx.fillText('Missed: ' + missed, 10, 40);
}

function updateItems() {
    items.forEach((item, index) => {
        item.y += item.dy;
        if (item.y + item.radius > basket.y && item.x > basket.x && item.x < basket.x + basket.width) {
            items.splice(index, 1);
            score++;
        } else if (item.y > canvas.height) {
            items.splice(index, 1);
            missed++;
            if (missed >= 5) {
                clearInterval(gameInterval);
                clearInterval(itemInterval);
                alert('Game Over! Your score: ' + score);
            }
        }
    });
}

function moveBasket() {
    if (basket.rightPressed && basket.x + basket.width < canvas.width) {
        basket.x += basket.dx;
    }
    if (basket.leftPressed && basket.x > 0) {
        basket.x -= basket.dx;
    }
}

function spawnItem() {
    const radius = 10;
    const x = Math.random() * (canvas.width - radius * 2) + radius;
    const y = 0;
    const dy = Math.random() * 3 + 2;

    items.push({ x, y, radius, dy });
}

function gameLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBasket();
    drawItems();
    drawScore();
    moveBasket();
    updateItems();
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') {
        basket.rightPressed = true;
    } else if (e.key === 'ArrowLeft') {
        basket.leftPressed = true;
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowRight') {
        basket.rightPressed = false;
    } else if (e.key === 'ArrowLeft') {
        basket.leftPressed = false;
    }
});

gameInterval = setInterval(gameLoop, 1000 / 60);
itemInterval = setInterval(spawnItem, 1000);
