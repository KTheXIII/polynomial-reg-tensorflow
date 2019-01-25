let x_vals = []
let y_vals = []

let a, b, c, d

const learningRate = 0.5
const optimizer = tf.train.sgd(learningRate)

function setup() {
    createCanvas(windowWidth, windowHeight)

    a = tf.variable(tf.scalar(random(-1, 1)))
    b = tf.variable(tf.scalar(random(-1, 1)))
    c = tf.variable(tf.scalar(random(-1, 1)))
    d = tf.variable(tf.scalar(random(-1, 1)))
}

function draw() {
    background(0)

    stroke(100)
    strokeWeight(1)
    let linePoint = [[width / 2, 0], [width / 2, height]]
    line(linePoint[0][0], linePoint[0][1], linePoint[1][0], linePoint[1][1])
    linePoint = [[0, height / 2], [width, height / 2]]
    line(linePoint[0][0], linePoint[0][1], linePoint[1][0], linePoint[1][1])

    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals)
            optimizer.minimize(() => loss(predict(x_vals), ys))
        }
    })

    stroke(255)
    strokeWeight(8)
    for (let i = 0; i < x_vals.length; i++) {
        let pX = map(x_vals[i], -1, 1, 0, width)
        let pY = map(y_vals[i], -1, 1, height, 0)
        point(pX, pY)
    }

    beginShape()
    strokeWeight(2)
    stroke(0, 255, 255)
    noFill()

    const curveX = []
    for (let x = -1; x <= 1; x += 0.01) {
        curveX.push(x)
    }

    const ys = tf.tidy(() => predict(curveX))
    let curveY = ys.dataSync()
    ys.dispose()

    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width)
        let y = map(curveY[i], -1, 1, height, 0)
        vertex(x, y)
    }
    endShape()

    console.log(tf.memory().numTensors)
}

function predict(x) {
    const xs = tf.tensor1d(x)
    // y = ax^2 + bx + c
    // const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
    // f(x) = ax^3 + bx^2 + cx + d
    const ys = xs
        .pow(tf.scalar(3))
        .mul(a)
        .add(xs.square().mul(b))
        .add(xs.mul(c))
        .add(c)
    return ys
}

function loss(pred, label) {
    // (pred, label) => pred.sub(label).square().mean(); <-- fancy ES6 function
    return pred
        .sub(label)
        .square()
        .mean()
}

function mousePressed() {
    let x = map(mouseX, 0, width, -1, 1)
    let y = map(mouseY, 0, height, 1, -1)
    x_vals.push(x)
    y_vals.push(y)
}
