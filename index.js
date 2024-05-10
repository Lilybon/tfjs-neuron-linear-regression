import * as tf from '@tensorflow/tfjs-node'
import { inputs, outputs, testInputs, testOutputs } from './data.js'

tf.util.shuffleCombo(inputs, outputs)

const inputsTensor = tf.tensor2d(inputs)
const outputsTensor = tf.tensor2d(outputs)
const maxsTensor = tf.max(inputsTensor, 0)
const minsTensor = tf.min(inputsTensor, 0)
const normalizedInputsTensor = normalize(inputsTensor, minsTensor, maxsTensor)
inputsTensor.dispose()

console.log('Maxs:')
maxsTensor.print()
console.log('Mins:')
minsTensor.print()
console.log('Normalized Inputs:')
normalizedInputsTensor.print()

const model = tf.sequential()
model.add(tf.layers.dense({ inputShape: [2], units: 1 }))
model.summary()

run()

function normalize(inputs, mins, maxs) {
  return tf.tidy(function () {
    return tf.div(tf.sub(inputs, mins), tf.sub(maxs, mins))
  })
}

async function run() {
  await train()
  evaluate()
}

async function train() {
  const LEARNING_RATE = 0.5

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError',
  })

  const results = await model.fit(normalizedInputsTensor, outputsTensor, {
    callbacks: { onEpochEnd: logProgress },
    validationSplit: 0.15,
    shuffle: true,
    batchSize: 64,
    epochs: 25,
  })

  console.log(
    'Average error loss: ',
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  )
  console.log(
    'Average validation error loss: ',
    Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  )

  outputsTensor.dispose()
  normalizedInputsTensor.dispose()
}

function evaluate() {
  tf.tidy(function () {
    const normalizedInput = normalize(
      tf.tensor2d(testInputs),
      minsTensor,
      maxsTensor
    )
    const predictedOutput = model.predict(normalizedInput)
    const realOutput = tf.tensor2d(testOutputs)

    console.log('Predicted Output: ')
    predictedOutput.print()
    console.log('Real Output: ')
    realOutput.print()
  })

  maxsTensor.dispose()
  minsTensor.dispose()
  model.dispose()
}

function logProgress(epoch, logs) {
  console.log('Data for epoch', epoch, Math.sqrt(logs.loss))
}
