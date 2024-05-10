import * as tf from '@tensorflow/tfjs-node'
import { LEARNING_RATE, MODEL_DIR_PATH } from './config.js'
import { TRAIN_INPUTS, TRAIN_OUTPUTS } from './data.js'
import { normalize, logProgress } from './utils.js'

async function train() {
  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [2], units: 1 }))
  model.summary()
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError',
  })

  tf.util.shuffleCombo(TRAIN_INPUTS, TRAIN_OUTPUTS)

  const tranInputs = tf.tensor2d(TRAIN_INPUTS)
  const trainOutputs = tf.tensor2d(TRAIN_OUTPUTS)

  const trainMaxs = tf.max(tranInputs, 0)
  const trainMins = tf.min(tranInputs, 0)
  const normalizedTrainInputs = normalize(tranInputs, trainMaxs, trainMins)
  tranInputs.dispose()
  trainMaxs.dispose()
  trainMins.dispose()

  const results = await model.fit(normalizedTrainInputs, trainOutputs, {
    callbacks: { onEpochEnd: logProgress },
    validationSplit: 0.15,
    shuffle: true,
    batchSize: 64,
    epochs: 25,
  })
  normalizedTrainInputs.dispose()
  trainOutputs.dispose()

  console.log(
    'Average error loss: ',
    Math.sqrt(results.history.loss[results.history.loss.length - 1])
  )
  console.log(
    'Average validation error loss: ',
    Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  )

  await model.save(MODEL_DIR_PATH)
  model.dispose()
}

train()
