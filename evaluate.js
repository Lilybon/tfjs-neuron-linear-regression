import * as tf from '@tensorflow/tfjs-node'
import { MODEL_FILE_PATH } from './config.js'
import { TRAIN_INPUTS, TEST_INPUTS, TEST_OUTPUTS } from './data.js'
import { normalize } from './utils.js'

async function evaluate() {
  const model = await tf.loadLayersModel(MODEL_FILE_PATH)
  tf.tidy(function () {
    const trainInputs = tf.tensor2d(TRAIN_INPUTS)
    const trainMaxs = tf.max(trainInputs, 0)
    const trainMins = tf.min(trainInputs, 0)
    const normalizedTestInputs = normalize(tf.tensor2d(TEST_INPUTS), trainMaxs, trainMins)
    const testOutputs = tf.tensor2d(TEST_OUTPUTS)
    const predictedTestOutputs = model.predict(normalizedTestInputs)

    console.log('Real Output: ')
    testOutputs.print()
    console.log('Predicted Output: ')
    predictedTestOutputs.print()
  })
  model.dispose()
}

evaluate()
