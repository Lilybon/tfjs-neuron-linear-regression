import * as tf from '@tensorflow/tfjs-node'

export function normalize(inputs, mins, maxs) {
  return tf.tidy(function () {
    return tf.div(tf.sub(inputs, mins), tf.sub(maxs, mins))
  })
}

export function logProgress(epoch, logs) {
  console.log('Data for epoch', epoch, Math.sqrt(logs.loss))
}
