module.exports = {
  apps: [{
    name: "flock-training-node",
    script: "./run_training_node.sh",
    env: {
      HF_TOKEN: process.env.HF_TOKEN,
      TASK_ID: process.env.TASK_ID,
      FLOCK_API_KEY: process.env.FLOCK_API_KEY,
      HF_USERNAME: process.env.HF_USERNAME,
      CUDA_VISIBLE_DEVICES: "0"
    },
    error_file: "logs/err.log",
    out_file: "logs/out.log",
    time: true
  }]
} 