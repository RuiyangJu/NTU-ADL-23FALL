import matplotlib.pyplot as plt
import json

with open('model/trainer_state.json', 'r') as f:
    data = json.load(f)

log_history = data['log_history']
x = [step['step'] for step in log_history if 'loss' in step.keys()]
plt.figure()
plt.title("Loss Curve")
y = [step['loss'] for step in log_history if 'loss' in step.keys()]
plt.plot(x, y)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')