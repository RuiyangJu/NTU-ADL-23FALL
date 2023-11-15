import matplotlib.pyplot as plt
import json

with open('model/trainer_state.json', 'r') as f:
    data = json.load(f)

log_history = data['log_history']
x = [step['step'] for step in log_history if 'learning_rate' in step.keys()]
plt.figure()
plt.title("Learning Rate Curve")
y = [step['learning_rate'] for step in log_history if 'learning_rate' in step.keys()]
plt.plot(x, y)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.savefig('learning_rate_curve.png')