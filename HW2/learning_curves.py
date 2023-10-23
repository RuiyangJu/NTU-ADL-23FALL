import matplotlib.pyplot as plt
import json

with open('model/trainer_state.json', 'r') as f:
    data = json.load(f)

log_history = data['log_history']
x = [step['step'] for step in log_history if 'eval_loss' in step.keys()]
plt.figure()
plt.title("Learning Curve")
# rouge-1
y = [step['eval_rouge-1_f'] for step in log_history if 'eval_rouge-1_f' in step.keys()]
plt.plot(x, y, label="rouge-1_f")
y = [step['eval_rouge-1_p'] for step in log_history if 'eval_rouge-1_p' in step.keys()]
plt.plot(x, y, label="rouge-1_p")
y = [step['eval_rouge-1_r'] for step in log_history if 'eval_rouge-1_r' in step.keys()]
plt.plot(x, y, label="rouge-1_r")
# rouge-2
y = [step['eval_rouge-2_f'] for step in log_history if 'eval_rouge-2_f' in step.keys()]
plt.plot(x, y, label="rouge-2_f")
y = [step['eval_rouge-2_p'] for step in log_history if 'eval_rouge-2_p' in step.keys()]
plt.plot(x, y, label="rouge-2_p")
y = [step['eval_rouge-2_r'] for step in log_history if 'eval_rouge-2_r' in step.keys()]
plt.plot(x, y, label="rouge-2_r")
# rouge-l
y = [step['eval_rouge-l_f'] for step in log_history if 'eval_rouge-l_f' in step.keys()]
plt.plot(x, y, label="rouge-l_f")
y = [step['eval_rouge-l_p'] for step in log_history if 'eval_rouge-l_p' in step.keys()]
plt.plot(x, y, label="rouge-l_p")
y = [step['eval_rouge-l_r'] for step in log_history if 'eval_rouge-l_r' in step.keys()]
plt.plot(x, y, label="rouge-l_r")
plt.xlabel('Traing Steps')
plt.ylabel('ROUGE')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig('Learning Curve.png')