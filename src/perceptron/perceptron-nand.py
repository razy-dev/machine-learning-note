from perceptron.percept import logic_gate_tester

if __name__ == '__main__':
    for w in range(-10, -1):
        logic_gate_tester(w / 10, w / 10, -1)
