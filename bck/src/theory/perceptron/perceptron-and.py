from perceptron.percept import logic_gate_tester

if __name__ == '__main__':
    for w in range(1, 10):
        logic_gate_tester(w / 10, w / 10)
