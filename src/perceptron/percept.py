def perceptron(x1, x2, w1, w2, theta, debug=True):
    y = round(w1 * x1 + w2 * x2, 10)
    exp = f'{y} = {w1} * {x1} + {w2} * {x2}'
    args = f'[ x1 = {x1}, x2 = {x2} ]'
    dec = 0 if y <= round(theta, 10) else 1
    debug and print(f"{args} : {exp} {' >' if dec else '<='} {theta}, return {dec}")
    return dec


def logic_gate_tester(w1, w2, theta=1, debug=False):
    print(f'\nw1 = {w1}, w2 = {w2}, theta={theta}')
    perceptron(0, 0, w1, w2, theta)
    perceptron(0, 1, w1, w2, theta)
    perceptron(1, 0, w1, w2, theta)
    perceptron(1, 1, w1, w2, theta)


if __name__ == '__main__':
    perceptron(2, 1, 0.1, 0.3, 0.6)
    perceptron(3, 1, 0.1, 0.3, 0.6)
    perceptron(4, 1, 0.1, 0.3, 0.6)
