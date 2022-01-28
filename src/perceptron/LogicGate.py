from perceptron.percept import perceptron


class LogicGate:
    # 논리 테스트 입력값
    input = [(0, 0,), (0, 1,), (1, 0,), (1, 1,)]

    # AND, NAND, OR Test 를 통해 구한 weights 와 편향(bias = -theta)
    map = {
        'AND': (.6, .6, -1,),
        'NAND': (-.6, -.6, 1,),
        'OR': (.6, .6, -.5,),
    }

    def and_(self, x1, x2):
        return self.exc(x1, x2, 'AND')

    def nand_(self, x1, x2):
        return self.exc(x1, x2, 'NAND')

    def or_(self, x1, x2):
        return self.exc(x1, x2, 'OR')

    def xor_(self, x1, x2):
        return self.and_(self.nand_(x1, x2), self.or_(x1, x2))

    def exc(self, x1, x2, mode: str):
        """'가설(추정) 함수' 에서 계산된 값을 '활성화 함수'에서 판단하여 최종 판단 결과를 응답 한다."""
        return self.activation_function(self.hypothesis(x1, x2, *self.map.get(mode.upper())))

    @staticmethod
    def hypothesis(x1, x2, w1, w2, bias):
        """가설(추정) 함수"""
        return round(w1 * x1 + w2 * x2 + bias, 10)

    @staticmethod
    def activation_function(y):
        """
        Activation Function : 가정 결과를 판정하는 함수.
        계단 함수(Step Function), 선형 함수(Linear Function), 시그모이드 함수(Sigmoid Function),
        소프트맥스 함수(Softmax Function), 하이퍼볼릭 탄젠트(Hyperbolic Tangent, tanh), 렐루 함수(Rectified Linear Unit, ReLU)
        등이 있다.
        """
        return 1 if y > 0 else 0

    def test(self, mode: str):
        print(f'\n{mode.upper()} Test')
        logic = getattr(self, f'{mode.lower()}_')
        for x1, x2 in self.input:
            print(f'[ {x1}, {x2} ] = {logic(x1, x2)}')


if __name__ == '__main__':
    gate = LogicGate()
    gate.test('and')
    gate.test('nand')
    gate.test('or')
    gate.test('xor')
