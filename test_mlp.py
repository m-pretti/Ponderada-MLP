import unittest
import numpy as np
from mlp.activations import ReLU, Softmax

class TestMLPComponents(unittest.TestCase):
    
    def test_relu_forward(self):
        """[ReLU] Verifica se valores negativos são zerados e positivos são mantidos (Forward)"""
        relu = ReLU()
        dados_teste = np.array([[-2.0, 0.0, 5.0]])
        esperado = np.array([[0.0, 0.0, 5.0]])
        np.testing.assert_array_equal(relu.forward(dados_teste), esperado)
        
    def test_relu_backward(self):
        """[ReLU] Verifica se o gradiente é zerado para ativações menores ou iguais a zero (Backward)"""
        relu = ReLU()
        dados_teste = np.array([[-1.0, 2.0]])
        _ = relu.forward(dados_teste) # Popula o estado interno
        
        grad_saida = np.array([[10.0, 10.0]])
        grad_esperado = np.array([[0.0, 10.0]]) # Zera onde a entrada original era <= 0
        np.testing.assert_array_equal(relu.backward(grad_saida), grad_esperado)

    def test_softmax_probabilidades(self):
        """[Softmax] Verifica se a saída gera probabilidades válidas que somam 1.0 (Forward)"""
        softmax = Softmax()
        logits = np.array([[1.0, 1.0, 1.0]]) # Entradas iguais devem gerar probabilidades iguais
        probas = softmax.forward(logits)
        
        # A soma das probabilidades deve ser exatamente 1.0
        self.assertAlmostEqual(np.sum(probas), 1.0, places=5)
        # Cada classe deve receber exatamente 1/3 de probabilidade
        np.testing.assert_array_almost_equal(probas, np.array([[1/3, 1/3, 1/3]]))

if __name__ == '__main__':
    # O argumento argv=['first-arg', '-v'] força o modo verbose mesmo rodando o script puro
    import sys
    unittest.main(argv=[sys.argv[0], '-v'])