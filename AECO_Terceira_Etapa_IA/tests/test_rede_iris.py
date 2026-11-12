"""Testes de preparação, arquitetura, treino e predição (unittest padrão)."""

import unittest

import numpy as np

from main import RedeNeuralIris, entropia_cruzada, preparar_dados, relu, derivada_relu, softmax


class TesteRedeIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dados = preparar_dados(42)

    def test_divisao_normalizacao_e_one_hot(self):
        d = self.dados
        self.assertEqual(d["x_treino"].shape, (120, 4))
        self.assertEqual(d["x_teste"].shape, (30, 4))
        self.assertEqual(d["y_treino_one_hot"].shape, (120, 3))
        np.testing.assert_allclose(d["x_treino"].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(d["x_treino"].std(axis=0), 1, atol=1e-12)
        np.testing.assert_array_equal(d["y_treino_one_hot"].sum(axis=1), np.ones(120))

    def test_relu_e_derivada(self):
        x = np.array([[-1.0, 0.0, 2.0]])
        np.testing.assert_array_equal(relu(x), [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(derivada_relu(x), [[0, 0, 1]])

    def test_softmax_soma_um(self):
        probs = softmax(np.array([[1000.0, 1001.0, 999.0], [1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(probs).all())

    def test_arquitetura(self):
        rede = RedeNeuralIris()
        self.assertEqual(rede.w1.shape, (4, 8))
        self.assertEqual(rede.b1.shape, (1, 8))
        self.assertEqual(rede.w2.shape, (8, 3))
        self.assertEqual(rede.b2.shape, (1, 3))
        probs = rede.forward(self.dados["x_teste"])
        self.assertEqual(probs.shape, (30, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_treinamento_reduz_perda_e_aprende(self):
        d = self.dados
        rede = RedeNeuralIris()
        antes = entropia_cruzada(rede.forward(d["x_treino"]), d["y_treino_one_hot"])
        rede.treinar(d["x_treino"], d["y_treino_one_hot"], epocas=1000, taxa_aprendizado=0.1, intervalo=1000)
        depois = entropia_cruzada(rede.forward(d["x_treino"]), d["y_treino_one_hot"])
        self.assertLess(depois, antes)
        self.assertGreaterEqual(np.mean(rede.prever(d["x_teste"]) == d["y_teste"]), 0.85)


if __name__ == "__main__":
    unittest.main()
