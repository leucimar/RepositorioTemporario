"""AECO - Aula 10: preparação dos dados Iris para a rede neural.

Etapa 2 de 3. Execute: python main.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def preparar_dados(semente=42):
    """Separa os dados e padroniza usando apenas estatísticas do treinamento."""
    iris = load_iris()
    entradas = np.asarray(iris.data, dtype=np.float64)
    classes = np.asarray(iris.target, dtype=np.int64)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        entradas, classes, test_size=0.20, random_state=semente, stratify=classes
    )

    media = np.mean(x_treino, axis=0)
    desvio = np.std(x_treino, axis=0)
    desvio = np.where(desvio == 0.0, 1.0, desvio)

    x_treino = (x_treino - media) / desvio
    x_teste = (x_teste - media) / desvio
    y_treino_one_hot = np.eye(3, dtype=np.float64)[y_treino]
    y_teste_one_hot = np.eye(3, dtype=np.float64)[y_teste]

    return {
        "x_treino": x_treino,
        "x_teste": x_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
        "y_treino_one_hot": y_treino_one_hot,
        "y_teste_one_hot": y_teste_one_hot,
        "nomes_classes": tuple(iris.target_names),
    }


# Etapa 2: rede neural 4 -> 8 (ReLU) -> 3 (Softmax).
def relu(z):
    return np.maximum(z, 0.0)


def derivada_relu(z):
    return (z > 0.0).astype(np.float64)


def softmax(z):
    deslocado = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(deslocado)
    return exp / np.sum(exp, axis=1, keepdims=True)


class RedeNeuralIris:
    """Rede MLP com parâmetros aleatórios e propagação direta feita com NumPy."""

    def __init__(self, semente=42):
        gerador = np.random.default_rng(semente)
        self.w1 = gerador.standard_normal((4, 8)) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros((1, 8), dtype=np.float64)
        self.w2 = gerador.standard_normal((8, 3)) * np.sqrt(2.0 / 8)
        self.b2 = np.zeros((1, 3), dtype=np.float64)

    def forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2


def main():
    dados = preparar_dados()
    rede = RedeNeuralIris()
    probabilidades = rede.forward(dados["x_teste"])
    print("AECO - Etapa 2/3: arquitetura e forward pass")
    print("Arquitetura: 4 entradas -> 8 neurônios ReLU -> 3 saídas Softmax")
    print(f"Divisão: {len(dados['x_treino'])} treino / {len(dados['x_teste'])} teste")
    print(f"Pesos: W1={rede.w1.shape}, W2={rede.w2.shape}")
    print(f"Probabilidades da primeira amostra: {np.round(probabilidades[0], 4)}")
    print(f"Soma das probabilidades: {probabilidades[0].sum():.6f}")
    print("Status: rede inicializada; treinamento será implementado na etapa 3.")


if __name__ == "__main__":
    main()
