"""AECO - Aula 10: rede neural artificial para o conjunto Iris.

Implementação do forward e backpropagation apenas com NumPy.
Scikit-learn é usado exclusivamente para carregar e dividir o conjunto de dados.
"""

import argparse

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


def relu(z):
    return np.maximum(z, 0.0)


def derivada_relu(z):
    return (z > 0.0).astype(np.float64)


def softmax(z):
    deslocado = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(deslocado)
    return exp / np.sum(exp, axis=1, keepdims=True)


def entropia_cruzada(probabilidades, y_one_hot):
    """Cross-entropy média das amostras."""
    probabilidades = np.clip(probabilidades, 1e-12, 1.0)
    return -np.mean(np.sum(y_one_hot * np.log(probabilidades), axis=1))


class RedeNeuralIris:
    """MLP 4-8-3 com gradiente descendente em lote (batch)."""

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


    def backward(self, x, y_one_hot, taxa_aprendizado):
        """Calcula gradientes pela regra da cadeia e atualiza parâmetros."""
        n = x.shape[0]
        dz2 = (self.a2 - y_one_hot) / n
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ self.w2.T) * derivada_relu(self.z1)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w1 -= taxa_aprendizado * dw1
        self.b1 -= taxa_aprendizado * db1
        self.w2 -= taxa_aprendizado * dw2
        self.b2 -= taxa_aprendizado * db2

    def treinar(self, x, y_one_hot, epocas=1000, taxa_aprendizado=0.1, intervalo=200):
        """Treina por batch gradient descent e registra a evolução da perda."""
        historico = []
        for epoca in range(1, epocas + 1):
            self.forward(x)
            self.backward(x, y_one_hot, taxa_aprendizado)
            if epoca == 1 or epoca % intervalo == 0 or epoca == epocas:
                perda = entropia_cruzada(self.forward(x), y_one_hot)
                historico.append((epoca, perda))
                print(f"Época {epoca:4d}/{epocas} | perda: {perda:.6f}")
        return historico

    def prever(self, x):
        return np.argmax(self.forward(x), axis=1)


def main():
    parser = argparse.ArgumentParser(description="AECO Aula 10 - Classificador Iris com rede neural do zero")
    parser.add_argument("--epocas", type=int, default=1000, help="Número de épocas (padrão: 1000)")
    parser.add_argument("--taxa", type=float, default=0.1, help="Taxa de aprendizado (padrão: 0.1)")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória (padrão: 42)")
    parser.add_argument("--intervalo", type=int, default=200, help="Intervalo para mostrar a perda")
    args = parser.parse_args()
    if args.epocas <= 0 or args.taxa <= 0 or args.intervalo <= 0:
        parser.error("--epocas, --taxa e --intervalo devem ser positivos")

    dados = preparar_dados(args.semente)
    rede = RedeNeuralIris(args.semente)

    print("AECO - Aula 10 | Rede Neural Artificial - Iris")
    print("Arquitetura: 4 entradas -> 8 ReLU -> 3 Softmax")
    print(f"Divisão: {len(dados['x_treino'])} treino / {len(dados['x_teste'])} teste")
    print(f"Épocas: {args.epocas} | Taxa: {args.taxa} | Semente: {args.semente}")
    print("\nTreinamento:")
    rede.treinar(dados["x_treino"], dados["y_treino_one_hot"], args.epocas, args.taxa, args.intervalo)

    previsoes = rede.prever(dados["x_teste"])
    matriz = np.zeros((3, 3), dtype=int)
    np.add.at(matriz, (dados["y_teste"], previsoes), 1)
    corretas = int(np.sum(previsoes == dados["y_teste"]))
    acuracia = 100.0 * corretas / len(previsoes)
    print("\nAvaliação no conjunto de teste:")
    print(f"Acertos: {corretas}/{len(previsoes)}")
    print(f"Acurácia: {acuracia:.2f}%")
    print("Matriz de confusão (linhas: classe real; colunas: classe prevista):")
    print("Classes:", ", ".join(dados["nomes_classes"]))
    print(matriz)


if __name__ == "__main__":
    main()
