# AECO — Etapa 3: Rede Neural Artificial com Iris

Implementação da atividade da Aula 10 de Inteligência Artificial: classificação das três espécies do conjunto **Iris** com uma rede neural construída do zero.

As operações de rede neural, funções de ativação, cálculo da perda, retropropagação e atualização de pesos utilizam **NumPy**. O **Scikit-Learn** é usado somente para carregar o dataset e separar treino e teste; nenhum classificador ou rotina de treinamento pronta é utilizado.

## 1. Requisitos

- Python 3.10 ou superior;
- Bibliotecas `numpy` e `scikit-learn` (arquivo `requirements.txt`).

O Iris é carregado automaticamente com `load_iris()`; não é necessário baixar dados.

## 2. Preparar o ambiente

Abra um terminal na pasta do projeto.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a política do PowerShell bloquear a ativação, use diretamente `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` e `.\.venv\Scripts\python.exe main.py`.

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Execução

```bash
python main.py
```

Por padrão: **1.000 épocas**, taxa de aprendizado **0,1**, semente **42** e registro da perda a cada **200 épocas**.

Para personalizar:

```bash
python main.py --epocas 1500 --taxa 0.05 --semente 42 --intervalo 300
```

Ajuda dos parâmetros:

```bash
python main.py --help
```

O programa imprime o andamento da perda (cross-entropy), os acertos, a acurácia final e a matriz de confusão.

Para executar os testes automatizados:

```bash
python -m unittest discover -s tests -v
```

## 4. Requisitos do enunciado implementados

| Item | Implementação |
| --- | --- |
| Conjunto | Iris: 150 amostras, 4 características, 3 classes |
| Pré-processamento | Divisão estratificada 80% treino (120) e 20% teste (30) |
| Padronização | Média zero e desvio padrão um, calculados no treino e aplicados ao teste |
| One-hot | Vetores com três posições, com NumPy |
| Arquitetura | 4 entradas → 8 neurônios ReLU → 3 saídas Softmax |
| Inicialização | Pesos aleatórios com semente fixa e vieses inicialmente zerados |
| Aprendizado | Forward, cross-entropy, backward pela regra da cadeia e gradiente descendente em lote |
| Avaliação | Forward do teste, acurácia e matriz de confusão |

## 5. Relatório dos resultados

**Ambiente verificado:** NumPy 2.3.5 e Scikit-Learn 1.8.0. Comando `python main.py` e parâmetros padrão.

**Saída obtida em execução real:**

```text
AECO - Aula 10 | Rede Neural Artificial - Iris
Arquitetura: 4 entradas -> 8 ReLU -> 3 Softmax
Divisão: 120 treino / 30 teste
Épocas: 1000 | Taxa: 0.1 | Semente: 42

Treinamento:
Época    1/1000 | perda: 1.233701
Época  200/1000 | perda: 0.162582
Época  400/1000 | perda: 0.086891
Época  600/1000 | perda: 0.064662
Época  800/1000 | perda: 0.054730
Época 1000/1000 | perda: 0.049468

Avaliação no conjunto de teste:
Acertos: 29/30
Acurácia: 96.67%
Matriz de confusão (linhas: classe real; colunas: classe prevista):
Classes: setosa, versicolor, virginica
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

**Análise:** esperava-se diminuição da perda e classificação correta da maioria dos exemplos. Observou-se redução de **1,233701** para **0,049468** em 1.000 épocas e **29/30 acertos (96,67%)** no conjunto de teste. A única confusão ocorreu quando uma flor *versicolor* foi classificada como *virginica*. Os testes automatizados executaram **5 verificações, todas aprovadas**.

Esses números são **resultados obtidos**, não uma promessa de que toda execução com outras versões de bibliotecas, sementes ou parâmetros produzirá a mesma acurácia.

## 6. Estrutura

```text
.
├── main.py
├── requirements.txt
├── tests/
│   └── test_rede_iris.py
├── .gitignore
└── README.md
```

## 7. Histórico de desenvolvimento

1. **Commit 1:** carregamento, divisão 80/20, padronização e one-hot; `python main.py` exibe as dimensões e estatísticas do pré-processamento.
2. **Commit 2:** arquitetura 4–8–3, ReLU/derivada, Softmax e forward pass; `python main.py` demonstra a saída **antes do treinamento**.
3. **Commit 3:** cross-entropy, backpropagation, gradiente descendente, inferência, acurácia, testes e este README.

A execução completa descrita acima pertence ao **terceiro commit**. Os dois anteriores são etapas intermediárias.
