# Multi-Layer Perceptron (MLP) do Zero em NumPy

Este repositório contém a implementação completa de uma Rede Neural Artificial do tipo Multi-Layer Perceptron (MLP) desenvolvida **100% do zero**, utilizando estritamente a biblioteca **NumPy** para operações de álgebra linear. O objetivo do projeto é demonstrar o domínio dos fundamentos matemáticos por trás do *Deep Learning*, incluindo a formulação manual do algoritmo de *Backpropagation*, o cálculo de derivadas de funções de ativação e a otimização de parâmetros.

O modelo foi validado e avaliado utilizando o dataset **MNIST** (caracteres manuscritos de dígitos de 0 a 9), alcançando marcas excepcionais de acurácia.


## Requisitos Cumpridos

- [x] **Ao menos 2 camadas ocultas**: Arquitetura modular flexível permitindo empilhamento dinâmico de múltiplas camadas intermediárias.
- [x] **Função de ativação configurável**: Implementação isolada e parametrizada da função **ReLU** nas camadas ocultas.
- [x] **Camada de saída com Softmax + Cross-Entropy Loss**: Formulação matemática acoplada para estabilidade e otimização de gradientes na saída.
- [x] **Backpropagation manual**: Algoritmo de regra da cadeia codificado puramente via operações matriciais, sem o uso de autograd.
- [x] **Treinamento por Mini-Batches com SGD**: Suporte a lotes estocásticos configuráveis para convergência otimizada.
- [x] **Verificação de Gradientes (*Gradient Check*)**: Validação numérica via diferenças finitas convergindo em escala de erro infinitesimal ($10^{-11}$).
- [x] **Otimizador adicional**: Implementação do algoritmo **SGD com Momentum** para aceleração de gradiente e inércia.
- [x] **Análise de Ativações Internas**: Mapeamento visual e interpretação de esparsidade no disparo dos neurônios intermediários.
- [x] **Análise de Espaço Latente (Embeddings)**: Redução de dimensionalidade via **PCA desenvolvido do zero** em NumPy.
- [x] **Testes Unitários**: Suíte de testes automatizados com cobertura para os blocos fundamentais de ativação.

---

## Estrutura do Projeto

```text
├── mlp/
│   ├── __init__.py
│   ├── network.py       # Classes MLP e LinearLayer (com Momentum integrado)
│   ├── activations.py   # Implementações manuais de ReLU e Softmax (Forward/Backward)
│   ├── losses.py        # Cálculo da Cross-Entropy Loss
│   └── optimizers.py    # Funções auxiliares de treinamento e histórico
├── results/             # Gráficos e resultados salvos automaticamente
│   ├── curva_de_loss.png
│   ├── comparacao_learning_rates.png
│   ├── matriz_confusao.png
│   ├── ativacoes_internas.png
│   └── embeddings_pca_do_zero.png
├── experimentos.ipynb  # Notebook principal de treino e validação
├── test_mlp.py          # Suíte de testes unitários automatizados
├── requirements.txt     # Gerenciamento de dependências do projeto
├── .gitignore           # Filtro para ignorar arquivos de cache e locais
└── README.md            # Documentação do projeto
```

## Resultados e Análises Técnicas

O modelo foi avaliado em diferentes configurações de hiperparâmetros, demonstrando excelente capacidade de generalização e revelando a física dos algoritmos de otimização através dos gráficos salvos na pasta `./results/`.

### 1. Evolução da Perda (Loss) e Performance com Momentum

Ao incorporar o otimizador **SGD com Momentum** na arquitetura principal ($784 \rightarrow 128 \rightarrow 64 \rightarrow 10$), o modelo demonstrou uma aceleração notável no início do treinamento, mitigando oscilações espúrias entre mini-batches. A evolução do erro e o resultado final consolidaram-se em:

* **Época 1**: Loss 0.2609  *(Início significativamente mais baixo devido ao efeito cumulativo de velocidade)*
* **Época 2**: Loss 0.1293
* **Época 3**: Loss 0.0981
* **Época 4**: Loss 0.0812
* **Época 5**: Loss 0.0711
* **Acurácia Final no Conjunto de Testes**: **96.72%**

### 2. Estudo do Impacto da Taxa de Aprendizado (Learning Rate)

O experimento comparativo isolando três taxas de aprendizado distintas sob o efeito do Momentum elucidou perfeitamente o comportamento assintótico dos gradientes:

* **LR = 0.1 (Otimização Eficiente)**: Apresentou a descida mais íngreme, iniciando em `0.2585` e convergindo rapidamente para `0.0740` na 5ª época. A combinação de passos largos com a força de inércia do Momentum permitiu que o modelo contornasse platôs de forma ágil.

* **LR = 0.01 (Estabilidade Gradual)**: Demonstrou um decaimento altamente previsível e linear (`0.3704` $\rightarrow$ `0.0706`), mostrando-se uma taxa robusta para garantir estabilidade assintótica.

* **LR = 0.001 (Inércia Insuficiente)**: Iniciou com erro elevado (`0.9149`) e, apesar de reduzir a perda para `0.2477`, provou que passos excessivamente diminutos tornam a convergência inviável dentro do limite de poucas épocas, mantendo o modelo sub-otimizado.

*(Esses comportamentos estão mapeados visualmente e comparados no gráfico gerado em `./results/comparacao_learning_rates.png`)*

### 3. Espaço de Embeddings Latentes (PCA do Zero)

A extração das ativações da última camada oculta, projetadas em duas dimensões por um algoritmo de **PCA conhecido e implementado nativamente em NumPy** (`np.linalg.eigh`), gerou o seguinte mapa:

* **Separação Semântica Clara**: O surgimento de clusters geometricamente isolados (como a segregação nítida do dígito `0` na extrema esquerda) comprova que a rede agrupa imagens de uma mesma classe em coordenadas próximas por similaridade abstrata, indo muito além da mera decoreba de pixels.

* **Zonas de Ambiguidade**: Regiões centrais de sutil sobreposição de cores evidenciam as fronteiras difusas do modelo, onde caligrafias ambíguas (como confusões morfológicas mapeadas entre `4`, `7` e `9`) compartilham assinaturas de ativação semelhantes nos neurônios latentes.

### 4. Interpretabilidade de Ativações Internas (ReLU)

A análise dos disparos da primeira camada oculta ($8 \times 16$ neurônios) ao processar o dígito **5** destaca propriedades fundamentais:

* **Esparsidade Ativa**: A presença de grandes zonas escuras (inativas) valida matematicamente a propriedade de esparsidade induzida pela função $\max(0, x)$, otimizando o custo computacional e mitigando a co-adaptação nociva de parâmetros.

* **Detectores de Características (*Feature Detectors*)**: Neurônios com alta luminescência (tons claros no mapa `magma`) funcionam como extratores de bordas locais, reagindo de forma extrema a subestruturas geométricas específicas do dígito, como a curva inferior ou a quina reta superior.

## Engenharia de Software e Validação Matemática

### Validação de Gradientes (Gradient Checking)

Para blindar o código contra "bugs silenciosos" nas derivadas parciais, foi desenvolvido um algoritmo de verificação numérica por diferenças finitas. O teste foi desenhado isolando o fluxo conjunto da `LinearLayer` com o gradiente da saída.

Ao aplicar uma perturbação física bidirecional de limite com $\epsilon = 10^{-5}$, a diferença relativa calculada foi de:


$$\text{Diferença Relativa} = 1.23 \times 10^{-11}$$

Esse valor, por situar-se na escala infinitesimal de onze casas decimais após a vírgula, crava o **sucesso e a precisão absoluta do Backpropagation manual**.

### Testes Unitários Automatizados

O projeto conta com testes unitários formais para validar o comportamento dos blocos constitutivos isolados. Eles utilizam a diretiva de *docstrings* descritivas no terminal para máxima legibilidade.

Para executar os testes, utilize o comando:

```bash
python test_mlp.py

```

**Saída esperada no terminal:**

```text
[ReLU] Verifica se valores negativos são zerados e positivos são mantidos (Forward) ... ok
[ReLU] Verifica se o gradiente é zerado para ativações menores ou iguais a zero (Backward) ... ok
[Softmax] Verifica se a saída gera probabilidades válidas que somam 1.0 (Forward) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.149s

OK

```

## Desafios e Dificuldades Enfrentadas

Desenvolver uma biblioteca de Deep Learning completamente do zero, sem o auxílio de abstrações prontas, trouxe desafios complexos de depuração lógica e matemática. Destaco as principais dificuldades que enfrentei e como as superei:

1. **Efeito de Mutação Involuntária no Gradient Checking:**
Inicialmente, meu script de *Gradient Check* numérico retornava um erro de 100% de divergência relativa (`1.00e+00`). Foi um processo complexo de depuração até compreender que, como meu método `LinearLayer.backward` realiza a atualização imediata e mutável dos pesos (`self.weights -= ...`), o laço numérico subsequente estava perturbando matrizes de pesos que já tinham sido alteradas pelo passo analítico. Superei esse problema isolando o teste em uma instância imaculada, realizando cópias profundas estáticas (`.copy()`) dos pesos antes de qualquer mutação e isolando o gradiente analítico puro do otimizador.

2. **Ajuste de Escala Matemática:**
Ainda na verificação de gradientes, enfrentei problemas com o descasamento de escala de cerca de 7% (`6.95e-02`). Descobri que isso ocorria porque a formulação analítica do meu loop principal de treino calcula o gradiente médio dividindo pelo tamanho do mini-batch, enquanto a aproximação por diferenças finitas avalia a perturbação sobre a função de perda cumulativa (total). Tive que readequar a matemática do script de teste multiplicando a perda pelo comprimento das amostras fictícias para reestabelecer o alinhamento perfeito de escala.

3. **Persistência de Cache e Estado no Jupyter Notebook:**
Durante a implementação do otimizador Momentum, modifiquei o arquivo modular `mlp/network.py` para incluir as novas variáveis de estado e parâmetros no construtor `__init__`. Ao rodar os experimentos no notebook, me deparei com erros de assinatura de função (`TypeError: got an unexpected keyword argument 'momentum'`). Demorei a perceber que o Kernel do Jupyter mantém os módulos antigos alocados em cache na memória. Resolvi essa inconsistência técnica configurando as diretivas de extensão `%load_ext autoreload` e `%autoreload 2` no topo do arquivo, forçando a atualização reativa dos scripts `.py`.

## Como Executar o Projeto

1. Certifique-se de ter o **Python 3.x** instalado. Instale as dependências oficiais do repositório a partir do arquivo de requerimentos:

```bash
pip install -r requirements.txt

```

2. Para explorar o treinamento, os experimentos visuais e a geração dos gráficos de resultados, abra e execute as células do arquivo localizado na raiz:

```text
Ponderada_MLP.ipynb

```

3. Os gráficos de loss, matriz de confusão comentada, ativações internas e projeção PCA serão salvos automaticamente na pasta `./results/`.
