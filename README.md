# Análise do Impacto da Compressão na Segmentação de Imagens Lunares

## 📖 Descrição do Problema

Este projeto investiga sistematicamente como algoritmos de compressão de imagens com perdas afetam as etapas posteriores de visão computacional, especificamente a segmentação de crateras e variações de relevo na superfície lunar. 

Para garantir controle absoluto sobre os parâmetros e facilitar a análise minuciosa, todas as operações centrais (DCT, BTC, limiarização, convolução, morfologia matemática e watershed) foram implementadas na mão utilizando manipulação matricial com NumPy, sem delegar a lógica principal a bibliotecas "caixa-preta".

O objetivo não é buscar uma precisão perfeita em relação a um anotador humano, mas avaliar a **estabilidade** dos algoritmos: ou seja, quantificar o quanto as degradações visuais (suavização de bordas, efeito de blocagem, mosaicos) introduzidas pela compressão alteram a saída dos segmentadores quando comparadas às máscaras geradas pelas imagens originais (utilizadas como *pseudo-ground truth*).

---

## 🧠 Técnicas Utilizadas

### Métodos de Compressão
* **DCT (Transformada Discreta de Cosseno):** A imagem é dividida em blocos de 8x8 pixels e levada ao domínio da frequência. Coeficientes de alta frequência são quantizados (Qualidade = 50, inspirada no padrão JPEG). Esse método tende a preservar a aparência global da cena, mas pode suavizar contornos críticos.
* **BTC (Block Truncation Coding):** Trabalha puramente no domínio espacial com blocos de 4x4 pixels. A partir da média e do desvio padrão locais, agrupa a intensidade do bloco em apenas dois níveis. O BTC preserva o contraste interno, mas cria transições artificiais bruscas (efeito mosaico/blocagem).

### Estratégias de Segmentação
* **Limiarização Adaptativa e Fechamento Morfológico:** A imagem em escala de cinza sofre um corte binário baseado na sua própria média de intensidade. Logo após, aplica-se um fechamento (dilatação seguida de erosão com kernel 3x3) para unir falhas de contorno sem expandir massivamente a área segmentada.
* **Sobel e Watershed (Orientado a Gradientes):** É aplicada uma leve suavização Gaussiana (sigma 1.0) para diminuir ruídos sem destruir bordas. Em seguida, as componentes de borda são extraídas via filtros de Sobel. Pixels de baixo gradiente são definidos como marcadores internos e propagados (como no algoritmo de Watershed) até atingirem regiões de forte transição (bordas estabelecidas).

### Métricas de Avaliação
As segmentações das imagens comprimidas são contrastadas com as das imagens intactas por meio de:
* **mIoU (Intersection over Union)** e **Coeficiente Dice:** Medem a qualidade da sobreposição entre as áreas.
* **Pixel Accuracy (PA):** Proporção de acerto direto por pixel.
* **Taxa de Pixels Alterados:** Indica a fragilidade direta da segmentação frente aos diferentes padrões de ruído introduzidos.

---

## 💻 Instruções de Execução e Organização do Projeto

O repositório foi construído de forma modular, permitindo tanto a reprodução integral do experimento (em lote) quanto a execução isolada de cada etapa metodológica.

### 1. Organização da Base de Dados
O projeto utiliza o dataset **Luna-1** (imagens da superfície lunar em 256x256).
* **Download:** Baixe o conjunto de dados no [repositório oficial do Luna-1](https://github.com/droneslab/Luna-1/tree/main).
* **Estruturação:** O conjunto completo contém 5067 imagens PNG. Ao executar os scripts, o código extrai automaticamente uma amostra pseudoaleatória de **357 imagens** (utilizando uma semente fixa igual a 42). Isso garante representatividade estatística (95% de confiança) mantendo a reprodutibilidade exata do experimento.

### 2. Visão Geral dos Algoritmos
A pipeline é dividida em métodos de compressão e estratégias de segmentação, implementados nativamente em NumPy:

* **Compressão BTC (`compress/btc.py`):** Opera no domínio espacial dividindo a imagem em blocos de 4x4. Usando a média e o desvio padrão locais, reduz a intensidade do bloco a apenas dois níveis. Preserva o contraste, mas gera artefatos de "mosaico".
* **Compressão DCT (`frequency_compression.py`):** Opera no domínio da frequência com blocos de 8x8. Descarta detalhes finos quantizando os coeficientes de alta frequência (fator de qualidade 50). Suaviza a imagem globalmente, mas pode borrar bordas sensíveis.
* **Segmentação (Limiarização + Fechamento):** Binariza a imagem com base na sua própria média de intensidade e aplica fechamento morfológico (kernel 3x3) para unir contornos de crateras fragmentadas.
* **Segmentação (Sobel + Watershed):** Extrai mapas de gradiente após uma leve suavização Gaussiana. Pixels de baixo contraste viram marcadores internos (sementes) que se propagam até encontrar fortes transições de borda (crateras/relevo).

### 3. Executando o Experimento

Primeiro, garanta que as dependências estão instaladas:
```bash
pip install -r requirements.txt
```

**Opção A: Execução Completa (Pipeline Automatizada)**
Para rodar todo o experimento de uma só vez (amostragem, pré-processamento, compressão, segmentação e cálculo das métricas), basta executar o orquestrador principal:
```bash
python main.py
```

**Opção B: Execução Modular (Passo a Passo)**
Se desejar avaliar os efeitos de forma isolada, siga a ordem lógica da arquitetura do repositório:

1. **Compressão Espacial:** Executa o algoritmo BTC em toda a amostra do dataset.
   ```bash
   python compress/btc.py
   ```
2. **Compressão Frequencial:** Executa a compressão por DCT.
   ```bash
   python frequency_compression.py
   ```
3. **Segmentação:** Aplica as pipelines morfológicas e orientadas a gradiente sobre as imagens (comprimidas e originais).
   ```bash
   python segmentation.py
   ```
4. **Avaliação Quantitativa:** Calcula as métricas (mIoU, Dice, Pixel Accuracy e Taxa de Pixels Alterados) comparando as máscaras.
   ```bash
   python analysis/metrics.py
   ```

### 4. Resultados e Exemplos Visuais
A pasta `results/` contém as saídas das execuções, onde você pode conferir visualmente:
* Exemplos de imagens antes e após as compressões (DCT e BTC).
* Máscaras de segmentação finais.
* Gráficos e tabelas resultantes da análise quantitativa dos artefatos.