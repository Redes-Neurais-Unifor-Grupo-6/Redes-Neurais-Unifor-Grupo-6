# TrashNet: MLP, CNN, pesos de classe e resolução

Estudo de classificação de resíduos para a atividade final de Redes Neurais Artificiais. O protocolo compara seis modelos treinados do zero, preserva as seis classes do TrashNet e separa a seleção por validação da avaliação final no teste. Os resultados numéricos são mantidos nos arquivos gerados; este README não substitui o artigo nem os registros experimentais.

**Identificação provisória:** quatro integrantes `Lorem Ipsum`, matrícula `9999999`, conforme solicitado. **O repositório GitHub ainda precisa ser criado.** Antes da entrega, preencher os integrantes reais e `github_url` em [`docs/metadata.json`](docs/metadata.json), regenerar o artigo e conferir sua primeira página. A equipe pode ter de quatro a seis integrantes.

> **Condição expressa do enunciado:** o uso de IA para código é permitido, mas a delegação de 100% do projeto à IA implica nota zero. A equipe precisa revisar e compreender o código, conduzir as decisões e assumir a análise e a interpretação. O [guia da equipe](docs/GUIA_DA_EQUIPE.md) organiza essa revisão; sua existência não atesta que ela já foi realizada.

## Onde encontrar cada entrega

| Material | Caminho |
|---|---|
| Artigo técnico (10 páginas; identificação provisória) | [`docs/artigo.pdf`](docs/artigo.pdf) |
| Texto editável gerado do artigo | `docs/artigo.md` |
| Identificação e link usados pelo artigo | [`docs/metadata.json`](docs/metadata.json) |
| Análise exploratória | [`notebooks/01_analise_exploratoria.ipynb`](notebooks/01_analise_exploratoria.ipynb) |
| Resultados e discussão dos gráficos | [`notebooks/02_resultados.ipynb`](notebooks/02_resultados.ipynb) |
| Configurações dos seis experimentos | [`configs/experiments.json`](configs/experiments.json) |
| Busca de taxa de aprendizagem da MLP, separada dos seis modelos finais | `artifacts/search/search_summary.csv` |
| Auditoria e divisão dos dados | `data/audit.json`, `data/splits.csv`, `data/{train,val,test}.csv`, `data/protocol.json` |
| Modelos e registros | `artifacts/runs/<id>/` |
| Seleção congelada pela validação | `artifacts/selection.json` |
| Resumo calculado no teste | `artifacts/summary.csv` |
| Tabelas e figuras reproduzíveis | `artifacts/tables/`, `artifacts/figures/` |

Os seis treinos finais, a avaliação e os dois notebooks estão concluídos no pacote distribuído. A CNN 128 sem pesos, selecionada pela validação (F1 macro 0,633256), obteve acurácia de teste 0,626316 e F1 macro 0,594358 (238 acertos em 380 imagens). O extra YOLO não foi realizado. Consulte [a verificação técnica](docs/VERIFICACAO.md) e [a revisão item a item da rubrica](docs/REVISAO_RUBRICA.md) para os checks realizados e as pendências de entrega. A geração falha se faltarem evidências; não produz resultados simulados. O PDF no AVA é obrigatório: cada integrante deve enviar o mesmo arquivo, com até dez páginas e sem capa separada, até **27/11/2026 às 23h59**, conforme o enunciado. Disponibilizar apenas o GitHub não satisfaz a entrega.

## Instalação

Execute todos os comandos na raiz desta pasta, onde estão `src/` e `configs/`. Os exemplos de ativação abaixo usam Linux/macOS; no Windows, use `.venv\Scripts\Activate.ps1` no PowerShell. O código usa CPU e quatro threads; ter GPU disponível não muda automaticamente o dispositivo.

### Ambiente com versões exatas

O arquivo [`requirements.txt`](requirements.txt) registra as versões utilizadas no ambiente Python 3.14, incluindo PyTorch CPU. `requirements-lock.txt` registra também as dependências transitivas instaladas. Requer Python 3.14 instalado e acesso aos índices de pacotes:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Python, PyTorch, NumPy, sistema, CPU, memória e número de threads efetivamente usados em cada treino também constam de `artifacts/runs/<id>/metrics.json`, campo `hardware`. As versões das demais dependências estão no arquivo de requisitos.

### Alternativa portável: Python 3.11 ou superior

Use um ambiente separado se a plataforma não disponibilizar todas as versões exatas. Esta alternativa permite executar o projeto, mas não promete resultados idênticos aos do ambiente original:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-portable.txt
```

Os requisitos portáveis usam intervalos; o instalador resolve versões compatíveis com o Python escolhido. Sementes fixas e algoritmos determinísticos não garantem identidade entre bibliotecas, sistemas e dispositivos diferentes, conforme a [documentação do PyTorch](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Dados e preparação

Usa-se exclusivamente `dataset-resized.zip`, com 2.527 imagens RGB de **512 × 384 pixels**. Não use o conjunto completo exibido automaticamente no visualizador do Hugging Face: ele reúne variantes e não corresponde ao corpus exigido. As dimensões neste projeto são sempre **largura × altura**; o tensor do PyTorch usa lote × canais × altura × largura.

```bash
python -m src.download --root .
python -m src.prepare --root .
```

O download usa o [arquivo publicado por Gary Thung no Hugging Face](https://huggingface.co/datasets/garythung/trashnet/blob/main/dataset-resized.zip). O SHA-256 esperado do ZIP é:

```text
c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c
```

Se o download automático não estiver disponível, salve esse mesmo ZIP em `data/raw/dataset-resized.zip` e repita os comandos; o script verifica o arquivo existente. Um ZIP divergente é recusado. `prepare` extrai o arquivo, verifica integridade, dimensões, canais, contagens e duplicatas, grava os manifestos e cria caches nas três resoluções. Cada variante é gerada diretamente da imagem de referência, convertida para RGB e redimensionada com Lanczos. Os pixels são armazenados como `uint8` e convertidos para `float32 / 255` por lote. Não há aumento de dados, oversampling ou estatísticas de normalização estimadas nos dados.

O ZIP, as imagens extraídas e os caches **não são incluídos na distribuição do projeto nem versionados**. Os caches ocupam cerca de **1,96 GB decimais / 1,82 GiB**; reserve espaço adicional para imagens, ambiente Python e checkpoints. Os caches são abertos por mapeamento de memória, sem carregar todas as resoluções na RAM de uma vez.

### Divisão auditável

Semente 42; divisão estratificada aproximada de 70%/15%/15%. O arredondamento inteiro resulta em 1.768/379/380 imagens. A ordem dos rótulos, das saídas e das matrizes é fixa:

| Índice | Classe | Total | Treino | Validação | Teste |
|---:|---|---:|---:|---:|---:|
| 0 | glass | 501 | 350 | 75 | 76 |
| 1 | paper | 594 | 416 | 89 | 89 |
| 2 | cardboard | 403 | 282 | 61 | 60 |
| 3 | plastic | 482 | 337 | 72 | 73 |
| 4 | metal | 410 | 287 | 61 | 62 |
| 5 | trash | 137 | 96 | 21 | 20 |
| | **Total** | **2.527** | **1.768** | **379** | **380** |

A auditoria encontrou **três pares de imagens idênticas com rótulos divergentes**. Todos ficaram no treino, mantendo os 2.527 arquivos e seus rótulos originais. Uma troca determinística entre imagens da mesma classe foi necessária: `plastic332.jpg` passou da validação ao treino; `plastic1.jpg` fez o caminho inverso. A troca conserva exatamente as contagens estratificadas. Não há pixels idênticos entre partições. Os pares e hashes estão em `data/audit.json`.

Essa medida evita o vazamento exato identificado, mas preserva ruído de supervisão no treino. Hashes não detectam todas as imagens parecidas ou fotografias diferentes do mesmo objeto. Os manifestos publicados são a referência: não os altere ao avaliar os modelos salvos. Havendo uma seleção congelada, `prepare` compara os manifestos candidatos com os hashes registrados antes de sobrescrever qualquer manifesto. Versões diferentes do scikit-learn podem gerar divisões distintas; uma divergência é recusada, preservando os CSVs publicados. Nesse caso, use o ambiente exato e investigue a diferença antes de continuar.

## Experimentos e treinamento

| ID | Arquitetura | Entrada | Pesos na perda |
|---|---|---|---|
| `mlp_128` | MLP | 128 × 96 | Não |
| `mlp_128_weighted` | MLP | 128 × 96 | Sim |
| `cnn_128` | CNN | 128 × 96 | Não |
| `cnn_128_weighted` | CNN | 128 × 96 | Sim |
| `cnn_256` | CNN | 256 × 192 | Não |
| `cnn_512` | CNN | 512 × 384 | Não |

A MLP usa vetor RGB, camadas de 128 e 64 neurônios, ReLU e dropout 0,3. A CNN usa três blocos com 16/32/64 filtros, BatchNorm, ReLU e pooling, seguidos de global average pooling, dropout 0,3 e saída de seis logits. A CNN mantém **24.198 parâmetros** nas três resoluções. A entrada maior aumenta o custo das ativações e convoluções, mesmo com a mesma quantidade de parâmetros.

O dropout declarado no JSON é aplicado pelo construtor das redes. Outros campos descrevem escolhas fixas desta implementação: Adam, CPU, normalização por 255, Lanczos, ausência de aumento, seleção por F1 macro de validação e perda de validação não ponderada. Valores não implementados nesses campos são recusados; alterar uma descrição no JSON não implementa uma nova técnica.

O protocolo usa Adam, lote 32, até 30 épocas, semente 42 e CPU com quatro threads. A taxa de aprendizagem efetiva de cada arquitetura deve ser consultada em `configs/experiments.json` e no `config.json` de cada treino. A perda é entropia cruzada; nas variantes ponderadas, o peso da classe é `N_treino / (6 × n_classe_treino)`. As demais condições, incluindo a taxa e a inicialização, são preservadas entre cada par com/sem pesos; as três CNNs sem pesos também compartilham a taxa.

A busca separada da MLP compara taxas `1e-3`, `1e-4` e `3e-4` somente pela validação, antes do teste final. Seus registros ficam em `artifacts/search/`, incluindo `search_summary.csv` e `decision.json`; não são confundidos com os seis modelos obrigatórios. Se a taxa escolhida difere da inicial, ela é aplicada aos dois treinos finais da MLP, reiniciados com semente 42, e os resultados iniciais são preservados em `artifacts/search/initial_runs/`. A CNN mantém seu protocolo. Portanto, caso as taxas finais difiram entre arquiteturas, a comparação MLP/CNN representa essas configurações selecionadas: não permite atribuir toda diferença apenas à arquitetura. Essa busca limitada não comprova otimalidade dos hiperparâmetros e seu custo deve ser distinguido do custo dos seis treinos finais.

O checkpoint é selecionado por maior F1 macro de validação; empate usa menor perda de validação. A parada ocorre após sete épocas sem melhora estrita do F1 macro ou no limite de 30 épocas. Um empate que melhora a perda pode atualizar `best.pt`, mas não reinicia a paciência. A perda de validação é sempre não ponderada. Cada experimento começa com modelo, otimizador e geradores reinicializados.

```bash
# Executa os seis; cada experimento usa um processo próprio.
python -m src.train --root .

# Executa ou retoma apenas um ID.
python -m src.train --root . --experiment cnn_512
```

O treinamento usa somente treino e validação. Após concluir os seis, grava `artifacts/selection.json`: maior F1 macro de validação, menor perda em empate e ID como desempate final. Esse arquivo congela os hashes dos seis checkpoints e dos manifestos antes de abrir o teste.

### Repetir a investigação preliminar, opcionalmente

Para reproduzir apenas os seis experimentos finais, use diretamente o `configs/experiments.json` distribuído em uma pasta experimental nova. A busca não é necessária para carregar/avaliar os seis checkpoints publicados.

Para repetir também a investigação da taxa, use outra pasta experimental nova, sem registros de avaliação anterior, e restaure como `configs/experiments.json` a configuração inicial preservada em `artifacts/search/initial_experiments.json`. Preserve o projeto entregue como referência. Nessa nova execução, a sequência é:

```bash
python -m src.download --root .
python -m src.prepare --root .
python -m src.train --root .
python -m src.search --root .
python -m src.train --root .
python -m src.evaluate --root .
python -m src.reporting --root .
python -m src.article --root .
```

Execute cada etapa até concluir, sem treinos concorrentes. `src.search` exige os seis treinos iniciais concluídos e recusa iniciar uma nova busca após encontrar previsões de teste. Se houver mudança de taxa, preserva a seleção inicial em `artifacts/search/initial_selection_before_search.json` e prepara os dois novos treinos da MLP; o segundo `src.train` conclui esses treinos e congela a seleção final. Os checkpoints das CNNs são preservados. A busca é uma etapa anterior ao teste, não uma ação posterior ao fluxo de reprodução da avaliação.

### Retomada e arquivos salvos

Em `artifacts/runs/<id>/`:

- `best.pt`: melhor checkpoint de validação, com pesos, configuração, classes, época e semente; **é distribuído e deve ser incluído no GitHub**.
- `last.pt`: estado completo ao fim da última época concluída, com otimizador e geradores aleatórios; é usado para retomada local e não é incluído no pacote/GitHub.
- `config.json`, `history.csv`, `metrics.json`: configuração, evolução por época, hardware, tempo, memória, época selecionada e hash do modelo.
- `validation_predictions.csv`: previsões do melhor checkpoint na validação.
- Após avaliação: `test_predictions.csv`, `classification_report.csv`, `confusion_matrix.csv` e `confusion_matrix_percent.csv`.

Para retomar uma interrupção, mantenha `last.pt`, `best.pt`, dados e configuração no lugar e execute novamente o mesmo comando. A época parcial pode ser repetida; o tempo ocioso da interrupção não integra os tempos registrados das épocas concluídas. Só retome arquivos `last.pt` produzidos por uma execução confiável deste projeto, pois a carga restaura objetos Python.

Experimentos já concluídos são preservados, com verificação do hash do modelo. Configurações incompatíveis são recusadas. **O pacote publicado não contém `last.pt`: ele permite avaliar os melhores pesos, mas não retomar o otimizador de um treino encerrado.** Para realizar um novo protocolo ou repetir treinos do zero, use outra pasta experimental com os códigos e configurações, sem reaproveitar os registros de execução. Preserve a execução original. Depois de observar o teste, novas escolhas exigem uma avaliação final independente.

## Avaliar os seis modelos sem treinar novamente

Com os seis `best.pt`, históricos, métricas, manifestos e `selection.json` distribuídos no projeto, basta instalar o ambiente e executar:

```bash
python -m src.download --root .
python -m src.prepare --root .
python -m src.evaluate --root .
python -m src.reporting --root .
python -m src.article --root .
```

**Não há chamada a `src.train` nesse fluxo.** `evaluate` verifica hashes e a correspondência entre configurações atuais, registros e checkpoints antes de carregar imagens de teste. Carrega os seis modelos em CPU e recalcula as previsões no mesmo teste. A repetição serve para conferir reprodutibilidade, não para escolher novos hiperparâmetros com base no teste. `reporting` reconstrói tabelas e figuras dos registros. `article` gera `docs/artigo.pdf` e `docs/artigo.md`, usando `docs/metadata.json`; exige dez páginas e sinaliza divergências de paginação. A discussão contém interpretações específicas desta execução: novos experimentos exigem revisão humana dos textos em `src/article.py` e nos notebooks, além da atualização dos números.

Para somente refazer tabelas/figuras com as previsões existentes, execute `python -m src.reporting --root .`. As imagens originais ainda são necessárias para os painéis de exemplos e erros. Para apenas análise exploratória, antes dos treinos, use `python -m src.reporting --root . --eda-only`.

Abra os notebooks em um editor Jupyter/VS Code, selecione o Python do ambiente instalado e execute todas as células na ordem. Eles procuram a raiz do projeto em sua pasta e pastas ancestrais. O notebook de resultados reconstrói e confere os registros; não realiza treinamento.

## Como ler as medidas

O F1 macro é a média simples do F1 das seis classes e é a métrica principal de seleção. Acurácia é a fração de acertos. Os relatórios por classe incluem precisão, recall, F1 e suporte, com `zero_division=0` para razões indefinidas. Todas as matrizes têm classe real nas linhas, prevista nas colunas e percentuais calculados por linha. As saídas softmax dos exemplos de erro não foram calibradas.

Os tempos são somas de `perf_counter` por época, incluindo treino, carregamento de lotes e validação; excluem download, preparação prévia, inicialização, gravação dos checkpoints e exportação final de previsões. Tempo médio é total dividido pelas épocas executadas. O pico RSS do processo é amostrado a cada 100 ms durante o laço, incluindo salvamentos, bibliotecas e páginas dos caches mapeadas. Trata-se de uma aproximação em MiB, não de memória exclusiva da rede. Não há medida de VRAM porque os treinos usam CPU. Os processos separados reduzem interferência da memória retida por experimentos anteriores; condições de carga da máquina ainda afetam tempos.

Nos experimentos ponderados, treino e validação usam perdas com definições distintas. A acurácia de treino é acumulada durante as atualizações com dropout ativo; a validação é calculada após a época, em modo de avaliação. Leia tendências e desempenho de validação, sem tratar uma diferença isolada entre curvas como prova de sobreajuste.

Para verificar as invariantes implementadas, sem novos treinos reais:

```bash
python -m pytest -q
```

## Preparar a publicação

1. Revisar o material com o [guia](docs/GUIA_DA_EQUIPE.md), substituir os integrantes/matrículas provisórios e criar um repositório público na conta da equipe.
2. Inserir a URL real em `docs/metadata.json`, campo `github_url`; revisar `title`, `authors`, `access_date` e `note`. Regenerar o artigo com `python -m src.article --root .` e conferir primeira página, tabelas, figuras e dez páginas.
3. Conferir a presença dos seis `artifacts/runs/<id>/best.pt`, além da seleção, configurações, históricos, métricas e previsões. O `.gitignore` mantém `best.pt` e exclui dados brutos, caches, ambiente, temporários e `last.pt`.
4. Dentro da raiz `trashnet`, executar os comandos abaixo, substituindo `SUA_CONTA/SEU_REPOSITORIO` pela URL real. Eles são instruções para a equipe; a criação/publicação não foi realizada por este pacote.

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Adiciona estudo reproduzível de classificação TrashNet"
git remote add origin https://github.com/SUA_CONTA/SEU_REPOSITORIO.git
git push -u origin main
```

Revise o que foi preparado antes do commit. Depois do envio, abra o repositório sem sessão autenticada, confira o download dos seis modelos e teste o fluxo de avaliação em uma cópia limpa. Envie a mesma versão final do PDF no AVA por todos os integrantes. Não há extensão YOLO nesta implementação; o bônus é opcional e não é reivindicado.

## Fontes, atribuição e licenças

O TrashNet foi criado por **Gary Thung e Mindy Yang**. O [repositório original](https://github.com/garythung/trashnet) disponibiliza uma [licença MIT](https://github.com/garythung/trashnet/blob/master/LICENSE), e a [página do dataset no Hugging Face](https://huggingface.co/datasets/garythung/trashnet) identifica a mesma licença. Preserve a atribuição e os avisos de licença dos materiais de terceiros ao redistribuí-los. O corpus permanece disponível na origem; imagens reproduzidas nas figuras são atribuídas ao TrashNet. A licença externa não atribui à equipe autoria das imagens nem define automaticamente uma licença para todo este projeto. Bibliotecas mantêm suas próprias licenças.

- YANG, Mindy; THUNG, Gary. [Classification of trash for recyclability status](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf). Stanford, 2016. A referência relata 63% com SVM/seis classes e 22% com CNN/cinco classes; os aproximadamente 75% do README são posteriores. Protocolos diferentes impedem uma comparação experimental direta.
- PASZKE, Adam et al. [PyTorch: an imperative style, high-performance deep learning library](https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html). NeurIPS, 2019.
- PEDREGOSA, Fabian et al. [Scikit-learn: machine learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html). JMLR, v. 12, p. 2825–2830, 2011.
- KINGMA, Diederik P.; BA, Jimmy. [Adam: a method for stochastic optimization](https://arxiv.org/abs/1412.6980). ICLR, 2015.
- Documentação: [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html), [métricas por classe](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) e [prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html).

O artigo contém as referências utilizadas em seu texto. As limitações incluem apenas uma semente/divisão, conjunto pequeno e desbalanceado, rótulos conflitantes e imagens com objetos isolados/fundos controlados. Os resultados não demonstram desempenho em cenários reais de coleta nem superioridade estatística universal de uma arquitetura.
