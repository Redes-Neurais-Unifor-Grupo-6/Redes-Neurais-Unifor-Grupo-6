# Guia de revisão e defesa do projeto

Este guia organiza a revisão que a equipe ainda precisa realizar. Não declara que os integrantes já compreenderam o código, tomaram decisões ou validaram o texto. Use os arquivos do projeto e os resultados efetivamente calculados para preparar as respostas; evite decorar conclusões sem conseguir mostrar sua evidência.

**Identificação atual — Grupo 1 (30 out. 2026):** o usuário informou que renomeou a organização e o repositório já existentes para [Redes-Neurais-Unifor-Grupo-1](https://github.com/Redes-Neurais-Unifor-Grupo-1/Redes-Neurais-Unifor-Grupo-1). Esta atualização local de identificação e link ainda precisa ser publicada. O acesso externo e o conteúdo remoto não foram reconferidos nesta atualização. Nomes e matrículas permanecem provisórios; a revisão humana e a entrega no AVA continuam pendentes.

> **Regra da atividade:** IA pode apoiar a geração de código, mas o enunciado atribui nota zero à delegação de 100% do projeto à IA. A equipe deve revisar e compreender o código, conduzir as decisões metodológicas e assumir análise e interpretação. A entrega também exige o PDF no AVA por cada integrante; apenas publicar código não basta.

## 1. O que o estudo testa

Há três comparações: MLP versus CNN em 128 × 96; perda sem/com pesos em cada arquitetura nessa resolução; CNN sem pesos em 128 × 96, 256 × 192 e 512 × 384. Cada experimento reinicia modelo, otimizador e geradores. Semente, partição, lote e política de parada permanecem comuns. A taxa é preservada dentro de cada par com/sem pesos e entre resoluções da CNN, mas pode diferir entre MLP e CNN após a seleção por validação. As seis classes são mantidas.

A equipe deve ser capaz de relacionar cada afirmação do artigo a uma tabela, curva ou registro. Uma melhora numérica neste teste é uma observação desta execução. Com uma semente e uma divisão, não há base para afirmar superioridade geral ou significância estatística.

## 2. Caminho dos dados e prevenção de vazamento

1. `src.download` obtém o ZIP de referência e confere seu SHA-256.
2. `src.prepare` verifica 2.527 imagens RGB, 512 × 384, integridade e duplicatas; faz a divisão estratificada com semente 42 e grava os manifestos.
3. Cada resolução é derivada diretamente da referência. RGB e divisão por 255 são transformações fixas; não há média/desvio estimados no conjunto completo.
4. Treino ajusta pesos; validação escolhe checkpoints e modelo; teste mede desempenho somente depois de congelar a seleção.

Os tamanhos são 1.768/379/380. Os CSVs guardam caminhos, classes e hashes; não se deve refazer uma divisão conveniente após observar métricas. A ordem é **glass, paper, cardboard, plastic, metal, trash**, com índices de 0 a 5.

Foram encontrados três pares de duplicatas exatas com rótulos conflitantes. Todos estão no treino. Apenas uma troca foi necessária: `plastic332.jpg` foi para treino e `plastic1.jpg` para validação, preservando as contagens de `plastic`. A regra usa hashes, classe e ordem determinística, sem resultados de modelos. Manter os rótulos preserva o corpus solicitado, mas mantém ruído na supervisão. Isso deve ser discutido como decisão e limitação. Não representa um teto direto da acurácia do teste, pois os pares conflitantes não estão no teste. Fotografias quase iguais ou do mesmo objeto podem continuar sem identificação.

**Revisão prática:** localizar os pares em `data/audit.json`, conferir os caminhos em `data/splits.csv` e explicar por que a separação de uma cópia no treino e outra no teste inflaria a avaliação.

## 3. Arquiteturas e escolhas de treinamento

A MLP transforma cada imagem 128 × 96 RGB em 36.864 entradas. Camadas lineares de 128 e 64 unidades com ReLU e dropout 0,3 produzem seis logits. Ela não impõe a estrutura local da convolução; sua primeira camada conecta todos os pixels a todos os neurônios.

A CNN usa convoluções 3 × 3 com 16, 32 e 64 filtros, BatchNorm, ReLU e pooling. O primeiro bloco tem stride 2. Global average pooling converte cada mapa final em um valor, de modo que a camada classificadora sempre recebe 64 entradas. Isso mantém 24.198 parâmetros nas três resoluções; as operações e ativações intermediárias ainda crescem com a entrada. A equipe deve explicar esse custo, sem confundir número de parâmetros com tempo ou RAM.

Adam usa lote 32 e nenhum weight decay; máximo de 30 épocas e paciência 7. A taxa efetiva de cada arquitetura está em `configs/experiments.json` e nos registros dos treinos. Dropout e parada por validação limitam sobreajuste, mas não o eliminam por definição. Não há pré-treinamento nesta comparação.

A busca separada da MLP compara `1e-3`, `1e-4` e `3e-4`, com escolha por F1 macro de validação e teste ainda fechado. Consulte `artifacts/search/search_summary.csv` e `decision.json`. Se a taxa muda, os dois treinos finais da MLP são reiniciados com semente 42 e os registros iniciais são preservados; a CNN mantém seu protocolo. A equipe deve verificar a taxa efetivamente escolhida e explicar que a busca é limitada, sem demonstrar otimalidade global. Se as arquiteturas usam taxas diferentes, a diferença final não pode ser atribuída causalmente apenas à arquitetura. O custo da busca deve ser discutido separadamente dos seis treinos finais. A reprodução opcional dessa investigação exige a configuração inicial e uma nova pasta experimental, antes da avaliação de teste, conforme o README.

A entropia cruzada recebe logits; softmax é aplicado para exportar probabilidades, não antes da função de perda. Nos modelos ponderados, `peso_c = 1768 / (6 × quantidade_c_no_treino)`. O peso maior de `trash` aumenta sua contribuição na perda, sem criar imagens novas. Ele pode melhorar recall e prejudicar precisão ou outras classes; a resposta depende dos resultados reais. Os pesos são calculados só no treino.

## 4. Checkpoint, seleção e teste

Dentro de cada treino, maior F1 macro de validação define `best.pt`; empate usa menor perda de validação. A paciência conta épocas sem melhora estrita do F1 e não é reiniciada apenas por empate com perda menor. A época salva pode ser anterior à última época executada.

Após os seis treinos, `artifacts/selection.json` registra o modelo selecionado por F1 macro de validação, desempates e hashes. O arquivo é anterior à avaliação de teste. `src.evaluate` exige essa seleção e verifica os hashes. Mudar a seleção depois de comparar métricas de teste seria usar o teste para decidir.

`last.pt` permite retomar um treino interrompido com estado do otimizador e dos geradores. `best.pt` permite inferência e é o modelo distribuído. Eles têm objetivos diferentes. Não reiniciar a partir de `best.pt` fingindo que é uma continuação idêntica: faltaria o estado completo da otimização.

**Revisão prática:** abrir `selection.json`, localizar o vencedor entre as linhas de validação e carregar os seis modelos pelo comando de avaliação do README. A equipe deve explicar por que o maior resultado de teste não determina o modelo da análise de erros.

## 5. Métricas sem ambiguidades

Para uma classe, VP significa verdadeiros positivos, FP falsos positivos e FN falsos negativos:

| Métrica | Cálculo | Leitura |
|---|---|---|
| Acurácia | acertos / total | Fração de imagens corretamente classificadas. |
| Precisão | VP / (VP + FP) | Entre as previsões dessa classe, quantas estavam corretas? |
| Recall | VP / (VP + FN) | Entre os exemplos reais dessa classe, quantos foram reconhecidos? |
| F1 | 2VP / (2VP + FP + FN) | Compromisso entre precisão e recall. |
| F1 macro | soma dos seis F1 / 6 | Dá o mesmo peso a cada classe, independentemente do suporte. |
| Suporte | número de exemplos reais | Quantidade avaliada da classe, não quantidade de previsões. |

`trash` tem apenas 20 imagens no teste: acertar mais uma muda seu recall em **5 pontos percentuais**. Isso ilustra a sensibilidade da medida a poucos exemplos. `zero_division=0` define zero quando uma razão seria indefinida; se o modelo não prevê uma classe, não se deve interpretar a precisão zero como ausência de exemplos reais.

Nas matrizes, reais estão nas linhas e previsões nas colunas. O percentual de cada célula é dividido pelo total da linha. A diagonal dá acertos; normalizar por linha ajuda a comparar classes de tamanhos diferentes. A soma dos suportes deve ser 380 para todos os modelos.

Nos erros ilustrados, a probabilidade é a saída softmax atribuída à classe prevista. Ela não foi calibrada. Um erro com probabilidade 0,95 não comprova que o modelo teria 95% de chance de estar certo em situações equivalentes. O painel escolhe até seis erros mais confiantes do modelo selecionado pela validação, com desempate pelo caminho; portanto, não é uma amostra aleatória de todas as falhas.

## 6. Ler curvas e custo corretamente

A acurácia de treino é acumulada durante as atualizações, com dropout ativo; validação ocorre depois da época em modo de avaliação. Nas execuções ponderadas, a perda de treino tem pesos, mas a de validação é não ponderada. Por isso, diferenças verticais isoladas não provam sobreajuste. Examine tendências: treino melhora enquanto validação estagna/piora? O melhor checkpoint precede o fim? A falha afeta todas as classes ou concentra-se em poucas?

Tempo total soma treino e validação das épocas concluídas; média divide esse total pelo número de épocas. Download, caches, inicialização, salvamento de checkpoints e exportação final ficam fora dessa medida. Compare total e média: early stopping pode encerrar um modelo antes, mesmo que cada época seja mais cara. Interrupções não contam como tempo ocioso; uma época interrompida pode ser refeita a partir do último estado completo.

O RSS é memória residente do processo, amostrada a cada 100 ms e expressa em MiB. Inclui bibliotecas, modelo, otimizador, páginas dos caches e o período de salvamento. Pode perder picos breves. A medição não equivale ao tamanho de `best.pt`, nem à memória total da máquina. Os treinos em CPU não têm custo de VRAM. A comparação de tempo depende da carga da máquina e não constitui benchmark universal.

## 7. Discussão que a equipe deve conferir nos resultados

Preencha suas conclusões consultando `artifacts/tables/experiment_comparison.csv`, `per_class_comparison.csv`, `computational_cost.csv`, figuras e históricos. Não use esta lista como conclusões prontas.

- **MLP versus CNN:** comparar os pares 128 × 96 equivalentes, com e sem pesos. Anotar F1 macro, acurácia e classes que explicam a diferença. Verificar se as curvas sustentam a hipótese de sobreajuste ou dificuldade de otimização.
- **Pesos de classe:** em cada arquitetura, quantificar a mudança de precisão, recall e F1 de `trash`; verificar perdas e ganhos em todas as demais classes. Acurácia menor pode acompanhar F1 macro maior, ou ambos podem cair.
- **Resolução:** registrar a mudança de F1/acurácia contra tempo por época, tempo total e pico RSS. Explicar o que significa “compensar o custo” para uma aplicação concreta, sem assumir que mais pixels sempre ajudam.
- **Erros:** partir das maiores confusões direcionais da matriz e das imagens exibidas. Transparência, reflexo, textura e formato são hipóteses visuais; os resultados não provam essas causas.
- **Aplicação real:** discutir fundos variados, múltiplos objetos, oclusão, sujeira e iluminação ausentes ou pouco representados. Este é um classificador da imagem inteira; ele não localiza objetos e não é um detector.
- **Referência:** Yang e Thung relatam 63% com SVM em seis classes e 22% com CNN em cinco. O README apresenta cerca de 75% em uma continuação posterior. Classes, divisões, resolução e treinamento diferentes impedem declarar uma comparação controlada com esses números.

## 8. Perguntas para a defesa

| Pergunta provável | O que mostrar para sustentar a resposta |
|---|---|
| Por que usar F1 macro se já existe acurácia? | Distribuição desigual e métricas por classe; explicar peso igual das classes. |
| Como sabem que não houve vazamento exato? | Hashes, manifestos, três pares, troca determinística e ausência de pixels idênticos entre splits. |
| Por que manter rótulos conflitantes? | Preservação do corpus de referência, registro transparente do ruído e alternativas para estudo posterior. |
| Os pesos foram calculados no teste? | `class_weights` em `src/common.py` e rótulos de `train.csv`. |
| A CNN maior tem mais parâmetros? | Código com global average pooling; mesma contagem, custo de ativações diferente. |
| Por que essas arquiteturas e hiperparâmetros? | Configurações, capacidade, custo e busca limitada da MLP por validação; reconhecer que ela não comprova otimalidade. |
| A comparação MLP/CNN isola apenas a arquitetura? | Conferir as taxas de aprendizagem finais; taxas diferentes tornam a comparação entre configurações, sem causalidade exclusiva da arquitetura. |
| Qual modelo foi escolhido e em qual época? | `selection.json`, `metrics.json`, `history.csv`; usar valores reais. |
| Por que não escolher o maior F1 do teste? | Separar ajuste/seleção da estimativa final; explicar viés de usar teste para decisões. |
| Como provar que o modelo salvo funciona? | Rodar `src.evaluate` sem `src.train` e comparar previsões/métricas. |
| Por que treino e validação têm perdas tão diferentes? | Pesos de classe e modos treino/avaliação; ler tendências com essas convenções. |
| Probabilidade alta significa previsão confiável? | Softmax não calibrada e exemplos de erros confiantes. |
| O projeto serviria em uma esteira de reciclagem? | Mudança de distribuição, múltiplos objetos, latência e necessidade de validação independente. |
| O que a equipe efetivamente revisou ou mudou? | Relatar somente ações realizadas, com arquivos, resultados e decisões verificáveis. |

## 9. Registro de revisão humana

Preencher após executar as atividades, com nomes reais. Campos vazios são pendências, não evidência de revisão.

| Atividade | Responsável | Evidência/decisão | Data |
|---|---|---|---|
| Conferir corpus, divisão e duplicatas | A preencher | A preencher | A preencher |
| Explicar e revisar MLP/CNN | A preencher | A preencher | A preencher |
| Conferir seleção e reproduzir avaliação | A preencher | A preencher | A preencher |
| Interpretar classes minoritárias e custos | A preencher | A preencher | A preencher |
| Revisar artigo, referências e figuras | A preencher | A preencher | A preencher |
| Publicar e testar download dos modelos | A preencher | A preencher | A preencher |

Decisões a revisar incluem manter rótulos originais conflitantes, treinar sem aumento de dados, usar uma única semente e priorizar seis comparações controladas. Se a equipe decidir novos experimentos após ver o teste, deve preservá-los como estudo separado e prever avaliação final independente; não editar retroativamente o protocolo desta execução.

## 10. Conferência final antes da entrega

- [ ] Substituir `Lorem Ipsum` e `9999999` pelos dados reais dos quatro a seis integrantes em `metadata.json`.
- [x] Registrar o Grupo 1 e o endereço do repositório já existente informado pelo usuário em 30 out. 2026.
- [ ] Publicar a atualização local e conferir o acesso externo ao repositório, ao PDF e aos modelos.
- [ ] Conferir seis `best.pt`, configurações, históricos, métricas, previsões e seleção congelada.
- [ ] Conferir todos os valores e conclusões do artigo contra os registros.
- [ ] Regenerar o PDF; verificar até dez páginas, uma coluna, título, nomes, matrículas, resumo e link na primeira página, sem capa separada.
- [ ] Conferir legibilidade de tabelas, legendas e painéis; revisar citações e referências.
- [ ] Testar avaliação sem retreino em uma cópia limpa e acesso público aos modelos.
- [ ] Manter dados brutos, caches, `.venv` e `last.pt` fora do Git; preservar os seis `best.pt`.
- [ ] Registrar honestamente o uso de IA e as revisões/decisões realmente realizadas, conforme a orientação do professor.
- [ ] Cada integrante enviar o mesmo PDF no AVA até 27/11/2026 às 23h59.

## Fontes para estudar e conferir

As referências completas e orientações de execução estão no [README](../README.md) e no artigo. Para a revisão do método, consulte as fontes primárias:

- [TrashNet e descrição da coleta](https://github.com/garythung/trashnet), [licença externa MIT](https://github.com/garythung/trashnet/blob/master/LICENSE) e [dataset oficial no Hugging Face](https://huggingface.co/datasets/garythung/trashnet).
- [Relatório de Yang e Thung (2016)](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf).
- [PyTorch: entropia cruzada e pesos](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), [global average pooling adaptativo](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html) e [reprodutibilidade](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
- [scikit-learn: relatório de classificação](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [pesos balanceados](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) e [vazamento de dados](https://scikit-learn.org/stable/common_pitfalls.html).
- [Adam, Kingma e Ba](https://arxiv.org/abs/1412.6980).

O artigo segue a estrutura específica exigida pelo professor; essa estrutura deve ser conferida junto às orientações institucionais aplicáveis de ABNT. Não declarar revisão integral de normas ou revisão humana que ainda não ocorreu.
