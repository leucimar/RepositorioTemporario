# Integração gradual da Etapa 2

## Estado

O oficial `leucimar/Trabalho_Etapa2_IA_Upgrade_Sistema-Especialista-Carreira-TI` tem apenas o marco inicial: cópia integral da árvore da Atividade 08 no commit `c91b220a3803dd1306e5f9bc737c5185e581a2b8`.

Trata-se de uma cópia rastreável com um novo commit raiz, sem importar os seis commits antigos. Não é um fork nativo vinculado pela interface do GitHub: o destino já foi criado pelo aluno. A árvore inicial foi comparada à origem para garantir igualdade.

No temporário, as branches `etapa2/main` e `etapa2/develop` e as features `feature/e2-*` compõem uma linha com raiz própria, cujo primeiro commit é a mesma base completa. As branches antigas `main`, `develop` e `feature/01-*` até `feature/05-*` da AECO foram preservadas.

## Ordem de integração no oficial

| Ordem | Branch no temporário | Conteúdo |
|---|---|---|
| 0 | etapa2/main inicial | Atividade 08 completa, sem alterações |
| 1 | feature/e2-01-conhecimento | Base JSON, esquema e decisões de modelagem |
| 2 | feature/e2-02-inferencia | Validação da base e motor ponderado com penalidades |
| 3 | feature/e2-03-roadmap | Gaps, pré-requisitos, esforço e exportação Markdown |
| 4 | feature/e2-04-interface | Wizard Streamlit, gráfico, plano e preservação do app antigo |
| 5 | feature/e2-05-validacao | Personas, testes de robustez, retenção da navegação e avisos de empates |
| 6 | feature/e2-06-documentacao | Guia para o PC, README e evidências de validação |

As integrações no oficial dependem das próximas solicitações do aluno. Não há agendamento. Cada etapa deve produzir um commit com as diferenças daquele marco, sem publicar a árvore final antecipadamente. Registre o SHA de origem no corpo do commit de integração.

Ao integrar a última etapa no oficial, atualize o README, o guia e este documento para refletir que a versão completa está no novo oficial e que o clone deve usar sua URL. No temporário, as instruções apontam corretamente para `etapa2/main`.

Datas de commits representam trabalho ou integração efetivamente realizados, sem retrodatação. Não reescreva o histórico existente para simular outra cronologia.
