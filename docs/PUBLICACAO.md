# Publicação gradual

## Organização

`main` contém a versão estável. `develop` integra as etapas. As cinco branches `feature/01-conhecimento`, `feature/02-inferencia`, `feature/03-interface`, `feature/04-validacao` e `feature/05-documentacao` preservam os marcos de cada grupo de funcionalidades. Integrações em develop são fast-forward; a versão final é integrada à main com commit de merge.

Não foram alteradas datas de commits. Cada commit corresponde ao trabalho efetivamente produzido na etapa. O envio posterior ao repositório oficial deve ser apresentado como integração da implementação existente, preservando a rastreabilidade.

## Repositórios

- Desenvolvimento completo: `leucimar/RepositorioTemporario` (privado).
- Entrega: `leucimar/Atividade08-Construcao-Sistema-de-Carreiras-de-TI` (público).

Por instrução do aluno, apenas o commit inicial de planejamento será publicado no oficial nesta rodada. Próximas etapas dependem de novas instruções; não há agendamento de commits.

| Ordem | Origem no temporário | Conteúdo a integrar |
|---|---|---|
| Inicial | main inicial | README com objetivo e etapas |
| 1 | feature/01-conhecimento | Catálogo, estrutura e matriz |
| 2 | feature/02-inferencia | Fatos, regras, evidências e dependências |
| 3 | feature/03-interface | Entrevista Streamlit |
| 4 | feature/04-validacao | Personas, testes e identificação estável dos botões |
| 5 | feature/05-documentacao | README completo, validação e ambiente reproduzível |

Ao retomar, compare as branches reais dos dois repositórios antes de transferir. Não publique toda a árvore final como se fosse a primeira funcionalidade. Os primeiros marcos podem ser incompletos e devem permanecer em develop até a versão executável ser integrada à main.

Uma transferência local possível é buscar o remoto temporário e aplicar os commits de implementação, em ordem, via cherry-pick (sem reaplicar o README inicial). Alternativamente, o plugin GitHub pode transferir cada conjunto de arquivos em um novo commit de integração. Em ambos os casos, registre no commit a etapa e o SHA de origem. Não force atualizações nem reescreva o histórico.
