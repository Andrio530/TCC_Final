# TCC – Unisinos (abnTeX2)

Predição dinâmica de falhas em transformadores de potência. Template abnTeX2 adaptado para a Unisinos.

## Requisitos

- VS Code
- LaTeX Workshop (extensão)
- MiKTeX
- Strawberry Perl (necessário para o `latexmk` no Windows)

## Instalação

    winget install MiKTeX.MiKTeX --scope user
    winget install StrawberryPerl.StrawberryPerl --scope user

Na primeira compilação o MiKTeX baixa sozinho os pacotes que faltarem (abntex2, abntex2cite etc.).
Se aparecer pedido de instalação, permita.

## Compilação

Abra a pasta no VS Code e pressione **Ctrl + Alt + B**.

Pelo terminal:

    latexmk -pdf main.tex

Limpar arquivos auxiliares:

    latexmk -c

## Compilação automática (GitHub Actions)

A cada push o workflow `.github/workflows/build-pdf.yml` compila o `main.tex` e disponibiliza o PDF
na aba **Actions → execução → Artifacts → TCC-pdf**.

## Estrutura

- `main.tex` – documento principal
- `pretextual.tex`, `introducao.tex`, `fundamentacao.tex`, `rev_bibliografica.tex`, `apendices.tex`, `anexos.tex` – capítulos
- `referencias.bib` – bibliografia
- `unisinos.sty` – customização Unisinos
- `figuras/` – imagens. Os diagramas em SVG têm uma versão `.pdf` (usada no documento);
  se editar o SVG no draw.io, exporte de novo como PDF com o mesmo nome.
