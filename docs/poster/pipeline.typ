#import "@preview/cetz:0.3.4": canvas, draw

#figure(
  canvas(length: 2.2em, {
    import draw: *

    let colors = (
      rgb("#BFDBFE"), // Entrada
      rgb("#93C5FD"), // Conversão FPS
      rgb("#A5B4FC"), // Extração Esqueletos
      rgb("#C4B5FD"), // Tracking
      rgb("#D8B4FE"), // Atribuição Boxeadores
      rgb("#E879F9"), // Preenchimento Denso
      rgb("#F0ABFC"), // Segmentação
      rgb("#F9A8D4"), // Classificação
      rgb("#FBA4B8"), // Lead/Rear
      rgb("#FDA4AF"), // Renderização
    )

    // (label, ferramenta ou "")
    let row0-stages = (
      ("Vídeo de\nEntrada", ""),
      ("Conversão\n25 FPS", "FFmpeg"),
      ("Extração de\nEsqueletos", "YOLOv8m-Pose"),
      ("Tracking\nMulti-Pessoa", "ByteTrack"),
      ("Atribuição de\nBoxeadores", ""),
    )
    let row1-stages = (
      ("Preenchimento\nDenso", ""),
      ("Segmentação\nde Golpes", ""),
      ("Classificação\ndo Golpe", "Modelo ML"),
      ("Inferência\nLead / Rear", ""),
      ("Renderização\nFinal", ""),
    )

    let step = 2.0    // espaçamento horizontal entre nós
    let r = 0.32      // raio do nó
    let row-gap = 1.6 // distância vertical entre as duas linhas

    let y0 = 0.0
    let y1 = -row-gap
    let n  = row0-stages.len()   // 5 em cada linha
    let width = (n - 1) * step

    // --- linha central: linha 0 (esquerda -> direita) ---
    line((0, y0), (width, y0), stroke: 1.2pt + black)

    // --- linha central: linha 1 (direita -> esquerda) ---
    line((width, y1), (0, y1), stroke: 1.2pt + black,
      mark: (end: ">", size: 0.35))

    // --- conector em "S": fim da linha 0 desce até o início da linha 1 ---
    line((width, y0), (width, y1), stroke: 1.2pt + black,
      mark: (end: ">", size: 0.35))

    // função auxiliar pra desenhar uma linha de nós
    let draw-row(stages, y, x-of, col-offset, above-first) = {
      for (i, (label, tool)) in stages.enumerate() {
        let x = x-of(i)
        let above = (stages == row0-stages)
    
        circle((x, y), radius: r, fill: colors.at(i + col-offset), stroke: black)
        content((x, y), text(size: 0.6em, fill:black, weight: "bold", str(i + 1 + col-offset)))
    
        // distância do título até a bola (reduz -> aproxima)
        let label-dist = 0.8
        let label-y = if above { y + label-dist } else { y - label-dist }
    
        // até onde a linha vai (menor que label-dist deixa um gap antes do texto)
        let line-dist = 0.5
        let line-y = if above { y + line-dist } else { y - line-dist }
    
        line((x, y + (if above { r } else { -r })), (x, line-y),
          stroke: 1pt + black)
    
        content((x, label-y),
          text(size: 0.45em, fill: black, align(center, label)))
    
        if tool != "" {
          let tool-y = if above { y - 0.5 } else { y + 0.5 }
          content((x, tool-y),
            text(size: 0.38em, fill: rgb("#0D47A1"), style: "italic", tool))
        }
      }
    }

    // linha 0: esquerda -> direita, começa com rótulo acima
    draw-row(row0-stages, y0, i => i * step, 0, 0)

    // linha 1: direita -> esquerda (espelha o eixo x), continua numeração 6-10
    draw-row(row1-stages, y1, i => width - i * step, n, 1)
  })
) <fig-pipeline-boxe>