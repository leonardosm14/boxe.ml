#import "@preview/cetz:0.3.4": canvas, draw
#figure(
  canvas(length: 3em, {
    import draw: *

    let c-input = rgb("#BFDBFE")  // azul claro
    let c-lstm  = rgb("#A5B4FC")  // azul-roxo
    let c-drop  = rgb("#C4B5FD")  // roxo claro
    let c-attn  = rgb("#D8B4FE")  // roxo médio
    let c-res   = rgb("#E879F9")  // roxo-rosa vibrante
    let c-norm  = rgb("#F0ABFC")  // lilás-rosa
    let c-pool  = rgb("#F9A8D4")  // rosa médio
    let c-dense = rgb("#FBA4B8")  // rosa-salmão
    let c-out   = rgb("#FDA4AF")  // rosa avermelhado

    let box-h  = 1.1
    let box-w  = 1.6
    let gap    = 0.35
    let row-gap = 0.5

    // Linha 0 (esquerda → direita): Input até Residual Add
    let row0 = (
      ("Input\n(25×102)",   c-input),
      ("Bi-LSTM\n(64 un.)", c-lstm),
      ("Dropout\n(p=0.30)", c-drop),
      ("Multi-Head\nAttention", c-attn),
      ("Residual\nAdd",     c-res),
    )

    // Linha 1 (direita → esquerda): Layer Norm até Softmax
    let row1 = (
      ("Layer\nNormalization",        c-norm),
      ("Average\nPool",          c-pool),
      ("Dense\n(64, ReLU)",  c-dense),
      ("Dropout\n(p=0.30)",  c-drop),
      ("Softmax\n(3 classes)",   c-out),
    )

    let y0 = 0.0
    let y1 = -(box-h + row-gap)

    // --- Desenha linha 0 ---
    for (i, (label, color)) in row0.enumerate() {
      let x = i * (box-w + gap)
      rect((x, y0), (x + box-w, y0 + box-h),
        fill: color, stroke: black, radius: 0.15)
      content((x + box-w / 2, y0 + box-h / 2),
        text(size: 0.55em, fill: black, align(center, label)))
      if i < row0.len() - 1 {
        line((x + box-w, y0 + box-h / 2),
             (x + box-w + gap, y0 + box-h / 2),
             mark: (end: ">", size: 0.2), stroke: black)
      }
    }

    // --- Seta de descida: Residual Add → Layer Norm ---
    let res-x = 4 * (box-w + gap)
    let res-cx = res-x + box-w / 2
    let norm-x = 4 * (box-w + gap)
    let norm-cx = norm-x + box-w / 2

    line(
      (res-cx, y0),
      (res-cx, y1 + box-h),
      mark: (end: ">", size: 0.5),
      stroke: black,
    )

    // --- Desenha linha 1 (da direita para esquerda) ---
    for (i, (label, color)) in row1.enumerate() {
      let x = (4 - i) * (box-w + gap)
      rect((x, y1), (x + box-w, y1 + box-h),
        fill: color, stroke: black, radius: 0.15)
      content((x + box-w / 2, y1 + box-h / 2),
        text(size: 0.55em, fill: black, align(center, label)))
      if i < row1.len() - 1 {
        let next-x = (4 - (i + 1)) * (box-w + gap)
        line(
          (x, y1 + box-h / 2),
          (next-x + box-w, y1 + box-h / 2),
          mark: (end: ">", size: 0.2),
          stroke: black,
        )
      }
    }

    // --- Skip connection: Dropout(i=2, row0) → Residual Add(i=4, row0) ---
    let skip-start = 2 * (box-w + gap) + box-w / 2
    let skip-end   = 4 * (box-w + gap) + box-w / 2
    let skip-y     = y0 + box-h + 0.35

    line((skip-start, y0 + box-h), (skip-start, skip-y),
      stroke: (paint: rgb("#0D47A1"), dash: "dashed"))
    line((skip-start, skip-y), (skip-end, skip-y),
      stroke: (paint: rgb("#0D47A1"), dash: "dashed"))
    line((skip-end, skip-y), (skip-end, y0 + box-h),
      mark: (end: ">", size: 0.2),
      stroke: (paint: rgb("#0D47A1"), dash: "dashed"))
    content(
      ((skip-start + skip-end) / 2, skip-y + 0.2),
      text(size: 0.5em, fill: rgb("#0D47A1"), style: "italic", [skip connection]),
    )
  })
) <fig-modelo-arch>