# 開発メモ — Fluid Lab

作品としての説明（概要・見どころ・技術解説）は [../README.md](../README.md) にあります。
ここは実行・ビルド・公開の手順です。

## 実行方法

`fluid-lab.html` をブラウザで開くだけ。WebGL2 + `EXT_color_buffer_float` があるとフル品質(GPU海洋シミュレーション・高解像度地形)、ない環境ではCPUフォールバックで動作する。

## ディレクトリ / ビルド

シミュレーションの実体はJavaScript(+文字列埋め込みのGLSL)。Artifactの制約(厳格CSP・外部ファイル禁止)のため配布物は単一HTMLで、読めるソースは `src/` に分割してある。編集は `src/` で行い、`node build.js` で `fluid-lab.html` を組み立てる(単純なスプライスなので出力は決定的)。

```
build.js                  src/ → fluid-lab.html を組み立て(依存なし)
fluid-lab.html            ビルド成果物(公開する単一ファイル)
index.html                GitHub Pages の紹介ページ(手書き・ビルド対象外)
assets/                   紹介ページの画像(各モードの実写)とフォントCSS
src/
  template.html           HTML骨格(タイトル・マークアップ・挿入位置)
  style.css               UIスタイル+Chakra Petchフォント(data URI)
  00-shared.js            IIFE開始・GLヘルパー(glProgram等)・色変換
  10-ink2d.js             インク: 2D Stable Fluids(GPU、渦度強化)
  20-water2d.js           水2D: FLIP/PICクラス+メタボール水面
  30-water3d-earth.js     ★根幹。水3D+地球の共通ファクトリ:
                            FlipFluid3(3D FLIP)、SSFRパイプライン、
                            地形生成(GPUノイズ+川彫り)、球面浅水方程式
                            (GPU/CPU両実装)、潮汐・天体、レイマーチ描画
  90-app.js               モード登録・サイドバーUI・入力(IIFE終了)
legacy/                   統合前の単体バージョン(開発履歴)
```

注意: `00-shared.js` の先頭と `90-app.js` の末尾がIIFEの開き/閉じを持つ(全ファイルは順結合されて1つのスクリプトになる)。

## 検証

`node --check` はGLSLを検出できない。シェーダ変更時はヘッドレスChromium(Playwrightキャッシュの `chrome --headless=new --enable-unsafe-swiftshader`)で全モード生成+数フレーム実行のスモークテストを通すこと(GLSL予約語 `patch` で全滅した事故あり)。

## 公開

- GitHub Pages: https://kosei-matsuzaki.github.io/fluid-lab/
- Artifact: https://claude.ai/code/artifact/d3e7d264-c467-42cd-9a98-2e2d9c2df422

GitHub Pages は `main` ブランチのルートを配信する。`index.html` は作品紹介のランディングページ(OGP・各モードの実写つき)で、そこから `fluid-lab.html` に飛ばす。`.nojekyll` でJekyll処理を無効化している。

`node build.js` して `fluid-lab.html` をコミット・pushすれば数十秒でPagesに反映される。

### 紹介ページの画像
`assets/*.webp` は実際の動作画面をヘッドレスChromiumで撮ったもの。撮り直す場合は puppeteer-core で各モードを起動し、UI(サイドバー・ヒント・浮動ボタン)を隠して数秒進めてからスクリーンショットを撮る。インクは `page.mouse` で実際にドラッグさせないと何も描かれない。

### 断片HTMLであることの注意
`fluid-lab.html` はArtifactのラッパー前提の断片(`<!DOCTYPE>`/`<html>`/`<head>`/`<body>` を持たない)。そのままPagesで配信すると:

- `<meta charset>` と `<meta name="viewport">` は自前で持つ必要がある → `src/template.html` の先頭に置いてある。**消すとスマホでレイアウト幅980px・表示倍率0.4になり、キャンバスが6倍の画素数になって実質使えなくなる**
- doctypeがないため互換モード(quirks)で描画される。現状のCSSは標準/互換どちらでも同じ見た目になることを確認済み。doctypeを足すとArtifact側でラッパーと二重になるため足していない
