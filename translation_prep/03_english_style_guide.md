# 国際会議向け英文スタイル指針

この文書は [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex) を LNCS 系国際会議向けに英文化する際の、文単位の最小限のスタイル基準である。

## 文章レベル

- 1 文 1 主張を基本とする。
- 日本語の長い因果連鎖は、英語では 2 文以上に分ける。
- 段落冒頭で結論を先に書き、その後で根拠を書く。
- `This suggests that ...` の前に、何が観測されたかを明示する。
- abstract は 150--250 words を目安に圧縮する。

## 時制

- 一般的事実、論文構成、定義: 現在形
- 実験で観測したこと: 過去形または現在形のどちらでもよいが、論文内で統一
- 図表が示す内容: 現在形

## 主語と態

- 不要な無主語受動態を避ける。
- `We define`, `We model`, `We compare`, `We observe` を基本にする。
- ただし強い断定は避け、解釈では `may`, `might`, `can` を使う。

## 査読向けの弱め方

- `proves` は使わず `shows`, `demonstrates`, `suggests` を使う。
- `because` を乱用せず、解釈では `which may be due to`, `which is consistent with` を使う。
- `important role` のような強い表現は、必要なら `potentially important role` に弱める。

## 数式・図表まわり

- 数式の前に直感、数式の後に役割を書く。
- 記号の説明と主張を同じ一文に詰め込まない。
- 図の説明では `Figure X shows ...` から始める。
- 数値比較を入れるなら、固定条件と変動条件の差を明示する。
- 表キャプションは文として書き、表の上に置く。
- 図キャプションは文として書き、図の下に置く。

## 用語の一貫性

- 訳語は [01_terminology_ja_en.md](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/translation_prep/01_terminology_ja_en.md) を正本とする。
- `market size`, `demand scale`, `demand-scale factor` を混ぜない。
- `competitive`, `collusive`, `monopolistic` の対応関係を初出で固定する。

## 図表キャプション

- 完全文で書く。
- 略語を使うときは本文初出後に限る。
- 左右比較なら `Left: ...; right: ...` を基本にする。
- `Comparison of ...` のような名詞句だけで終えない。

## 見出しと段落

- 番号付き見出しは section と subsection までを前提にする。
- subsubsection は run-in heading 的に短く使うか、必要性が薄ければ増やさない。
- 節や小節の最初の段落では、英語化の際に無理に接続文を足さない。

## 今の原稿で英訳前に意識すべき点

- [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex:381) 以降に `Fixed demand`, `Collusion Index`, `collusion index` が混在しているので、英訳時に表記を統一する。
- [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex:473) からの PPO/DQN 解釈は、観測事実と仮説説明を分けて書く。
- [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex:69) の abstract は 1 段落 5 文程度に分割し直す前提で英語化する。
- [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex:160) 以降の図キャプションは、英訳時も説明的な完全文のまま維持する。
