# 英訳前の下準備フロー

この文書は [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex) を国際会議向けに英語化する前の最小限の作業順を固定するためのメモである。文章構成を大きく変えず、用語と英文表現をそろえることを主眼とする。

## 基本方針

- 大きな段落再編や論旨の組み替えは行わない。
- 先行研究と用語を合わせつつ、本稿の記法を保つ。
- 英文化でぶれやすい語と表現だけ先に固定する。
- 強い因果断定だけは避ける。
- LLNCS テンプレートの体裁ルールに反しない範囲で英語化する。

## 実施フロー

1. 用語と記号の英訳を固定する。
2. 図表・数式・評価指標の英語表現を固定する。
3. 強すぎる断定表現だけ洗う。
4. ここまで終わってから、タイトル・要旨・本文の順で英語化する。

## 各ステップの具体作業

### 1. 用語と記号の英訳を固定する

- 用語対応は [01_terminology_ja_en.md](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/translation_prep/01_terminology_ja_en.md) を正本とする。
- 特に `市場規模`, `需要規模係数`, `共謀指数`, `競争価格`, `共謀価格`, `逸脱`, `固定需要`, `変動需要` は訳語がぶれやすいので、本文全体で固定する。
- `market size` と `demand scale` の使い分けは要注意で、数式中の $\lambda_t$ には `demand-scale factor` を優先する。

### 2. 図表・数式・評価指標の説明方針を固定する

- 図表キャプションは「何を比較しているか」を一文で言い切る。
- 数式の直後では、記号の説明とその式の役割を分ける。
- `collusion index` は定義、集約方法、解釈範囲を毎回混ぜずに順番に説明する。
- 競争基準と共謀基準は「price level」ではなく「price path / pricing strategy」に立脚した定義である点を強調する。
- 表キャプションは表の上、図キャプションは図の下に置く。
- 線画や模式図は、可能ならラスタ画像よりベクター画像を優先する。

### 3. 査読で突かれやすい主張を調整する

- `because` で強く因果を断定している箇所は、必要に応じて `may`, `suggest`, `is consistent with` に弱める。
- PPO と DQN の違いの説明は、観測結果からの解釈であることを明示する。
- `Nash equilibrium price does not need to be recomputed` は、近似条件と離散化の但し書きを保ったまま説明する。

## LLNCS 由来で今回維持するルール

- abstract は 150--250 words を目安にする。
- 番号付き見出しは第 2 レベルまでに留める。
- 見出しレベルは最大 4 段階までにする。
- 節や小節の最初の段落、および図表・数式直後の最初の段落は字下げしない。
- 独立表示の数式は中央配置・別行とする。
- 参考文献は `splncs04` を使い、引用は角括弧付きの通し番号スタイルに合わせる。
- Web 参照を使う場合は必要に応じて `last accessed` を入れる。

## 今回作成した成果物

- 用語集: [01_terminology_ja_en.md](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/translation_prep/01_terminology_ja_en.md)
- 英文スタイル指針: [03_english_style_guide.md](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/translation_prep/03_english_style_guide.md)

## 英語化の実行順

1. タイトル、running title、abstract、keywords
2. Introduction
3. Related Work
4. Problem Statement
5. Experimental Design
6. Results
7. Conclusion
8. 図表キャプションと細部の表現統一
