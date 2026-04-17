# 日本語 -> 英語 用語対応表

この表は [main.tex](/home/kitamura/work/EpisodicCollusion-demand_scale_varying/PAAMSreport_Kitamura/main.tex) の英語化で使う訳語の基準である。特に断らない限り、`Preferred English` を優先する。

| 日本語 | Preferred English | 備考 |
|---|---|---|
| アルゴリズム共謀 | algorithmic collusion | 法政策文脈では `algorithm-facilitated collusion` もあるが、本稿はこれで統一 |
| 暗黙的共謀 | tacit collusion | `implicit collusion` より先行研究に合わせやすい |
| 共謀的結果 | collusive outcome | |
| 共謀的価格形成 | collusive pricing | |
| 共謀水準 | level of collusion | `collusion level` でも可。本文ではどちらかに固定 |
| 共謀維持 | sustain collusion | |
| 競争価格 | competitive price | Friedrich の文脈では `competitive (Nash) equilibrium price` が明確 |
| 共謀価格 | collusive price | 初出では `monopolistic optimum price` を併記するとよい |
| 競争基準 | competitive benchmark | |
| 共謀基準 | collusive benchmark | |
| ナッシュ均衡 | Nash equilibrium | |
| ナッシュ均衡価格 | Nash equilibrium price | |
| 独占最適 | monopolistic optimum | Friedrich に合わせる |
| 共同利潤最大化 | joint profit maximization | |
| 逸脱 | deviation | |
| 報酬・懲罰構造 | reward-punishment scheme | |
| 価格戦略 | pricing strategy | |
| 価格系列 | price path | `price sequence` でもよいが `price path` の方が論文らしい |
| エピソード | episode | |
| エピソディック市場 | episodic market | |
| 有限期間市場 | finite-horizon market | |
| 在庫制約市場 | inventory-constrained market | タイトルでもこの形を使う |
| 在庫制約 | inventory constraint | |
| 在庫可用性 | inventory availability | |
| 在庫残量 | remaining inventory | |
| 販売量 | sales quantity | `quantity sold` でもよい |
| 需要量 | demand quantity | |
| 需要シェア | demand share | |
| 需要モデル | demand model | |
| 多項ロジット | multinomial logit | 初出で `(MNL)` を付ける |
| outside option | outside option | 和訳せずそのままでよい |
| 水平差別化 | horizontal differentiation | |
| 垂直差別化 | vertical differentiation | |
| 市場規模 | market size | 一般概念として使うとき |
| 需要のスケール | demand scale | $\lambda$ に対応する説明で使いやすい |
| 需要規模係数 | demand-scale factor | $\lambda$ や $\lambda_t$ の正式名称として推奨 |
| 市場規模係数 | demand-scale factor | `market-size factor` よりこちらを優先 |
| 時間変動需要 | time-varying demand | |
| 固定需要 | fixed demand | |
| 指数増加需要 | exponentially increasing demand | |
| 指数トレンド需要 | exponential demand trend | |
| 需要の時間構造 | temporal structure of demand | 強い主張を避けたいときは `timing structure of demand` も可 |
| 需要の時間変動 | temporal variation in demand | |
| 市場規模スケジュール | demand schedule | $\{\lambda_t\}$ の説明として使える |
| 学習推移 | learning dynamics | 図キャプションでは `training dynamics` でも可 |
| 学習曲線 | learning curve | |
| 学習済み方策 | learned policy | |
| 評価エピソード | evaluation episode | |
| 強制逸脱 | forced deviation | |
| 価格応答 | price response | |
| 共謀指数 | collusion index | 本稿の主要評価指標 |
| エピソード共謀指数 | episode-level collusion index | |
| 一般化平均 | generalized mean | |
| エピソード利得 | episode-level profit gain | 元式の `episodic profit gain` よりやや自然 |
| 実験設定 | experimental setup | セクション名は `Experimental Setup` か `Experimental Design` |
| 実験設計 | experimental design | 条件設計の節ではこちらを優先 |
| 実験結果 | results | |
| 行動分析 | behavioral analysis | `behavior analysis` より自然 |
| 先行研究 | prior work | |
| 関連研究 | related work | 節名として使用 |
| 貢献 | contribution | 複数なら `contributions` |
| 予約市場 | reservation market | airline / hotel booking 文脈に合う |
| 予約曲線 | booking curve | ABCDEF law の説明に使う |
| 売り切れ | sold out | |
| 売り切れ企業 | sold-out seller | |
| 需要到着 | demand arrival | |
| 価格グリッド | price grid | |
| 離散価格 | discrete price | |
| 行動空間 | action space | |
| 状態空間 | state space | |
| 断りがない限り | unless otherwise stated | |
| 傾向が確認された | we observe that / the results show that | 英訳時に受動態を避けやすい |
| 可能性を示唆する | suggests that / may indicate that | 査読向けに弱めの表現 |

## 固定したい表現

- `市場規模` は一般概念なら `market size`、$\lambda_t$ の説明なら `demand-scale factor` を使う。
- `共謀価格` は単独で置かず、初出では `collusive price (monopolistic optimum)` としておく。
- `固定需要` と `変動需要` は図表中でも `fixed-demand` と `time-varying-demand` で揃える。
- `共謀指数` は `CI` と略すなら、本文初出後に限定する。

## 避けたい直訳

- `market scale` は使わない。
- `collusion degree` は使わない。
- `price competition market` は使わず、`pricing game`, `pricing market`, `price competition setting` を使う。
- `the demand is increasing exponentially in time` のような冗長表現は避け、`demand follows an exponential schedule` を優先する。
