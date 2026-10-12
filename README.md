# RoboSynChallenge 予選：進捗まとめ（チーム maru29）

最終更新：**2026-10-12 19:31 JST** ／ 締め切り：**2026-10-12 20:59 JST**

> **こちらから再提出はしていません。** チームメイトからの連絡では、チームの混合モデル構成が18:45に提出済みです。最後の有効な提出だけが評価されるため、こちらの全ACT構成でチームの提出を上書きしません。
>
> **共有したい変更はDrawerだけです。** 公式ACT＋グリッパー単位変換で、チームと同じ内容の `random` 設定ファイルを使い、**18/20成功**を確認しました。他9タスクを保持する最小変更を、チーム側で確認・採用判断してもらうための資料です。

## 1. Drawerの実測結果

| 評価 | 成功 | 平均動作ステップ | 備考 |
|---|---:|---:|---|
| 修正した公式ACT、random 20回 | **18/20（90%）** | **422.40 / 900** | job 31877。比較用パレット固定・姿勢再適用なし |
| 修正した公式ACT、別seed系列のclear 20回 | 5/20（25%） | 743.35 / 900 | job 31862。上記とは条件・seed群が異なる |
| 修正した公式ACT、clearの初期動作確認 | 2/3 | 545.67 / 900 | 実測関節値と指令値も確認 |
| 自作Drawer ACT、単位修正後のclear確認 | 3/3 | 349.00 / 900 | 少数回なので、これだけで公開版より良いとは判断しない |

採用を提案する重みは**新規モデルではなく、公式公開ACT**です。

- HF：<https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place/tree/592e30434aad83cf7bd8f1ee105d7c401488743d>
- 予測長50、実行長50、元の正規化統計を使用。
- GPUはNVIDIA H200、PyTorch 2.7.1+cu128、LeRobot 0.3.3（`b883328e6c95681ca90a18b102e4ae5e1f91e2bf`）。
- 評価コードはRoboSynChallenge `ecbf88ae26bbd8e369eba868e23354c8f64e083f`、DexSim 0.4.3。
- 20/24の候補seedが公式のexpert feasibility checkを通過。成功率は受理された20エピソードに対する値。

### 物理設定について

こちらで追加の剛性・減衰・把持力変更を行った結果ではありません。評価に用いた `configs/drawer_open_place/random/gym_config.json` と、確認時点のチームmainの同ファイルをJSONとして比較し、**全内容が一致**しました。

「公式の設定ファイルに手を加えていない」という意味であり、すべての歴史的な評価設定・非公開本番設定と同一という意味ではありません。公式のACT 29%に付いている `drawer_drive050_grip20` の注記とは区別してください。本番でも90%になると保証するものではありません。

SYNKA固有の対処は割当GPUのCUDA/Vulkan/NVML整合と、内容が同一のURDFを短いパスで読み込む処理です。パレット固定・物体姿勢再適用を使った別の比較実験もありますが、**上の18/20はそれらを使わない評価**です。

## 2. 修正の内容とチームへの受け渡し

Drawerのcheckpointが出すグリッパー成分6・13は0〜1の開度です。環境の物理関節範囲は0〜0.05 mで、**0が閉、0.05が開**です。モデル出力の平均・標準偏差を戻した後、環境へ渡す直前に変換します。

| モデル出力 | 正しい物理指令 |
|---:|---:|
| 0 | 0 m |
| 0.5 | 0.025 m |
| 1 | 0.05 m |

元の処理では0.5をそのまま渡し、環境側が上限0.05へ切り詰めていました。中間の開度や閉じる指令の小さな誤差が、大きな物理値になってしまいます。

公開できる最小コード差分：[drawer_adapter_only.patch](patches/drawer_adapter_only.patch)。公式の公開ACTアダプタを元に作成し、Drawerの場合にだけ変換する条件分岐です。チームmain向けの**対応表変更も含む2ファイル版**は、別途添付した `drawer_only.patch` を使用してください。

チーム側で必要なのは次の3点です。

1. Drawerにだけ単位変換を適用。**他タスクへ一律に適用しない**。
2. `submission/checkpoints.json` のDrawerを上記の公式ACT・固定revisionへ変更。他9タスクを維持。
3. `bash eval.sh drawer_open_place random 0 --headless True` で確認し、採用する場合はフォームのDrawerのHF欄と提出メモも整合させる。

共有ファイルに共通アダプタの変更があっても、変換は `task_name == "drawer_open_place"` の場合だけです。CPUテストではDrawerの0/0.5/1変換、腕の関節値・モデル内の元tensor保持、ClickBell・Water・Handleの指令値不変を確認しました。これはチーム環境でのGPU動作確認を代替するものではありません。

## 3. ClickBell

**新しいClickBellモデルは作成していません。** チームが現在使っているFT版は、昨日公開した250ステップ選択のモデルです。

<https://huggingface.co/yukib0123/robosyn-act-clickbell-20261012/tree/0ba62e98e11840703168facd7237d9640313295c>

我々のclear評価は公式ACT18/20、FT版17/20です。これだけでFT版の優位性は確認できません。ClickBellのDP追加評価は見送り、こちらの独立候補では公式ACTを維持しました。チームの提出済みFT版を勝手に変更していません。

## 4. その他の単位修正後の確認

以下は我々のH200環境での `clear` 結果です。少数回の確認と、別条件の公式リーダーボードを直接比較しないでください。

| タスク | 公式ACT | 我々の自作ACT |
|---|---:|---:|
| Water pouring | 1/3 | 1/3 |
| Handle basket | 0/3 | 0/3 |
| Items handover | 0/3 | 0/3 |
| Mixer operating | 3/3 | 3/3 |
| Item assembly | 0/3 | 0/3 |
| Manipulate pipette | 7/10 | 7/10 |
| Sample loading | 0/3 | 0/3 |

Tableはもともと物理単位のactionなので変換不要です。以前の同seedのclear20比較は公式20/20、自作20/20で、我々の独立候補では平均ステップが少し短い自作版を選びました。

Handoverの3失敗動画では、安定したペン把持・持ち上げが成立しないまま受け渡し動作へ進む場面が共通しています。途中で再推論しても進行するため、古いactionキューだけが原因とは判断していません。追加学習・全面的な実行長探索は行っていません。

## 5. 公開成果物と確認済み事項

- [我々の独立した提出候補・main](https://github.com/yukib0123/robosyn-act-unitfixed-20261012/tree/main)：**未提出。チーム提出の置き換えには使わない**。Tableのみ自作、他9タスクは公式ACT、実行長50。
- [全10タスクの自作FT候補](https://huggingface.co/yukib0123/robosyn-act-alltasks-20261012/tree/4dbacab83136d3b6e277313368547ea6d28f3d31)：各タスクのconfig・重み・学習履歴・データ分割を保存。すべてを提出に採用しているわけではありません。
- 最終候補10checkpointの匿名アクセス、固定revision、重みのSHA-256を確認。
- 提出用ローダーで全10タスクをGPUでstrict reloadし、実際の記録画像から有限の `[1,50,14]` 予測と物理単位actionを確認（31888）。
- 提出用 `eval.sh` でClickBellとTableを各1回random評価し、両方正常終了・成功（31889、31890）。1回の結果は成功率推定として使わない。

詳細な評価値・受理seed・保存先：[final_selection_checks.json](verification/final_selection_checks.json)。全10タスクの読み込み結果：[selected_checkpoints.json](verification/selected_checkpoints.json)。

## 6. 最終判断の分担

こちらの役割は、Drawerの実測結果・固定checkpoint・単位変換コードを共有することです。チームメイトが、既存の自作Handle basketとSmolVLA構成を保持したまま、Drawerの動作確認・採用判断・提出内容の更新を行う想定です。**こちらから再提出・チームmainへの変更・チームメイトへの自動送信は行っていません。**
