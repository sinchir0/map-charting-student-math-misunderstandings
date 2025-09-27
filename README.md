# map-charting-student-math-misunderstandings

## VSCodeの拡張機能をインストール
```
python
jupyter
ruff(任意)
GitHub Copilot(任意)
```

## 環境構築
```
uv sync & uv sync --extra flash-attn
```

## バックグラウンドにて、複数のノートブックを直列に実行する
```
# あらかじめ、mutiple_run.sh内に実行したいnotebookを記載する
nohup ./shell/multiple_run.sh &
```

## 動いているかの確認
```
ps aux | grep python
```

## Kill
```
pkill multiple_run && pkill runnb
```

## git系

```
git add -u
git config --global user.email "dekunattou@gmail.com"
git commit -m "add"
git push origin main
```

## コンペティションデータのダウンロード

### セッティング
- kaggle.jsonをダウンロードする
  - https://www.kaggle.com/settings
- kaggle.jsonを`env_file`配下にアップロードする
- 適切なパスと権限を付与し、kaggleをinstallする
```
./shell/set_kaggle_api.sh
```

### データのダウンロード
```
./shell/download_competition_data.sh
```

## シェルの設定

```
./shell/setting_shell.sh
```

## GitHubへのアクセス権限の設定
```
./shell/make_github_key_and_set_email.sh
```

以下のリンクで、New SSH Keyを行い、出力された公開鍵を登録する

https://github.com/settings/keys


## GitHub上でのリポジトリの作成
まずは、GitHub上でmap-charting-student-math-misunderstandingsという名前のrepoを作成する

次に、ローカルのファイルをpushする。
```
./shell/git_first_push.sh
```

## 環境変数の設定
```
cp env_file/.env.default env_file/.env
```
envファイルに対し、WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEYを設定する

## 並列分散学習
```bash
uv run accelerate launch --config_file config/accelerate_config_zero3.yaml \
    exp/exp007/train.py
```

## Kaggle からのダウンロード
```bash
kaggle datasets download sinchir0/exp018-20250924144032 -p outputs/exp018
unzip outputs/exp018/exp018-20250924144032.zip -d outputs/exp018
rm -rf outputs/exp018/exp018-20250924144032.zip
```

## 注意点

### 仮想環境
- レポジトリはrequirements.txtで管理する。ryeの方が高速だが、最近のLLMのレポジトリがpipでの方法のみ紹介することも多く、対応に時間を割きたくないため。

### nbconvertで変換した際に、ログに残るかどうか
- printで出力したものは残る
- notebookの一番最後に実行し、Notebookの機能で出力したものは残らない
