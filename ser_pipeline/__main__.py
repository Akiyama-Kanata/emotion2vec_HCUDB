"""`python -m ser_pipeline` をコマンドライン処理へ接続する入口。"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
