# transform _coordinates
横視点のトラッキング結果をコート座標に変換、コート座標を上視点のビデオ座標に変換する

### ディレクトリ構成
```
transform_coordinates
├── assets
│   ├── homography_matrix
│   ├── tracking
│   │   ├── 1
│   │   ├── ...
│   │   └── 6
│   └── video
│       ├── 1
│       ├── ...
│       └── 6
└── result
    ├── coordinates
    │   ├── 1
    │   ├── ...
    │   └── 6
    └── video
        ├── 1
        ├── ...
        └── 6
```

### 使い方
`python transform_coordinates`

* コート座標の保存
* 上視点のビデオ座標の保存 (デフォルト)
* コート座標でのプロット動画の保存
* 上視点のビデオ上にBounding Boxを表示した動画の保存

[`transform_coordinates.py`](https://github.com/shunsuke-iwashita/transform_coordinates/blob/main/transform_coordinates.py)中のコメントアウトで調整できる
