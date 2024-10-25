# transform _coordinates
横視点のトラッキング結果をコート座標に変換、コート座標を上視点のビデオ座標に変換する

### ディレクトリ構成
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

### 使い方
`python transform_coordinates`
デフォルトでは上視点のビデオ座標のみを保存するようになっているが、コメントアウトを解除することでコート座標の保存、コート座標でのプロット動画の保存、ビデオ上にBounding Boxを表示した動画の保存ができる。