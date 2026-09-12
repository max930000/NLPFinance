# 財經新聞情緒分析 × 股價預測

用 NLP 情緒分析結合股價資料，檢驗財經新聞的情緒是否對股價有預測力 —— 以台積電(2330)為研究對象，
比較三種情緒分類方法的準確率，分析情緒分數與股價報酬率的相關性，並檢驗「加入情緒特徵」能否提升次日漲跌方向的預測準確率。

這個專案的重點不是產出一個「準確的預測模型」，而是誠實地走一遍完整的研究流程 —— 包含清楚呈現**沒有效果**的部分。

## 資料來源

| 資料 | 來源 | 說明 |
|---|---|---|
| 新聞標題 | [FinMind](https://finmindtrade.com/) `taiwan_stock_news` | 2024-01-01 ~ 2025-09-01，台積電(2330)相關新聞，約 14,600 則(去重後) |
| 股價資料 | [yfinance](https://github.com/ranaroussi/yfinance) | 2330.TW 每日 OHLCV |
| 情緒標註測試集 | 人工標註 | 從新聞中隨機抽樣 150 則，人工標記正面(1)/中立(0)/負面(-1) |

## 方法論

```
新聞標題 ──┬─ 詞典法              ─┐
           ├─ 現成財經 BERT       ─┼─→ 與 150 則人工標註比較準確率
           └─ Fine-tuned BERT     ─┘         │
                                              ▼ (選用準確率最高的方法)
                                    為全部新聞打情緒分數
                                              │
                                              ▼
                              依日期聚合 → 每日平均情緒分數
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                                           ▼
          與股價報酬率做相關性分析                      加入價格特徵，預測次日漲跌方向
        (當日 vs 領先一日，檢驗是否為領先指標)      (Logistic Regression / Random Forest,
                                                    比較有無情緒特徵、配對 t 檢定顯著性)
```

1. **情緒分類方法比較**：詞典法 vs 現成財經領域 BERT vs 自己 fine-tune 的 BERT，用 150 則人工標註新聞當測試集
2. **相關性分析**：情緒分數 vs 股價報酬率(當日、領先一日)
3. **預測模型比較**：Logistic Regression / Random Forest，比較「有無情緒特徵」對次日漲跌方向預測準確率的影響，並做配對 t 檢定

## 關鍵發現

### 1. 情緒分類方法：fine-tuned BERT 準確率最高，但都不算「準」

在 150 則人工標註測試集上：

| 方法 | 準確率 |
|---|---|
| 詞典法(關鍵字規則) | **27%** |
| 現成財經 BERT(`sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction`) | **55%** |
| Fine-tuned BERT(`hfl/chinese-macbert-base` 在 ChnSentiCorp 微調) | **67%** |

詞典法準確率極低，主要是大多數新聞標題不含預先定義的關鍵字而被誤判為中立，暴露規則式方法在覆蓋率上的根本限制。
兩個 BERT 模型都明顯優於詞典法，其中在通用情感資料集(而非財經新聞)上微調的模型，表現反而超越專門宣稱做財經情緒分析的現成模型，
但兩者在「中立」類別上的 F1 都很低(現成 BERT 僅 0.05，fine-tuned BERT 0.13) —— 三分類本身對這個任務來說可能太細，模型很難穩定分辨「中立」與「弱正/負面」。

### 2. 情緒與股價：同步反映，而非領先指標

- 當日情緒分數 vs **當日**報酬率：r = 0.574，p < 0.001(顯著相關)
- 當日情緒分數 vs **次日**報酬率(領先一日)：r = -0.023，p = 0.648(不顯著)

當日相關性看似很強，但領先一日後幾乎完全消失。這個結果的合理解釋是：新聞標題傾向描述「已經發生」的價格變化(例如「大漲」「重挫」「創新高」這類字眼本身就是報酬率的產物)，
情緒分數其實是報酬率的**同步甚至落後**指標，而不是能提前預測隔天走勢的領先訊號。

### 3. 情緒特徵無法穩定提升次日漲跌預測

用「過去價格特徵」(無情緒)與「過去價格特徵 + 情緒特徵」(有情緒)分別訓練模型，用 5 折 TimeSeriesSplit 交叉驗證比較，並做配對 t 檢定：

| 模型 | 無情緒(平均準確率) | 有情緒(平均準確率) | 配對 t 檢定 |
|---|---|---|---|
| Logistic Regression | 54.4% | 54.7% | t = 0.23, **p = 0.83**(不顯著) |
| Random Forest | 50.6% | 49.7% | t = -0.43, **p = 0.69**(不顯著) |

- 兩個模型加入情緒特徵後，平均準確率都只有極小幅變化(LR 微升、RF 微降)，配對 t 檢定都不顯著 —— 沒有證據支持情緒特徵能穩定改善次日漲跌預測。
- 在單次 80/20 時間切分的測試集上，兩個簡單 baseline —— 多數類別(49.4%)、動量(昨漲今猜漲，55.8%) —— 甚至優於兩個 Logistic Regression 模型(50.7% / 46.8%)，說明這個任務本身的訊噪比很低。
- Random Forest 特徵重要性排序顯示 `sentiment_today`、`sentiment_lag1` 個別重要性其實不低(僅次於 `ma5`)，但這與「加入情緒特徵並沒有讓整體準確率變好」並不衝突 —— 重要性反映的是模型用了這個特徵做切分，不代表切分對的機率變高，這正是小樣本、高相關特徵下容易出現的現象。
- **一個值得注意的重現性細節**：股價資料重新從 yfinance 抓取時，少數幾天的收盤價與先前抓取的版本有極小差異(yfinance 的歷史資料會隨除權息等事件回溯調整)，導致極少數交易日的漲跌標籤翻轉。這個變動小到幾乎不影響 baseline 與 LR 的單次切分結果，但足以讓 Random Forest 交叉驗證的顯著性大幅改變 —— 這本身就說明了小樣本、單一股票研究的結論有多脆弱，也是為什麼不該只看單一次重跑的 p 值下結論。

### 這些「無顯著發現」本身的意義

這個專案沒有做出一個能穩定打敗大盤的訊號，這件事本身就是一個有意義的結果，而不是失敗：它與**效率市場假說**一致 —— 台積電是被高度關注、流動性極高的股票，公開的新聞情緒資訊會被市場價格快速消化，很難單靠公開文字訊息找到未被定價的預測優勢。若情緒特徵真的能輕易、穩定地預測次日漲跌，反而才是值得懷疑資料洩漏或方法論問題的訊號。

## 如何重現實驗

```bash
git clone <your-repo-url>
cd side
python -m venv .venv
.venv\Scripts\activate   # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
```

1. **取得資料**(`data/` 已被 `.gitignore` 排除，需自行產生)：
   - 新聞：呼叫 `src/data_collection.py` 的 `collect_news("2330", "2024-01-01", "2025-09-01")`(需要 FinMind，免費額度 300 次/小時)
   - 股價：`src/data_collection.py` 的 `fetch_price_data("2330.TW", start, end")`(yfinance，免安裝金鑰)
   - 或者直接抽樣 150 則新聞、人工標註，存成 `data/to_label.csv`(欄位：`date`, `title`, `label`)
2. **取得 fine-tuned BERT 模型**(`models/my_finetuned_bert/` 已被 `.gitignore` 排除，檔案約 400MB)：
   - 按 `src/train_bert.py`(或直接執行 [`notebooks/train_bert_colab.ipynb`](notebooks/train_bert_colab.ipynb))在 Google Colab(免費 T4 GPU)重新微調，再把輸出資料夾放到 `models/my_finetuned_bert/`
3. **執行分析**：開啟 [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)，由上到下執行即可重現所有結果(圖表與比較表輸出到 `results/`)
4. **(選用)啟動 Streamlit Dashboard**，即時比較三種方法對單一標題的判斷：
   ```bash
   streamlit run app.py
   ```

## 專案限制與未來工作

- **樣本量小**：人工標註測試集僅 150 則，預測模型的測試集只有 384 個交易日中切出的 77 天，配對 t 檢定的統計檢定力有限，任何「顯著」或「不顯著」的結論都需要更大樣本驗證。
- **單一股票**：只研究台積電一檔股票，結論(尤其是「情緒無領先預測力」)不能直接推論到其他股票或整體市場，換一檔關注度較低、流動性較差的股票，結果可能不同。
- **標籤雜訊**：人工標註只由單人完成，沒有多人標註取一致性(inter-annotator agreement)，「正面/中立/負面」的邊界主觀性高；情緒分數本身也有自相關(Ljung-Box 檢定 p = 0.013)，代表同一波新聞熱度會連續影響多天的分數，跨天樣本並非完全獨立，可能讓交叉驗證的顯著性檢定過於樂觀。
- **Fine-tuned BERT 用的是通用情感語料，不是財經語料**：ChnSentiCorp 主要是商品/服務評論，不是財經新聞，準確率能超越專門的財經情緒模型，某種程度上也反映現成財經模型本身品質有限，而非通用情感遷移特別有效。
- **未來可以嘗試**：擴大人工標註量並找多人標註取一致性、改用財經領域語料重新微調、將情緒特徵的時間窗口拉長(例如週線)、擴大到多檔股票或整體大盤指數以檢驗結論的普遍性、嘗試將情緒分數與其他另類資料(如成交量、法人買賣超)組合。

## 技術棧

- **語言**：Python
- **NLP / 深度學習**：Transformers (BERT)、PyTorch、jieba(斷詞)
- **機器學習**：scikit-learn(Logistic Regression、Random Forest、TimeSeriesSplit)
- **統計檢定**：SciPy(pearsonr、配對 t 檢定)、statsmodels(Ljung-Box 自相關檢定)
- **資料處理**：pandas、numpy
- **資料來源**：FinMind、yfinance
- **視覺化 / Dashboard**：Streamlit、Matplotlib

## 專案結構

```
side/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # 原始/處理過的資料(不上傳，見 .gitignore)
├── models/
│   └── my_finetuned_bert/   # fine-tuned BERT 權重(不上傳，見 .gitignore)
├── src/
│   ├── data_collection.py   # 抓新聞(FinMind)、抓股價(yfinance)
│   ├── sentiment_models.py  # 詞典法、兩個 BERT 模型的推論函式
│   ├── train_bert.py        # BERT fine-tune 訓練程式碼(Colab 版本)
│   ├── feature_engineering.py  # 合併資料、造特徵、造 target
│   └── prediction_models.py    # Logistic Regression、Random Forest、t 檢定
├── notebooks/
│   ├── analysis.ipynb           # 完整分析 notebook，可從頭執行到尾
│   └── train_bert_colab.ipynb   # 實際在 Colab 執行 BERT fine-tune 的原始 notebook(含真實輸出)
├── results/                 # 圖表、比較表格等輸出結果
└── app.py                   # Streamlit dashboard
```
