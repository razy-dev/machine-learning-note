# class DatasetBuilder:
#     data: DataFrame
#
#     def __init__(self, ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None, features: int = 1):
#         self.data = self.load(ticker, start, end, size, features)
#         pass
#
#     def build(self, time_steps: int, output_size: int, normalizer: Normalizer = None, train_rate: float = 0.7):
#         pass
#
#     @classmethod
#     def load(cls, ticker: str = 'AMZN', start: str = '2013-01-01', end: str = '2021-12-31', size: int = None, features: int = 1) -> DataFrame:
#         try:
#             size = size or 1
#             features = features or 1
#             df = yf.download(ticker, start=start, end=end, progress=False)[
#                 ['Adj Close', 'Open', 'High', 'Low', "Close", "Volume"]
#             ]
#             return df[:min(len(df), size), :min(features, len(df.columns))]
#         except Exception as e:
#             print(e)
#             return DataFrame([])


# for test
if __name__ == "__main__":
    # DatasetBuilder(features=6)
    # train_dataset, test_dataset, scaler = DatasetBuilder(100, 4).build(train_rate=0.7)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    pass
