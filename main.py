from graphdraw.btc_usd_visuals_basic import full_btc_trendline
from calc.statistic_params import calc_statistic_params
from model.sklearn_model import train_model

full_csv_file = './data/BTC-USD_full_copy.csv'

if __name__ == "__main__":
    # full_btc_trendline(full_csv_file)
    df = calc_statistic_params(full_csv_file)
    model = train_model(df)
    print("Well done! :)")
