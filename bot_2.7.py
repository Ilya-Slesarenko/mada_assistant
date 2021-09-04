import config, logging, random, warnings, io, base64
from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.exceptions import BadRequest as BRq
from scipy.stats import norm
from openapi_client import openapi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader.data as pdr
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')
# if  JSONDecodeError  HAPPENS - upgrade the yfinance!

# method for image capturing in temp.memory (instead of saving)
try:
    from StringIO import StringIO  ## for Python 2
except ImportError:
    from io import StringIO

# Получение списка акций   - всех доступных!!!
my_token = 't.NSmiFBa_aCeegxA4gCrVVI2pW_qYoDAKxm3M2_a4xq_jokCa-baZxJc1mjgr7VmO1HHpQFY5XOyjqlWcRSHXuA'
client = openapi.api_client(my_token)
stocks = client.market.market_stocks_get()
stocks_list = stocks.payload.instruments
tickers_list = []  # список всех доступных тикеров на Тинькове
for i in stocks_list:
    tickers_list.append(i.ticker)


# stock_market_data
def get_tf_companies_feedback(ticker):
    try:
        t_info = yf.Ticker(ticker).info

        try:
            company_name = t_info.get('shortName')
        except TypeError:
            company_name = None
            print(f'passing zero to {ticker}, for Company')

        try:
            sector = t_info.get('sector')
        except TypeError:
            sector = None
            print(f'passing zero to {ticker}, for Sector')

        try:
            country = t_info.get('country')
        except TypeError:
            sector = None
            print(f'passing zero to {ticker}, for Country')

        try:
            m_cap = round(t_info.get('marketCap') / 1000000, 2)
        except TypeError:
            m_cap = None
            print(f'passing zero to {ticker}, for M_cap')

        try:
            enterp_val = round(t_info.get('enterpriseValue') / 1000000, 2)
        except TypeError:
            enterp_val = None
            print(f'passing zero to {ticker}, for Enterp_val')

        try:
            P_S_12_m = round(t_info.get('priceToSalesTrailing12Months'), 2)
        except TypeError:
            P_S_12_m = None
            print(f'passing zero to {ticker}, for P_S_12_m')

        try:
            P_B = round(t_info.get('priceToBook'), 2)
        except TypeError:
            P_B = None
            print(f'passing zero to {ticker}, for P_B')

        try:
            marg = round(t_info.get('profitMargins'), 3)
        except TypeError:
            marg = None

        try:
            enterprToRev = t_info.get('enterpriseToRevenue')
        except TypeError:
            enterprToRev = None
            print(f'passing zero to {ticker}, for EnterprToRev')

        try:
            enterprToEbitda = t_info.get('enterpriseToEbitda')
        except TypeError:
            enterprToEbitda = None
            print(f'passing zero to {ticker}, for EnterprToEbitda')

        try:
            yr_div = round(t_info.get('trailingAnnualDividendYield'), 3) if t_info.get('trailingAnnualDividendYield') is not None else 0
        except TypeError:
            yr_div = None
            print(f'passing zero to {ticker}, for yr_div')

        try:
            exDivDate = datetime.fromtimestamp(t_info.get('exDividendDate'))
        except TypeError:
            exDivDate = None
            print(f'passing zero to {ticker}, for exDivDate')

        try:
            five_yr_div_yield = t_info.get('fiveYearAvgDividendYield') if t_info.get('fiveYearAvgDividendYield') is not None else 0
        except TypeError:
            five_yr_div_yield = None
            print(f'passing zero to {ticker}, for 5_yr_div_yield')

        try:
            div_date = exDivDate.strftime('%d.%m.%y')
        except AttributeError:
            div_date = 'Без дивидендов'

        normalized_output = str(f'Полное наименование: {company_name}\nСектор: {sector}\nСтрана: {country}\nРыночная капитализация: ${m_cap}млн.\nСтоимость компании: ${enterp_val}млн.\n' +
                                f'P/S: {P_S_12_m}\nP/B: {P_B}\nМаржинальность: {round(marg * 100, 2)}%\nСтоимость компании / Выручка: {round(enterprToRev, 2)}%\n' +
                                f'Стоимость компании / EBITDA: {round(enterprToEbitda, 2)}%\nГодовая дивидендная доходность: ~{round(yr_div * 100, 2)}%\n' +
                                f'Див.доходность за 5 лет: {round(five_yr_div_yield * 100, 2)}%\n' +
                                f'Крайняя дата выплаты дивидендов: {div_date}')

    except IndexError:
        error_text_1 = f'IndexError issue for: {ticker}, passing'
        return error_text_1
    except ImportError:
        error_text_2 = f'ImportError issue (maybe because too mush parsing... for: {ticker}, passing'
        return error_text_2

    return normalized_output


def riskAnalysis(ticker):
    verdict_list_whole_period = []
    probabilities_to_drop_over_40 = []
    probabilities_to_drop_over_5 = []

    start = datetime.now() - timedelta(365 * 2.5)
    start_2 = datetime.now() - timedelta(365)
    end = datetime.utcnow()

    try:
        yf.pdr_override()
        data = pdr.get_data_yahoo(ticker, start, end)
        current_rate = data['Close'][-1]
        data_2 = pdr.get_data_yahoo(ticker, start_2, end)

        data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
        data_2['PriceDiff'] = data_2['Close'].shift(-1) - data_2['Close']
        data['Return'] = data['PriceDiff'] / data['Close']
        data_2['Return'] = data_2['PriceDiff'] / data_2['Close']

        # Create a new column direction.
        # The list cmprehension means: if the price difference is larger than 0, donate as 1, otherwise, doate 0
        data['Direction'] = [1 if data['PriceDiff'].loc[ei] > 0 else 0 for ei in data.index]
        data_2['Direction'] = [1 if data_2['PriceDiff'].loc[ei] > 0 else 0 for ei in data_2.index]

        # making mean average for 10 and 50 days
        data['ma50'] = data['Close'].rolling(50).mean()
        data_2['ma50'] = data_2['Close'].rolling(50).mean()
        data['ma10'] = data['Close'].rolling(10).mean()
        data_2['ma10'] = data_2['Close'].rolling(10).mean()
        data['ma5'] = data['Close'].rolling(5).mean()
        data_2['ma5'] = data_2['Close'].rolling(5).mean()

        data['Shares'] = [1 if data.loc[ei, 'ma10'] > data.loc[ei, 'ma50'] else 0 for ei in data.index]
        data_2['Shares'] = [1 if data_2.loc[ei, 'ma10'] > data_2.loc[ei, 'ma50'] else 0 for ei in data_2.index]
        # close price for tomorrow (forecase this way!)  # later - apply the predict with Tensorflow one!
        data['Close1'] = data['Close'].shift(-1)
        data_2['Close1'] = data_2['Close'].shift(-1)
        data['Profit'] = [data.loc[ei, 'Close1'] - data.loc[ei, 'Close'] if data.loc[ei, 'Shares'] == 1 else 0 for ei in data.index]
        data_2['Profit'] = [data_2.loc[ei, 'Close1'] - data_2.loc[ei, 'Close'] if data_2.loc[ei, 'Shares'] == 1 else 0 for ei in data_2.index]

        data['wealth'] = data['Profit'].cumsum()
        data_2['wealth'] = data_2['Profit'].cumsum()
        verdict_whole_period = round(data['wealth'][-2], 2)
        verdict_whole_period_2 = round(data_2['wealth'][-2], 2)
        print_text = f'{ticker}: {verdict_whole_period}%'

        verdict_list_whole_period.append(print_text)

        data['LogReturn'] = np.log(data['Close']).shift(-1) - np.log(data['Close'])
        data_2['LogReturn'] = np.log(data_2['Close']).shift(-1) - np.log(data_2['Close'])
        data['LogReturn'].hist(bins=50)
        data_2['LogReturn'].hist(bins=50)

        mu = data['LogReturn'].mean()  # approximate mean
        sigma = data['LogReturn'].std(ddof=1)  # variance of the log daily return

        # what is the chance of losing over _n_% in a day?
        mu220 = 220 * mu
        sigma220 = 220 ** 0.5 * sigma
        prob_to_drop_over_40 = norm.cdf(-0.4, mu220, sigma220)
        append_40_text = f'{ticker}: {prob_to_drop_over_40}'
        probabilities_to_drop_over_40.append(append_40_text)

        # what is the chance of losing over _n_% in a day?
        mu5 = 5 * mu
        sigma5 = 5 ** 0.5 * sigma
        prob_to_drop_over_5 = norm.cdf(-0.4, mu5, sigma5)
        append_5_text = f'{ticker}: {prob_to_drop_over_5}'
        probabilities_to_drop_over_5.append(append_5_text)

        buy_now_decision_1 = round(data['ma10'][-2] - data['ma50'][-2], 2)  # 'Buy' if ...

        normalized_output = str(f'Теоретическая прибыльность при торговле в long, с {str(datetime.date(start))}: {str(round(verdict_whole_period * 10, 1))}%\n' +
                                f'Прибыльность за период с {str(datetime.date(start_2))}: {str(round(verdict_whole_period_2 * 10, 1))}%\n' +
                                f'Вероятность просадки стоимости акций ниже 40%: {str(round(prob_to_drop_over_40 * 100, 2))}' +
                                f'%\nТекущий уровень прибыльности при торговле в long: {str(buy_now_decision_1)}%\nТекущая стоимость акции: ~${str(round(current_rate, 2))}')

        # Plot the test predictions
        plt.clf()  # clear the previous requested content
        plt.plot(data_2['ma50'], color="green", label="за 50 дней")
        plt.plot(data_2['ma10'], color="orange", label="за 10 дней")
        plt.plot(data_2['ma5'], color="red", label="за 5 дней")
        plt.plot(data_2['Close'], color="black", label="Цена акции (Закрытие)")
        plt.title(f"{ticker} - График средних скользящих")
        # plt.xlabel(f"Time")
        plt.ylabel(f"$")
        plt.legend()

        sunalt = plt.gcf() # get current figure
        buf = io.BytesIO()
        sunalt.savefig(buf, format='png')
        buf.seek(0)
        buffer_image = buf

        print(f'Done, for {ticker}: {tickers_list.index(ticker) + 1} out of {len(tickers_list)}')

    except ValueError:
        pre_fail_text = f"Value Error, skipped: {ticker}"
        print(pre_fail_text)
        pass

    except IndexError:
        pre_fail_text = f"Index Error, skipped: {ticker}"
        print(pre_fail_text)
        pass

    return normalized_output, buffer_image


def get_sentiment_analysis(ticker):
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    # Парсинг контента по тикеру
    url = finwiz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
    response = urlopen(req)
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    # Add the table to our dictionary
    news_tables[ticker] = news_table

    parsed_news = []
    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text()
            # splite text in the td tag into a list
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element
            if len(date_scrape) == 1:
                time = date_scrape[0]
            # else load 'date' as the 1st element and 'time' as the second
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'
            ticker = file_name.split('_')[0]
            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, text])

    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']
    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    parsed_and_scored_news_list = parsed_and_scored_news.head().values.tolist()[0]
    latest_news_date = parsed_and_scored_news_list[1]
    latest_news = parsed_and_scored_news_list[2]
    average_score = parsed_and_scored_news_list[-1]
    print(f'Latest_news_date: {latest_news_date} - {type(latest_news_date)} ; latest news: {latest_news} - {type(latest_news)}, average score: {average_score} - {type(average_score)}')
    latest_feedback = str(f'Крайняя новость, на дату {str(latest_news_date)}: {str(latest_news)}\nСредняя оценка: {str(average_score)}')

    # showing the data on chart
    plt.clf()  # clear the previous requested content
    plt.rcParams['figure.figsize'] = [10, 6]
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.groupby(['ticker', 'date']).mean()
    # Unstack the column ticker
    mean_scores = mean_scores.unstack()
    # Get the cross-section of compound in the 'columns' axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()
    # Plot a bar chart with pandas
    mean_scores.plot(kind='bar')
    plt.title(f"{ticker} - Рейтинг новостей")
    plt.grid()

    sunalt = plt.gcf()  # get current figure
    buf = io.BytesIO()
    sunalt.savefig(buf, format='png')
    buf.seek(0)
    buffer_image = buf


    return latest_feedback, buffer_image


# log level
logging.basicConfig(level=logging.INFO)

# bot init
bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


# giving the ticker data
@dp.message_handler()
async def give_feedback(message: types.Message):
    try:
        split_text = message.text.split(" ")
        if (split_text[0] in tickers_list):  # and (re.match('conf', n) for n in split_text):
            chat_id = message.chat.id
            info = get_tf_companies_feedback(split_text[0])
            await message.reply(info)

            info_1 = riskAnalysis(split_text[0])
            info_11 = info_1[0]
            ma_pic = info_1[1]
            await message.reply(info_11)
            await bot.send_photo(chat_id, photo=ma_pic)

            info__sent = get_sentiment_analysis(split_text[0])
            info__sent_1 = info__sent[0]
            ma_pic_sent = info__sent[1]
            await message.reply(info__sent_1)
            await bot.send_photo(chat_id, photo=ma_pic_sent)


    except TypeError:
        print('Some TypeError happened, who knows what... ')
        pass

    except BRq:
        await message.answer("BadRequest (aiogram) happened")


# run long-polling
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
