from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pandas_datareader.data as pdr
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk, warnings, io

nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')
# if  JSONDecodeError  HAPPENS - upgrade the yfinance!

# method for image capturing in temp.memory (instead of saving)
try:
    from StringIO import StringIO  ## for Python 2
except ImportError:
    from io import StringIO


class RecommendAdvice():
    def __init__(self, req_ticker):
        self.req_ticker = req_ticker[0]

        nltk.download('vader_lexicon')
        self.finwiz_url = 'https://finviz.com/quote.ashx?t='

        # Получение списка акций  из готового листа Google Sheet
        self.CREDENTIALS_FILE = 'stock-spreadsheets-9974a749b7e4.json'
        tickers_page = '1s6uIbhIX4IYCmFYhfWgEklFqtLX95ky7GmJNRvVexeM'
        self.ranking_page = '1C_uAagRb_GV7tu8X1fbJIM9SRtH3bAcc-n61SP8muXg'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.CREDENTIALS_FILE, ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        httpAuth = credentials.authorize(httplib2.Http())  # Авторизуемся в системе
        self.service = apiclient.discovery.build('sheets', 'v4', http=httpAuth)  # Выбираем работу с таблицами и 4 версию API

        # reading data
        results = self.service.spreadsheets().values().batchGet(spreadsheetId=tickers_page, ranges='A:R', valueRenderOption='FORMATTED_VALUE', dateTimeRenderOption='FORMATTED_STRING').execute()
        sheet_values = results['valueRanges'][0]['values']
        values = sheet_values[1:]  # текущий рабочий список (весь!)

        self.tickers_list = []  # текущий рабочий список для работы с yfinance без излишек (если будет нужен, в скрипте не используется!)
        self.tickers_name_dict = {}  # словарь тикер - наименование компании
        for i in values:
            if i[-2] == 'yfinance':
                self.tickers_list.append(i[1])
                self.tickers_name_dict[i[1]] = i[2]

    # stock_market_data
    def get_fundamental_data(self, ticker):

        try:
            t_info = yf.Ticker(ticker).info

            try:
                company_name = t_info.get('shortName')
            except TypeError:
                company_name = None

            try:
                sector = t_info.get('sector')
            except TypeError:
                sector = None

            try:
                country = t_info.get('country')
            except TypeError:
                sector = None

            try:
                m_cap = round(t_info.get('marketCap') / 1000000, 2)
            except TypeError:
                m_cap = None

            try:
                enterp_val = round(t_info.get('enterpriseValue') / 1000000, 2)
            except TypeError:
                enterp_val = None

            try:
                P_S_12_m = round(t_info.get('priceToSalesTrailing12Months'), 2)
            except TypeError:
                P_S_12_m = None

            try:
                P_B = round(t_info.get('priceToBook'), 2)
            except TypeError:
                P_B = None

            try:
                marg = round(t_info.get('profitMargins'), 3)
            except TypeError:
                marg = None

            try:
                enterprToRev = t_info.get('enterpriseToRevenue')
            except TypeError:
                enterprToRev = None

            try:
                enterprToEbitda = t_info.get('enterpriseToEbitda')
            except TypeError:
                enterprToEbitda = None

            try:
                yr_div = round(t_info.get('trailingAnnualDividendYield'), 3) if t_info.get('trailingAnnualDividendYield') is not None else 0
            except TypeError:
                yr_div = None

            try:
                exDivDate = datetime.fromtimestamp(t_info.get('exDividendDate'))
            except TypeError:
                exDivDate = None

            try:
                five_yr_div_yield = t_info.get('fiveYearAvgDividendYield') if t_info.get('fiveYearAvgDividendYield') is not None else 0
            except TypeError:
                five_yr_div_yield = None

            try:
                div_date = exDivDate.strftime('%d.%m.%y')
            except AttributeError:
                div_date = 'Без дивидендов'

            try:
                FreeCashFlow = t_info.get('freeCashflow') if t_info.get('freeCashflow') is not None else 0
            except TypeError:
                FreeCashFlow = None

            try:
                DebtToEquity = t_info.get('debtToEquity') if t_info.get('debtToEquity') is not None else 0
            except TypeError:
                DebtToEquity = None

            try:
                ROA_ReturnOnAssets = t_info.get('returnOnAssets') if t_info.get('returnOnAssets') is not None else 0
            except TypeError:
                ROA_ReturnOnAssets = None

            try:
                EBITDA = t_info.get('ebitda') if t_info.get('ebitda') is not None else 0
            except TypeError:
                EBITDA = None

            try:
                TargetMedianPrice = t_info.get('targetMedianPrice') if t_info.get('targetMedianPrice') is not None else 0
            except TypeError:
                TargetMedianPrice = None

            try:
                NumberOfAnalystOpinions = t_info.get('numberOfAnalystOpinions') if t_info.get('numberOfAnalystOpinions') is not None else 0
            except TypeError:
                NumberOfAnalystOpinions = None

            try:
                Trailing_EPS_EarningsPerShare = t_info.get('trailingEps') if t_info.get('trailingEps') is not None else 0
            except TypeError:
                Trailing_EPS_EarningsPerShare = None


            final_params_fundamental = [company_name, sector, country, m_cap, enterp_val, P_S_12_m, P_B, marg, enterprToRev, enterprToEbitda, yr_div, five_yr_div_yield, div_date, FreeCashFlow, DebtToEquity, ROA_ReturnOnAssets, EBITDA, TargetMedianPrice, NumberOfAnalystOpinions, Trailing_EPS_EarningsPerShare]
            list_headers = ['Полное наименование компании', 'Сектор', 'Страна', 'Рыночная капитализация, $млн.', 'Стоимость компании, $млн.', 'P/S', 'P/B', 'Маржинальность', 'Стоимость компании / Выручка', 'Стоимость компании / EBITDA', 'Годовая дивидендная доходность', 'Див.доходность за 5 лет', 'Крайняя дата выплаты дивидендов', 'FreeCashFlow', 'DebtToEquity', 'ROA_ReturnOnAssets', 'EBITDA', 'TargetMedianPrice', 'NumberOfAnalystOpinions', 'Trailing_EPS_EarningsPerShare']

            a_dictionary = dict(zip(list_headers, final_params_fundamental))

        except IndexError:
            error_text_1 = f'IndexError issue for: {ticker}, passing'
            return error_text_1
        except ImportError:
            error_text_2 = f'ImportError issue (maybe because too mush parsing... for: {ticker}, passing'
            return error_text_2

        return a_dictionary


    def riskAnalysis(self, ticker):
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

            # определяем среднее число непрерывных положительных участков для определения оптимального числа дней для инвестирования
            values_temp_1 = data.values.tolist()
            list_of_periods_1 = []
            z = 0
            for v in values_temp_1:
                if v[-1] == 1:
                    z += 1
                else:
                    if z > 7:
                        list_of_periods_1.append(z)
                        z = 0
                    else:
                        z = 0

            sum = 0
            for i in list_of_periods_1:
                sum += i
            average_period_1 = round(sum / len(list_of_periods_1),0)

            values_temp_2 = data_2.values.tolist()
            list_of_periods_2 = []
            k = 0
            for v in values_temp_2:
                if v[-1] == 1:
                    k += 1
                else:
                    if k > 7:
                        list_of_periods_2.append(k)
                        k = 0
                    else:
                        k = 0

            sum_2 = 0
            for i in list_of_periods_2:
                sum_2 += i
            average_period_2 = round(sum_2 / len(list_of_periods_2), 0)

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

            final_params_tech = [start, start_2, verdict_whole_period, verdict_whole_period_2, prob_to_drop_over_40, buy_now_decision_1, current_rate, average_period_1, average_period_2]
            list_headers_tech = ['Период-1', 'Период-2', 'Вердикт-1', 'Вердикт-2', 'Вероятность падения', 'Текущий уровень роста в long', 'Текущий close', 'эффективный период инвестирования по 2-м годам', 'эффективный период инвестирования по 1-му году']

            tech_dictionary = dict(zip(list_headers_tech, final_params_tech))

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

            sunalt = plt.gcf()  # get current figure
            buf = io.BytesIO()
            sunalt.savefig(buf, format='png')
            buf.seek(0)
            buffer_image = buf

            print(f'{ticker}: is at the {self.tickers_list.index(ticker) + 1} position out of {len(self.tickers_list)} in parsed list for now')

            return tech_dictionary, buffer_image

        except:
            print(f"Error, skipped: {ticker}; most probably, ticker is not in my list")
            pass



    def get_sentiment_analysis(self, ticker):
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
                if x.a.has_attr('href'):
                    link = x.a['href']
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
                parsed_news.append([ticker, date, time, text, link])

        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()
        # Set column names
        columns = ['ticker', 'date', 'time', 'headline', 'href_link']
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
        latest_news_time = parsed_and_scored_news_list[2]
        latest_news_text = parsed_and_scored_news_list[3]
        latest_news_link = str(parsed_and_scored_news_list[4])
        average_score = parsed_and_scored_news_list[-1]
        # print(f'Latest_news_date: {latest_news_date} - {type(latest_news_date)} ; latest news: {latest_news} - {type(latest_news)}, average score: {average_score} - {type(average_score)}')
        latest_feedback = str(f'Крайняя новость, на дату {str(latest_news_date)}: {str(latest_news_time)}\n{str(latest_news_text)}\nСредняя оценка: {str(average_score)}')

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

        return latest_feedback, buffer_image, latest_news_link

    def Summarize(self):
        if self.req_ticker not in self.tickers_list:
            pass
        else:
            fundamental_info = self.get_fundamental_data(self.req_ticker)

            try:
                P_S = fundamental_info.get('P/S')
                if P_S is None:
                    P_S = 0
                else:
                    P_S = P_S
            except TypeError:
                P_S = 0

            try:
                P_B = fundamental_info.get('P/B')
                if P_B is None:
                    P_B = 0
                else:
                    P_B = P_B
            except TypeError:
                P_B = 0
            Sector = fundamental_info.get('Сектор')
            m_cap = fundamental_info.get('Рыночная капитализация, $млн.')
            enterp_val = fundamental_info.get('Стоимость компании, $млн.')
            FCF = fundamental_info.get('FreeCashFlow')
            DTE = fundamental_info.get('DebtToEquity')
            ROA = fundamental_info.get('ROA_ReturnOnAssets')
            EBIT = fundamental_info.get('EBITDA')
            NOA = fundamental_info.get('NumberOfAnalystOpinions')
            enterprToRev = fundamental_info.get('Стоимость компании / Выручка')
            enterprToEbitda = fundamental_info.get('Стоимость компании / EBITDA')

            if NOA is None:
                NOA = 0
            else:
                NOA = NOA
            T_EPS = fundamental_info.get('Trailing_EPS_EarningsPerShare')

            risk_analysis_data = self.riskAnalysis(self.req_ticker)

            tech_info = risk_analysis_data[0]
            period_1 = tech_info.get('Период-1')
            period_2 = tech_info.get('Период-2')
            verdict_1 = tech_info.get('Вердикт-1')
            verdict_2 = tech_info.get('Вердикт-2')
            prob_2_drop = tech_info.get('Вероятность падения')
            buy_now_decision = tech_info.get('Текущий уровень роста в long')
            current_close = tech_info.get('Текущий close')
            effective_shoulder_1 = tech_info.get('эффективный период инвестирования по 2-м годам')
            effective_shoulder_2 = tech_info.get('эффективный период инвестирования по 1-му году')

            try:
                P_E = current_close / T_EPS
            except ZeroDivisionError:
                P_E = 'нет данных'
            try:
                TMP = fundamental_info.get('TargetMedianPrice')
                EXP_G = (TMP / current_close - 1)
            except TypeError:
                TMP = 0
                EXP_G = 0

            first_part_fundamental = str(f'Полное наименование: {self.tickers_name_dict.get(self.req_ticker)}({country}); Сектор: {Sector}\nРыночная капитализация: ${m_cap}млн.\nСтоимость компании: ${enterp_val}млн.\n' +
                                         f'P/S: {P_S}; P/E: {P_E}; P/B: {P_B}\nEBITDA: {EBIT}\nСвободный поток денег: {FCF}\nСтоимость компании / Выручка: {round(enterprToRev, 2)}%\n' +
                                         f'Стоимость компании / EBITDA: {round(enterprToEbitda, 2)}%\nДолг к выручке: {round(DTE, 2)}%\nROA: {round(ROA, 2)}%\nГодовая дивидендная доходность: ~{round(yr_div * 100, 2)}%\n' +
                                         f'Див.доходность за 5 лет: {round(five_yr_div_yield * 100, 2)}%\n')


            second_part_technical = str(f'Теоретическая прибыльность при торговле в long, с {str(datetime.date(period_1))}: {str(round(verdict_1 * 10, 1))}%\n' +
                                    f'Прибыльность за период с {str(datetime.date(period_2))}: {str(round(verdict_2 * 10, 1))}%\n' +
                                    f'Вероятность просадки стоимости акций ниже 40%: {str(round(prob_2_drop * 100, 2))}' +
                                    f'%\nТекущий уровень прибыльности при торговле в long: {str(buy_now_decision)}%\nТекущая стоимость акции: ~${str(round(current_close, 2))}\n'+
                                    f'По взвешенной оценке экспертов, акция оценивается в ${round(TMP, 2)}(+{round(EXP_G * 100, 1)}%); Число экспертов по оценке: {NOA}\n' +
                                    f'%\nЭффективный период инвестирования по 2-м годам: {str(effective_shoulder_1)}дн.; по 1-му году: {str(effective_shoulder_2)дн.}')

            chart_1 = risk_analysis_data[1]

            sentiment_part = self.get_sentiment_analysis(self.req_ticker)
            sentiment_image = sentiment_part[1]

            summary = [first_part_fundamental + second_part_technical, chart_1, sentiment_image]
            return summary